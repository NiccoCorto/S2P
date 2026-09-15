"""
data_loader.py - Dataloader per S2P (Speech-to-Pose)
Carica audio (WAV) e pose (NPY con 3 angoli di rotazione) dal dataset EMOTE.
Segue la logica di s2l-s2d: split per soggetto, Wav2Vec2 per feature audio.
Ora implementa Lazy Loading tramite Disk Chunking per evitare System OOM.

Branch VarLoss:
- speaker_mapping.json viene costruito SOLO dalle identità presenti nel training set.
  Le identità di val/test non viste in train ricevono speaker_id = -1, che il modello
  gestirà con un vettore one-hot uniforme (prior 1/N), senza contaminare il mapping.
"""
import os
import pickle
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import json
import librosa
from scipy.signal import savgol_filter

def extract_identity(filename):
    """Estrae l'identità pura (es. AmandaStuck) dal nome del file audio."""
    parts = filename.replace(".wav", "").split('_')
    if parts[0] in ['RD', 'TH', 'WDA', 'WRA']:  # Format HDTF (es. RD_AmandaStuck_000)
        return parts[1]
    else:  # Format EMOTE (es. M003_contempt_2_022)
        return parts[0]

class PoseDataset(data.Dataset):
    """Dataset personalizzato per i 3 assi di rotazione (Pitch, Yaw, Roll).
    Usa il lazy loading per limitare la memoria RAM consumata."""

    def __init__(self, data_list, data_type="train"):
        self.data = data_list
        self.len = len(self.data)
        self.data_type = data_type

    def __getitem__(self, index):
        """Ritorna (Audio_features, Posa_target, nome_file, speaker_id).
        Carica i dati da disco solo in questo momento."""
        item = self.data[index]
        file_name = item["name"]
        speaker_id = item.get("speaker_id", -1)  # -1 = identità non vista in train
        
        # Lazy loading
        audio_features = np.load(item["audio_file"])
        pose_target = np.load(item["pose_file"])

        return torch.FloatTensor(audio_features), torch.FloatTensor(pose_target), file_name, speaker_id

    def __len__(self):
        return self.len


def read_data(args):
    """Carica e prepara i dati dal dataset.
    
    Branch VarLoss: il speaker_mapping.json viene costruito SOLO dalle identità
    di training, garantendo che il one-hot encoding non venga "contaminato" da
    identità di val/test. I soggetti non visti ricevono speaker_id = -1.
    
    Args:
        args: Namespace con almeno audio_path, pose_path, max_samples,
              cache_data, train_split, val_split.
    
    Returns:
        train_data, valid_data, test_data, subjects_dict
    """
    _project_dir = os.path.dirname(os.path.abspath(__file__))
    chunks_dir = os.path.join(_project_dir, "dataset_chunks")
    cache_file = os.path.join(_project_dir, "s2p_lazy_cache.pkl")
    
    if args.cache_data and os.path.exists(cache_file):
        print(f"Caricamento dati (metadata) da cache lazy: {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        print(f"Cache caricata - Train: {len(cached['train'])}, "
              f"Val: {len(cached['valid'])}, Test: {len(cached['test'])}")
        return cached["train"], cached["valid"], cached["test"], cached["subjects_dict"]

    print("Caricamento dati in corso e generazione chunk su disco...")
    print(f"  Audio: {args.audio_path}")
    print(f"  Pose:  {args.pose_path}")
    print(f"  Cartella chunks: {chunks_dir}")

    os.makedirs(chunks_dir, exist_ok=True)

    data_dict = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []
    skipped = 0

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    if not os.path.isdir(args.audio_path):
        raise FileNotFoundError(f"Cartella audio non trovata: {args.audio_path}")
    if not os.path.isdir(args.pose_path):
        raise FileNotFoundError(f"Cartella pose non trovata: {args.pose_path}")

    fs = sorted([f for f in os.listdir(args.audio_path) if f.endswith('.wav')])
    print(f"  Trovati {len(fs)} file audio")

    if args.max_samples is not None:
        fs = fs[:args.max_samples]
        print(f"  Limitato a {args.max_samples} campioni (--max_samples)")

    # -----------------------------------------------------------------------
    # Costruzione split PRIMA di creare il speaker_mapping
    # -----------------------------------------------------------------------
    splits_dir = os.path.join(args.data_dir, "splits")
    use_split_files = os.path.isdir(splits_dir)

    if use_split_files:
        print(f"\n  Trovata cartella splits in {splits_dir}. Lettura file txt...")
        def read_split_file(filename):
            path = os.path.join(splits_dir, filename)
            if not os.path.exists(path):
                return []
            with open(path, "r") as f:
                return [line.strip() for line in f if line.strip()]

        train_keys = set(read_split_file("train.txt"))
        val_keys   = set(read_split_file("val.txt"))
        test_keys  = set(read_split_file("test.txt"))

        subjects_dict = {
            "train": list(train_keys),
            "val":   list(val_keys),
            "test":  list(test_keys)
        }

        # Identità di training: solo dai file di train split
        train_identities = sorted(set(
            extract_identity(base_k.split("/")[-1] + ".wav")
            for base_k in train_keys
        ))

    else:
        # Split automatico per soggetto
        all_subjects = sorted(list(set([f.split("_")[0] for f in fs])))
        print(f"\n  Soggetti trovati: {len(all_subjects)} → {all_subjects}")

        n_train = int(len(all_subjects) * args.train_split)
        n_val   = int(len(all_subjects) * args.val_split)

        subjects_dict = {
            "train": all_subjects[:n_train],
            "val":   all_subjects[n_train:n_train + n_val],
            "test":  all_subjects[n_train + n_val:]
        }

        train_identities = subjects_dict["train"]
        print(f"  Split soggetti Train: {subjects_dict['train']}, "
              f"Val: {subjects_dict['val']}, Test: {subjects_dict['test']}")

    # -----------------------------------------------------------------------
    # speaker_mapping costruito SOLO sulle identità di train
    # -----------------------------------------------------------------------
    speaker_mapping_file = os.path.join(_project_dir, "speaker_mapping.json")
    identity_to_id = {ident: i for i, ident in enumerate(train_identities)}

    with open(speaker_mapping_file, "w") as f_map:
        json.dump(identity_to_id, f_map, indent=4)
    print(f"  Trovate {len(identity_to_id)} identità di TRAIN. "
          f"Mappatura salvata in speaker_mapping.json")
    print(f"  [INFO] Identità di val/test non viste in train riceveranno speaker_id=-1 "
          f"(vettore uniforme 1/N nel modello)")

    # -----------------------------------------------------------------------
    # Preprocessing e chunking
    # -----------------------------------------------------------------------
    for f in tqdm(fs, desc="Preprocessing e salvataggio chunk su disco"):
        wav_file = os.path.join(args.audio_path, f)
        npy_file = os.path.join(args.pose_path, f.replace(".wav", ".npy"))

        if not os.path.exists(npy_file):
            skipped += 1
            continue

        base_key = f.replace(".wav", "")
        identity  = extract_identity(f)

        try:
            # carica audio e porta a 16000Hz
            speech_array, sampling_rate = librosa.load(wav_file, sr=16000)

            # carica i 3 angoli di rotazione per frame
            pose_data = np.load(npy_file, allow_pickle=True)

            if pose_data.ndim != 2 or pose_data.shape[1] != 3:
                print(f"  [WARN] Shape inattesa per {f}: {pose_data.shape}, skip")
                skipped += 1
                continue

            if pose_data.shape[0] > 9:
                pose_data = savgol_filter(pose_data, window_length=9, polyorder=2, axis=0)

            input_values = np.squeeze(
                processor(speech_array, sampling_rate=16000).input_values
            )

            audio_samples = len(input_values)
            pose_frames   = len(pose_data)
            
            chunk_seconds = 20
            audio_chunk   = chunk_seconds * 16000
            pose_chunk    = int(chunk_seconds * (pose_frames / (audio_samples / 16000.0)))
            num_chunks    = int(np.ceil(audio_samples / audio_chunk))

            for i in range(num_chunks):
                start_a = i * audio_chunk
                end_a   = min((i + 1) * audio_chunk, audio_samples)

                start_p = i * pose_chunk
                end_p   = min((i + 1) * pose_chunk, pose_frames)

                chunk_audio = input_values[start_a:end_a]
                chunk_pose  = pose_data[start_p:end_p]

                if len(chunk_audio) < 2 * 16000:
                    continue

                key_chunk  = f"{base_key}_chunk{i}"
                audio_file = os.path.join(chunks_dir, f"{key_chunk}_audio.npy")
                pose_file  = os.path.join(chunks_dir, f"{key_chunk}_pose.npy")

                if not (os.path.exists(audio_file) and os.path.exists(pose_file)):
                    np.save(audio_file, chunk_audio)
                    np.save(pose_file, chunk_pose)

                # speaker_id: -1 se identità non vista in training set
                spk_id = identity_to_id.get(identity, -1)

                data_dict[key_chunk]["name"]       = key_chunk
                data_dict[key_chunk]["audio_file"] = audio_file
                data_dict[key_chunk]["pose_file"]  = pose_file
                data_dict[key_chunk]["base_key"]   = base_key
                data_dict[key_chunk]["speaker_id"] = spk_id

        except Exception as e:
            print(f"  [ERR] Errore processando {f}: {e}")
            skipped += 1
            continue

    if skipped > 0:
        print(f"  Saltati {skipped} file (mancanti o con errori)")

    if len(data_dict) == 0:
        raise RuntimeError("Nessun dato valido caricato. Controlla i percorsi.")

    # -----------------------------------------------------------------------
    # Assegnazione agli split
    # -----------------------------------------------------------------------
    if use_split_files:
        for k, v in data_dict.items():
            base_k = v["base_key"]
            if base_k in train_keys:
                train_data.append(v)
            elif base_k in val_keys:
                valid_data.append(v)
            elif base_k in test_keys:
                test_data.append(v)
            else:
                print(f"  [WARN] File base {base_k} (chunk {k}) non presente in nessuno split.")
    else:
        for k, v in data_dict.items():
            subject_id = v["base_key"].split("_")[0]
            if subject_id in subjects_dict["train"]:
                train_data.append(v)
            elif subject_id in subjects_dict["val"]:
                valid_data.append(v)
            elif subject_id in subjects_dict["test"]:
                test_data.append(v)

    print(f"\nDati caricati Train: {len(train_data)}, "
          f"Validation: {len(valid_data)}, Test: {len(test_data)}")

    if train_data:
        sample   = train_data[0]
        ex_audio = np.load(sample['audio_file'])
        ex_pose  = np.load(sample['pose_file'])
        print(f"  Esempio — Audio shape: {ex_audio.shape}, "
              f"Pose shape: {ex_pose.shape}")

    if args.cache_data:
        print(f"Salvataggio cache metadata in: {cache_file}")
        with open(cache_file, "wb") as f_cache:
            pickle.dump({
                "train":        train_data,
                "valid":        valid_data,
                "test":         test_data,
                "subjects_dict": subjects_dict
            }, f_cache)

    return train_data, valid_data, test_data, subjects_dict


def collate_fn(batch):
    """Funzione di collation custom per gestire sequenze audio e pose di lunghezza variabile.

    Necessaria per supportare batch_size > 1: le sequenze audio (raw waveform a 16kHz)
    e le sequenze pose (N_frames, 3) hanno lunghezze diverse tra campioni.
    La strategia è il padding a zero alla lunghezza massima del batch.

    Vengono restituite anche le lunghezze reali (pre-padding) di audio e pose, così che
    la loss e il modello possano ignorare i frame paddati (evitando loss poisoning).

    Args:
        batch: lista di tuple (audio_tensor, pose_tensor, file_name, speaker_id)

    Returns:
        padded_audio:  Tensor (B, max_audio_len)    — audio paddato a zero
        padded_pose:   Tensor (B, max_pose_len, 3)  — pose paddate a zero
        pose_lengths:  LongTensor (B,)              — lunghezza reale di ogni sequenza pose
        audio_lengths: LongTensor (B,)              — lunghezza reale di ogni sequenza audio
        names:         list[str]                    — nomi dei file nel batch
        speaker_ids:   LongTensor (B,)              — speaker IDs (-1 = sconosciuto)
    """
    audios, poses, names, speaker_ids = zip(*batch)

    # lunghezze reali (prima del padding) — usate come mask in loss e in model.forward()
    pose_lengths  = torch.tensor([p.shape[0] for p in poses],  dtype=torch.long)
    audio_lengths = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)

    # padding audio alla lunghezza massima nel batch
    max_audio_len = int(audio_lengths.max().item())
    padded_audio  = torch.zeros(len(audios), max_audio_len)
    for i, a in enumerate(audios):
        padded_audio[i, :a.shape[0]] = a

    # padding pose alla lunghezza massima nel batch
    max_pose_len = int(pose_lengths.max().item())
    padded_pose  = torch.zeros(len(poses), max_pose_len, 3)
    for i, p in enumerate(poses):
        padded_pose[i, :p.shape[0], :] = p

    speaker_ids_tensor = torch.tensor(speaker_ids, dtype=torch.long)

    return padded_audio, padded_pose, pose_lengths, audio_lengths, list(names), speaker_ids_tensor


def get_dataloaders(args):
    """Crea i DataLoader per train, validation e test.
    
    Args:
        args: Namespace con batch_size e tutti i parametri di read_data.
    
    Returns:
        dict con chiavi "train", "valid", "test" DataLoader
    """
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)

    batch_size = getattr(args, "batch_size", 1)

    # dataloader per il train (con shuffle=True e collate_fn per batch_size > 1)
    train_dataset = PoseDataset(train_data, "train")
    dataset["train"] = data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn
    )

    # dataloader per validation e test (shuffle=False, collate_fn per consistenza)
    valid_dataset = PoseDataset(valid_data, "val")
    dataset["valid"] = data.DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn
    )

    test_dataset = PoseDataset(test_data, "test")
    dataset["test"] = data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn
    )

    return dataset


if __name__ == "__main__":
    # test standalone del dataloader
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import get_args

    args = get_args()
    loaders = get_dataloaders(args)

    # verifichiamo cosa c'è dentro il loader di train
    for batch_audio, batch_pose, pose_lengths, audio_lengths, names, speaker_ids in loaders["train"]:
        print(f"\n PRIMO BATCH ESTRATTO DAL DATALOADER (TRAIN)")
        print(f"File processato: {names[0]}")
        print(f"Shape Tensor Audio: {batch_audio.shape}  (audio_lengths: {audio_lengths.tolist()})")
        print(f"Shape Tensor Pose:  {batch_pose.shape}  (pose_lengths: {pose_lengths.tolist()})")
        print(f"Speaker IDs: {speaker_ids.tolist()}")
        print(f"Primi 3 frame posa: \n{batch_pose[0, :3, :]}")
        break