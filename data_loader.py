"""
data_loader.py - Dataloader per S2P (Speech-to-Pose)
Carica audio (WAV) e pose (NPY con 3 angoli di rotazione) dal dataset EMOTE.
Segue la logica di s2l-s2d: split per soggetto, Wav2Vec2 per feature audio.
"""
import os
import pickle
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa


class PoseDataset(data.Dataset):
    """Dataset personalizzato per i 3 assi di rotazione (Pitch, Yaw, Roll).
    Struttura ispirata a S2L/data_loader.py."""

    def __init__(self, data_list, data_type="train"):
        self.data = data_list
        self.len = len(self.data)
        self.data_type = data_type

    def __getitem__(self, index):
        """Ritorna una coppia (Audio_features, Posa_target, nome_file)."""
        file_name = self.data[index]["name"]
        audio_features = self.data[index]["audio"]
        pose_target = self.data[index]["pose"]

        return torch.FloatTensor(audio_features), torch.FloatTensor(pose_target), file_name

    def __len__(self):
        return self.len


def read_data(args):
    """Carica e prepara i dati dal dataset EMOTE.
    
    Args:
        args: Namespace con almeno audio_path, pose_path, max_samples,
              cache_data, train_split, val_split.
    
    Returns:
        train_data, valid_data, test_data, subjects_dict
    """
    # --- Cache: se esiste e richiesta, carica da file ---
    cache_file = os.path.join(args.data_dir, "s2p_cache.pkl")
    if args.cache_data and os.path.exists(cache_file):
        print(f"Caricamento dati da cache: {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        print(f"Cache caricata - Train: {len(cached['train'])}, "
              f"Val: {len(cached['valid'])}, Test: {len(cached['test'])}")
        return cached["train"], cached["valid"], cached["test"], cached["subjects_dict"]

    print("Caricamento dati EMOTE in corso...")
    print(f"  Audio: {args.audio_path}")
    print(f"  Pose:  {args.pose_path}")

    data_dict = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []
    skipped = 0

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    # Troviamo tutti i file audio
    if not os.path.isdir(args.audio_path):
        raise FileNotFoundError(f"Cartella audio non trovata: {args.audio_path}")
    if not os.path.isdir(args.pose_path):
        raise FileNotFoundError(f"Cartella pose non trovata: {args.pose_path}")

    fs = sorted([f for f in os.listdir(args.audio_path) if f.endswith('.wav')])
    print(f"  Trovati {len(fs)} file audio")

    # Limita il numero di campioni se richiesto (per test veloci)
    if args.max_samples is not None:
        fs = fs[:args.max_samples]
        print(f"  Limitato a {args.max_samples} campioni (--max_samples)")

    for f in tqdm(fs, desc="Preprocessing audio+pose"):
        wav_file = os.path.join(args.audio_path, f)
        npy_file = os.path.join(args.pose_path, f.replace(".wav", ".npy"))

        if not os.path.exists(npy_file):
            skipped += 1
            continue

        try:
            # Carica audio e porta a 16000Hz
            speech_array, sampling_rate = librosa.load(wav_file, sr=16000)
            # Estrai le features con Wav2Vec2
            input_values = np.squeeze(
                processor(speech_array, sampling_rate=16000).input_values
            )

            # Carica i 3 angoli di rotazione per frame
            pose_data = np.load(npy_file, allow_pickle=True)

            # Verifica shape: deve essere (N_frames, 3)
            if pose_data.ndim != 2 or pose_data.shape[1] != 3:
                print(f"  [WARN] Shape inattesa per {f}: {pose_data.shape}, skip")
                skipped += 1
                continue

            # Chiave del dizionario (es. M003_angry_1_001)
            key = f.replace(".wav", "")

            data_dict[key]["name"] = f
            data_dict[key]["audio"] = input_values
            data_dict[key]["pose"] = pose_data

        except Exception as e:
            print(f"  [ERR] Errore processando {f}: {e}")
            skipped += 1
            continue

    if skipped > 0:
        print(f"  Saltati {skipped} file (mancanti o con errori)")

    if len(data_dict) == 0:
        raise RuntimeError("Nessun dato valido caricato! Controlla i percorsi.")

    # --- LOGICA DI SPLIT (Come S2L): split per soggetto ---
    # Estraiamo tutti i soggetti unici dai nomi dei file (es. M003, M012, ecc.)
    subjects = sorted(list(set([k.split("_")[0] for k in data_dict.keys()])))
    print(f"\n  Soggetti trovati: {len(subjects)} → {subjects}")

    # Dividiamo i soggetti: es. 80% Train, 10% Validation, 10% Test
    n_train = int(len(subjects) * args.train_split)
    n_val = int(len(subjects) * args.val_split)

    subjects_dict = {
        "train": subjects[:n_train],
        "val": subjects[n_train:n_train + n_val],
        "test": subjects[n_train + n_val:]
    }

    print(f"  Split soggetti → Train: {subjects_dict['train']}, "
          f"Val: {subjects_dict['val']}, Test: {subjects_dict['test']}")

    # Assegniamo i file alle liste corrette in base al soggetto
    for k, v in data_dict.items():
        subject_id = k.split("_")[0]
        if subject_id in subjects_dict["train"]:
            train_data.append(v)
        elif subject_id in subjects_dict["val"]:
            valid_data.append(v)
        elif subject_id in subjects_dict["test"]:
            test_data.append(v)

    print(f"\nDati caricati — Train: {len(train_data)}, "
          f"Validation: {len(valid_data)}, Test: {len(test_data)}")

    # --- Diagnostica shape ---
    if train_data:
        sample = train_data[0]
        print(f"  Esempio — Audio shape: {sample['audio'].shape}, "
              f"Pose shape: {sample['pose'].shape}")

    # --- Salva cache se richiesto ---
    if args.cache_data:
        print(f"Salvataggio cache in: {cache_file}")
        with open(cache_file, "wb") as f_cache:
            pickle.dump({
                "train": train_data,
                "valid": valid_data,
                "test": test_data,
                "subjects_dict": subjects_dict
            }, f_cache)

    return train_data, valid_data, test_data, subjects_dict


def get_dataloaders(args):
    """Crea i DataLoader per train, validation e test.
    
    Args:
        args: Namespace con batch_size e tutti i parametri di read_data.
    
    Returns:
        dict con chiavi "train", "valid", "test" → DataLoader
    """
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)

    batch_size = getattr(args, "batch_size", 1)

    # Dataloader per il Train (con shuffle=True per mescolare i dati)
    train_dataset = PoseDataset(train_data, "train")
    dataset["train"] = data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    # Dataloader per Validation e Test (shuffle=False)
    valid_dataset = PoseDataset(valid_data, "val")
    dataset["valid"] = data.DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=False
    )

    test_dataset = PoseDataset(test_data, "test")
    dataset["test"] = data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    return dataset


if __name__ == "__main__":
    # Test standalone del dataloader
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import get_args

    args = get_args()
    loaders = get_dataloaders(args)

    # Verifichiamo cosa c'è dentro il loader di Train
    for batch_audio, batch_pose, names in loaders["train"]:
        print(f"\n--- PRIMO BATCH ESTRATTO DAL DATALOADER (TRAIN) ---")
        print(f"File processato: {names[0]}")
        print(f"Shape Tensor Audio: {batch_audio.shape}")
        print(f"Shape Tensor Pose:  {batch_pose.shape}")
        print(f"Primi 3 frame posa: \n{batch_pose[0, :3, :]}")
        break