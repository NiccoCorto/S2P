import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa

class PoseDataset(data.Dataset):
    """Dataset personalizzato per i 3 assi (simile alla struttura S2L)."""

    def __init__(self, data_list, data_type="train"):
        self.data = data_list
        self.len = len(self.data)
        self.data_type = data_type

    def __getitem__(self, index):
        """Ritorna una coppia (Audio_features, Posa_target)."""
        file_name = self.data[index]["name"]
        audio_features = self.data[index]["audio"]
        pose_target = self.data[index]["pose"]

        return torch.FloatTensor(audio_features), torch.FloatTensor(pose_target), file_name

    def __len__(self):
        return self.len


def read_data(args=None):
    print("Caricamento dati EMOTE in corso...")
    data_dict = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []
    
    # Percorsi base
    base_dir = "/mnt/diskone-first/Mead_EMOTE_Elaborated_Pose"
    audio_path = os.path.join(base_dir, "audio")
    pose_path = os.path.join(base_dir, "pose")
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    # Troviamo tutti i file audio
    fs = [f for f in os.listdir(audio_path) if f.endswith('.wav')]

    # i primi test fino a 100
    fs = fs[:100] 

    for f in tqdm(fs):
        wav_file = os.path.join(audio_path, f)
        npy_file = os.path.join(pose_path, f.replace(".wav", ".npy"))
        
        if os.path.exists(npy_file):
            # Carica audio e porta a 16000Hz
            speech_array, sampling_rate = librosa.load(wav_file, sr=16000)
            # Estrai le features con Wav2Vec2
            input_values = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
            
            # Carica i 3 assi
            pose_data = np.load(npy_file, allow_pickle=True)
            
            # Chiave del dizionario (es. M003_angry_1_001)
            key = f.replace(".wav", "")
            
            data_dict[key]["name"] = f
            data_dict[key]["audio"] = input_values
            data_dict[key]["pose"] = pose_data

    # --- LOGICA DI SPLIT (Come S2L) ---
    # Estraiamo tutti i soggetti unici dai nomi dei file (es. M003, M012, ecc.)
    subjects = list(set([k.split("_")[0] for k in data_dict.keys()]))
    subjects.sort()

    # Dividiamo i soggetti: 80% Train, 10% Validation, 10% Test
    n_train = int(len(subjects) * 0.8)
    n_val = int(len(subjects) * 0.1)

    subjects_dict = {}
    subjects_dict["train"] = subjects[:n_train]
    subjects_dict["val"] = subjects[n_train:n_train + n_val]
    subjects_dict["test"] = subjects[n_train + n_val:]

    # Assegniamo i file alle liste corrette in base al soggetto
    for k, v in data_dict.items():
        subject_id = k.split("_")[0]
        if subject_id in subjects_dict["train"]:
            train_data.append(v)
        elif subject_id in subjects_dict["val"]:
            valid_data.append(v)
        elif subject_id in subjects_dict["test"]:
            test_data.append(v)

    print(f"\nDati caricati - Train: {len(train_data)}, Validation: {len(valid_data)}, Test: {len(test_data)}")
    return train_data, valid_data, test_data, subjects_dict


def get_dataloaders(args=None):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    
    # Dataloader per il Train (con shuffle=True per mescolare i dati)
    train_dataset = PoseDataset(train_data, "train")
    dataset["train"] = data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    
    # Dataloader per Validation e Test (shuffle=False)
    valid_dataset = PoseDataset(valid_data, "val")
    dataset["valid"] = data.DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)
    
    test_dataset = PoseDataset(test_data, "test")
    dataset["test"] = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    
    return dataset


if __name__ == "__main__":
    # Test del dataloader
    loaders = get_dataloaders()
    
    # Verifichiamo cosa c'è dentro il loader di Train
    for batch_audio, batch_pose, names in loaders["train"]:
        print(f"\n--- PRIMO BATCH ESTRATTO DAL DATALOADER (TRAIN) ---")
        print(f"File processato: {names[0]}")
        print(f"Shape Tensor Audio: {batch_audio.shape}")
        print(f"Shape Tensor Pose: {batch_pose.shape}")
        break