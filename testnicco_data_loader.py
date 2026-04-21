import os
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm
import librosa
from transformers import Wav2Vec2Processor

class PoseDataset(data.Dataset):
    "dataset per i 3 assi"

    def __init__(self, data_list):
        self.data = data_list
        self.len = len(self.data)

    def __getitem__(self, index):
        # ritorna una coppia audio_feature/pose_target
        file_name = self.data[index]["name"]
        audio_features = self.data[index]["audio"]
        pose_target = self.data[index]["pose"]
        return torch.FloatTensor(audio_features), torch.FloatTensor(pose_target), file_name

    def __len__(self):
        return self.len

def read_data():
    print("caricamento dati EMOTE...")
    data_list = []
    
    base_dir = "/mnt/diskone-first/Mead_EMOTE_Elaborated_Pose"
    audio_path = os.path.join(base_dir, "audio")
    pose_path = os.path.join(base_dir, "pose")
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    # solo 50 file per test
    fnames = [f for f in os.listdir(audio_path) if f.endswith('.wav')][:50]

    for f in tqdm(fnames):
        wav_file = os.path.join(audio_path, f)
        npy_file = os.path.join(pose_path, f.replace(".wav", ".npy"))
        
        if os.path.exists(npy_file):
            # carica audio e porta a 16000Hz
            speech_array, sampling_rate = librosa.load(wav_file, sr=16000)
            # Estrai le features con Wav2Vec2
            input_values = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
            
            # carica i 3 assi
            pose_data = np.load(npy_file, allow_pickle=True)
            
            # salva tutto
            data_list.append({
                "name": f,
                "audio": input_values,
                "pose": pose_data
            })

    print(f"\nCompletato! Caricati {len(data_list)} file accoppiati.")
    return data_list

def get_dataloaders(batch_size=1):
    data_list = read_data()
    dataset = PoseDataset(data_list)
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == "__main__":
    loader = get_dataloaders()
    for batch_audio, batch_pose, names in loader:
        print(f"\n--- PRIMO BATCH ESTRATTO DAL DATALOADER ---")
        print(f"File processato: {names[0]}")
        print(f"Shape Tensor Audio: {batch_audio.shape}")
        print(f"Shape Tensor Pose: {batch_pose.shape}")
        break  # fermo dopo il primo per capire se funziona (test)