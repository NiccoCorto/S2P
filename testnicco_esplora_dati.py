import numpy as np
import librosa
import os

base_dir = "/mnt/diskone-first/Mead_EMOTE_Elaborated_Pose"
file_nome = "M003_angry_1_001"

percorso_pose = os.path.join(base_dir, "pose", f"{file_nome}.npy")
percorso_audio = os.path.join(base_dir, "audio", f"{file_nome}.wav")

pose_data = np.load(percorso_pose)
print(f"dati posa:")
print(f"shape posa: {pose_data.shape}")
print(f"primi 3 frame della posa:\n{pose_data[:3]}\n")


speech_array, sampling_rate = librosa.load(percorso_audio, sr=16000)
durata_secondi = len(speech_array) / sampling_rate

print(f"dati audio")
print(f"sample rate (frequenza): {sampling_rate} Hz")
print(f"durata: {durata_secondi:.2f} secondi")
print(f"shape audio: {speech_array.shape}")