import sys
import os
import numpy as np
import trimesh
import soundfile as sf
import librosa
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "Audio2Pose"))
from render_comparison import render_vertices_to_video, merge_side_by_side
from utils import apply_rotation_to_vertices
from model import HeadPosePredictor

device = "cuda" if torch.cuda.is_available() else "cpu"

sample_name = "RD_AmandaStuck_000"
chunk_idx = 0
results_dir = "Results/exp5_HDTF_3L_h256_vel2"
dataset_dir = "/mnt/diskone-second/DiffPoseTalk/datasets/HDTF_TFHP_Elaborated_Pose"

out_dir = os.path.join(results_dir, f"{sample_name}_chunk{chunk_idx}_ep200")
os.makedirs(out_dir, exist_ok=True)

# 1. Carica le features audio salvate su disco (Wav2Vec2Processor raw)
chunks_dir = "dataset_chunks"
audio_npy_path = os.path.join(chunks_dir, f"{sample_name}_chunk{chunk_idx}_audio.npy")
pose_npy_path = os.path.join(chunks_dir, f"{sample_name}_chunk{chunk_idx}_pose.npy")

chunk_audio = np.load(audio_npy_path)
chunk_pose_gt = np.load(pose_npy_path)

print(f"Elaborazione Chunk: {sample_name} - Chunk {chunk_idx}")
print(f"Caricamento modello Epoca 200...")

# 2. Inferenza con il modello all'Epoca 200
checkpoint_path = f"Saves/exp5_HDTF_3L_h256_vel2/audio2pose_final_epoch_200.pth"
sd = torch.load(checkpoint_path, map_location=device)

class Args: pass
args = Args()
args.num_layers = 3
args.hidden_dim = 256
args.dropout = 0.0

model = HeadPosePredictor(args)
model.load_state_dict(sd)
model.to(device)
model.eval()

audio_tensor = torch.FloatTensor(chunk_audio).unsqueeze(0).to(device)
pose_len_tensor = torch.LongTensor([chunk_pose_gt.shape[0]]).to(device)

with torch.no_grad():
    # Passiamo anche pose_lengths per allineare correttamente la shape
    pred_pose_tensor = model(audio_tensor, pose_lengths=pose_len_tensor)
    predicted_pose = pred_pose_tensor.squeeze(0).cpu().numpy()

# Salva il file .npy predetto
np.save(os.path.join(out_dir, "predicted_pose.npy"), predicted_pose)
print(f"Posa predetta salvata. Shape: {predicted_pose.shape}")

# 3. Carica vertici dal dataset per il rendering
audio_path = os.path.join(dataset_dir, "audio", f"{sample_name}.wav")
vertices_path = os.path.join(dataset_dir, "vertices", f"{sample_name}.npy")
vertices_pose_path = os.path.join(dataset_dir, "vertices_pose", f"{sample_name}.npy")
template_path = "../ScanTalk/src/examples/FLAME_sample.ply"

vertices_static = np.load(vertices_path)
vertices_gt = np.load(vertices_pose_path)

fps = 25
n_frames = predicted_pose.shape[0]

# Poiché è il chunk 0, prende semplicemente i primi n_frames
vertices_static = vertices_static[:n_frames]
vertices_gt = vertices_gt[:n_frames]

# Ritaglia audio per ffmpeg
audio_data, sr = librosa.load(audio_path, sr=16000)
expected_samples = round(n_frames / fps * sr)
audio_trimmed = audio_data[:expected_samples]
trimmed_audio_path = os.path.join(out_dir, "trimmed_audio.wav")
sf.write(trimmed_audio_path, audio_trimmed, sr)

# 4. Applica rotazioni predette ai vertici statici
print("Applicazione rotazioni ai vertici...")
vertices_rotated = np.zeros((n_frames, vertices_static.shape[1], 3))
for i in range(n_frames):
    vertices_rotated[i] = apply_rotation_to_vertices(
        vertices_static[i], predicted_pose[i], use_origin=True
    )

# 5. Rendering Video
print("Rendering in corso...")
template = trimesh.load(template_path, process=False)
faces = np.array(template.faces)

gt_video_path = os.path.join(out_dir, "gt_video.mp4")
pred_video_path = os.path.join(out_dir, "pred_video.mp4")
confronto_path = os.path.join(out_dir, f"confronto_{sample_name}_chunk{chunk_idx}_ep200.mp4")

print(" rendering GT video...")
render_vertices_to_video(vertices_gt, faces, trimmed_audio_path, gt_video_path, fps=fps)

print(" rendering Prediction video...")
render_vertices_to_video(vertices_rotated, faces, trimmed_audio_path, pred_video_path, fps=fps)

print(" Unione affiancata...")
merge_side_by_side(gt_video_path, pred_video_path, confronto_path, label_left="GT", label_right="Pred_Ep200")
print(f"Completato! Video salvato in: {confronto_path}")
