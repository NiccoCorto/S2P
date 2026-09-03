import sys
import os
import glob
import numpy as np
import trimesh
import soundfile as sf
import librosa

sys.path.append(os.path.join(os.path.dirname(__file__), "Audio2Pose"))
from render_comparison import render_vertices_to_video, merge_side_by_side
from utils import apply_rotation_to_vertices

sample_name = "WDA_BarackObama_000"
results_dir = "Results/exp5_HDTF_3L_h256_vel2"
dataset_dir = "/mnt/diskone-second/DiffPoseTalk/datasets/HDTF_TFHP_Elaborated_Pose"

# 1. Trova e ordina i chunk
chunk_pattern = os.path.join(results_dir, f"{sample_name}_chunk*.npy")
chunk_files = sorted(glob.glob(chunk_pattern), key=lambda x: int(x.split("_chunk")[-1].replace(".npy", "")))

if not chunk_files:
    print(f"Nessun chunk trovato per {sample_name}")
    sys.exit(1)

print("Concatenazione dei seguenti chunk:")
pose_list = []
for cf in chunk_files:
    print(f" - {cf}")
    pose_list.append(np.load(cf))

predicted_pose = np.concatenate(pose_list, axis=0)
print(f"Shape totale delle pose predette: {predicted_pose.shape}")

# 2. Carica i dati interi dal dataset
audio_path = os.path.join(dataset_dir, "audio", f"{sample_name}.wav")
vertices_path = os.path.join(dataset_dir, "vertices", f"{sample_name}.npy")
vertices_pose_path = os.path.join(dataset_dir, "vertices_pose", f"{sample_name}.npy")
template_path = "../ScanTalk/src/examples/FLAME_sample.ply"

vertices_static = np.load(vertices_path)
vertices_gt = np.load(vertices_pose_path)
print(f"Shape vertici statici (base): {vertices_static.shape}")
print(f"Shape vertici GT: {vertices_gt.shape}")

# 3. Allinea le lunghezze al minimo comun denominatore
fps = 25  # HDTF_TFHP usa 25 fps come da paper
n_pose = predicted_pose.shape[0]
n_gt = vertices_gt.shape[0]
n_frames = min(n_pose, n_gt)
print(f"Utilizzo di {n_frames} frame per il rendering (minimo tra GT e Pose).")

predicted_pose = predicted_pose[:n_frames]
vertices_static = vertices_static[:n_frames]
vertices_gt = vertices_gt[:n_frames]

# 4. Ritaglia l'audio alla stessa lunghezza
audio_data, sr = librosa.load(audio_path, sr=16000)
expected_samples = round(n_frames / fps * sr)
audio_trimmed = audio_data[:expected_samples]
out_dir = os.path.join(results_dir, sample_name)
os.makedirs(out_dir, exist_ok=True)
trimmed_audio_path = os.path.join(out_dir, "trimmed_audio.wav")
sf.write(trimmed_audio_path, audio_trimmed, sr)

# 5. Applica le rotazioni predette ai vertici statici
print("Applicazione delle rotazioni ai vertici statici...")
vertices_rotated = np.zeros((n_frames, vertices_static.shape[1], 3))
for i in range(n_frames):
    vertices_rotated[i] = apply_rotation_to_vertices(
        vertices_static[i], predicted_pose[i], use_origin=True
    )

# 6. Renderizza i video
print("Caricamento mesh template...")
template = trimesh.load(template_path, process=False)
faces = np.array(template.faces)

gt_video_path = os.path.join(out_dir, "gt_video.mp4")
pred_video_path = os.path.join(out_dir, "pred_video.mp4")
confronto_path = os.path.join(out_dir, f"confronto_{sample_name}.mp4")

print("Rendering video Ground Truth...")
render_vertices_to_video(vertices_gt, faces, trimmed_audio_path, gt_video_path, fps=fps)

print("Rendering video Prediction (Audio2Pose)...")
render_vertices_to_video(vertices_rotated, faces, trimmed_audio_path, pred_video_path, fps=fps)

print("Unione affiancata dei video...")
merge_side_by_side(gt_video_path, pred_video_path, confronto_path, label_left="GT", label_right="Pred(vel2)")
print(f"Finito! Risultato finale salvato in: {confronto_path}")
