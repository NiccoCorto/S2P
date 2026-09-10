import os
import sys
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import soundfile as sf

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Audio2Pose"))
from model import HeadPosePredictor
import evaluate
from render_comparison import render_vertices_to_video, merge_side_by_side
import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--sample_chunk", type=str, default="RD_AmandaStuck_000_chunk0")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp_name = args.exp_name
    sample = args.sample_chunk
    
    res_dir = os.path.join(base_dir, "Results", exp_name, f"{sample}_ep200")
    quant_dir = os.path.join(res_dir, "Quantitative")
    os.makedirs(quant_dir, exist_ok=True)
    
    ckpt_path = os.path.join(base_dir, "Saves", exp_name, "audio2pose_epoch_200.pth")
    if not os.path.exists(ckpt_path):
        print(f"ERRORE: Checkpoint non trovato in {ckpt_path}")
        return

    print(f"=== Generazione Risultati per {exp_name} ===")
    
    # 1. Carica modello
    sd = torch.load(ckpt_path, map_location=device)
    _hidden_dim = sd['lstm.weight_ih_l0'].shape[0] // 4
    _num_layers = max(
        int(k.split('lstm.weight_ih_l')[1].split('_')[0])
        for k in sd if k.startswith('lstm.weight_ih_l') and '_reverse' not in k
    ) + 1
    
    class ModelArgs: pass
    m_args = ModelArgs()
    m_args.num_layers = _num_layers
    m_args.hidden_dim = _hidden_dim
    m_args.dropout = 0.0
    
    # Se One-Hot, passiamo num_speakers
    is_one_hot = "OneHot" in exp_name
    if is_one_hot:
        with open(os.path.join(base_dir, "speaker_mapping.json"), "r") as f:
            speaker_mapping = json.load(f)
        m_args.num_speakers = len(speaker_mapping)
        
    model = HeadPosePredictor(m_args)
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    
    # 2. Carica audio e predici
    audio_feat_path = os.path.join(base_dir, "dataset_chunks", f"{sample}_audio.npy")
    audio_feat = np.load(audio_feat_path)
    audio_tensor = torch.FloatTensor(audio_feat).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if is_one_hot:
            ident = sample.split('_')[1] # AmandaStuck
            spk_id = speaker_mapping.get(ident, 0)
            preds = model(audio_tensor, pose_lengths=None, speaker_ids=torch.LongTensor([spk_id]).to(device))
        else:
            preds = model(audio_tensor, pose_lengths=None)
            
    preds = preds.squeeze(0).cpu().numpy()
    pred_path = os.path.join(res_dir, "predicted_pose.npy")
    np.save(pred_path, preds)
    
    # 3. Calcola Metriche (Evaluate)
    print("Calcolo Metriche...")
    hdtf_dir = "/mnt/diskone-second/DiffPoseTalk/datasets/HDTF_TFHP_Elaborated_Pose"
    orig_sample = sample.replace("_chunk0", "")
    res = evaluate.evaluate_single_sample(preds, orig_sample, hdtf_dir, label=exp_name)
    metrics_path = os.path.join(quant_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(res, f, indent=4)
        
    # 5. Generazione Video Qualitativo
    print("Generazione Rendering Video...")
    n_frames = res["n_frames"]
    preds = preds[:n_frames]
    
    # Trova l'audio wav originale per il muxing (serve l'audio a 16khz)
    import librosa
    audio_wav_path = os.path.join("/mnt/diskone-second/DiffPoseTalk/datasets/HDTF_TFHP_Elaborated_Pose/audio", sample.replace("_chunk0", "") + ".wav")
    audio_data, sr = librosa.load(audio_wav_path, sr=16000)
    
    # Per essere precisi tagliamo i primi 20 secondi dell'audio (chunk0)
    audio_trimmed = audio_data[:20*16000]
    trimmed_audio_path = os.path.join(res_dir, "trimmed_audio.wav")
    sf.write(trimmed_audio_path, audio_trimmed, sr)
    
    # Vertici (uso ScanTalk vertices da HDTF dataset)
    hdtf_dir = "/mnt/diskone-second/DiffPoseTalk/datasets/HDTF_TFHP_Elaborated_Pose"
    orig_sample = sample.replace("_chunk0", "")
    
    verts_static = np.load(os.path.join(hdtf_dir, "vertices", f"{orig_sample}.npy"))[:n_frames]
    verts_gt = np.load(os.path.join(hdtf_dir, "vertices_pose", f"{orig_sample}.npy"))[:n_frames]
    
    verts_pred = np.zeros_like(verts_static)
    for i in range(n_frames):
        verts_pred[i] = utils.apply_rotation_to_vertices(verts_static[i], preds[i], use_origin=True)
        
    import trimesh
    template = trimesh.load(os.path.join(base_dir, "..", "ScanTalk/src/examples/FLAME_sample.ply"), process=False)
    faces = np.array(template.faces)
    
    gt_vid = os.path.join(res_dir, "gt_video.mp4")
    pred_vid = os.path.join(res_dir, "pred_video.mp4")
    render_vertices_to_video(verts_gt, faces, trimmed_audio_path, gt_vid, fps=args.fps)
    render_vertices_to_video(verts_pred, faces, trimmed_audio_path, pred_vid, fps=args.fps)
    
    confronto = os.path.join(res_dir, "confronto.mp4")
    merge_side_by_side(gt_vid, pred_vid, confronto, label_left="GT", label_right=exp_name)
    
    # Cleanup
    os.remove(gt_vid)
    os.remove(pred_vid)
    
    print(f"\nFatto! Risultati salvati in: {res_dir}")

if __name__ == "__main__":
    main()
