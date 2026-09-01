"""
evaluate.py - Script di Valutazione Quantitativa per S2P (Speech-to-Pose)

Calcola per ciascun campione:
1. Numero di frame presi in considerazione.
2. Head Pose (Orientamento):
   - PosLoss Metric: MAE e MSE angolari rispetto alla GT (totale e per asse Pitch, Yaw, Roll).
   - Temporal Consistency Metric (VelLoss): MSE/MAE sulla derivata prima temporale (Δθ) per verificare la fluidità.
3. 3D Face Mesh (Vertici FLAME):
   - PosLoss Metric: Mean Vertex Error (MVE) in mm (distanza Euclidea L2 media).
   - Temporal Consistency Metric (Mesh VelLoss): MSE della velocità dei vertici per misurare il jittering superficiale.

Uso:
  # Singolo file di predizione con nome campione:
  python S2P/Audio2Pose/evaluate.py \
      --pred_pose S2P/Results/expC_overfit_4L_vel2/esperimento_Train300_vel/predicted_pose.npy \
      --sample_name M003_contempt_2_022

  # Cartella di esperimento:
  python S2P/Audio2Pose/evaluate.py \
      --results_dir S2P/Results/expC_overfit_4L_vel2/esperimento_Test300_vel \
      --sample_name W037_disgusted_3_003

  # Batch di file .npy in una cartella:
  python S2P/Audio2Pose/evaluate.py \
      --results_dir S2P/Results/my_experiment
"""

import os
import sys
import argparse
import json
import csv
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

DEFAULT_DATASET_DIR = "/mnt/diskone-first/Mead_EMOTE_Elaborated_Pose"
AXES = ["Pitch", "Yaw", "Roll"]


def create_rotation_matrix(angles):
    """Crea matrice di rotazione 3x3 da vettore Rodrigues (Pitch, Yaw, Roll)."""
    R, _ = cv2.Rodrigues(angles.astype(np.float64))
    return R


def apply_rotation_to_vertices(vertices, angles, use_origin=True):
    """Applica la rotazione ai vertici 3D della mesh."""
    center = np.zeros(3) if use_origin else np.mean(vertices, axis=0)
    R = create_rotation_matrix(angles)
    return R.dot((vertices - center).T).T + center


def compute_pose_metrics(pred, gt):
    """Calcola le metriche di posizione e consistenza temporale sulla posa della testa."""
    n = min(len(pred), len(gt))
    pred = pred[:n]
    gt = gt[:n]

    # ── PosLoss (Posizione statica) ──────────────────────────
    err = pred - gt
    mae_per_axis = np.mean(np.abs(err), axis=0)
    mse_per_axis = np.mean(err ** 2, axis=0)
    mae_total = float(np.mean(np.abs(err)))
    mse_total = float(np.mean(err ** 2))

    # ── Temporal Consistency (Velocity / Derivata temporale) ──
    vel_pred = np.diff(pred, axis=0)
    vel_gt = np.diff(gt, axis=0)
    vel_err_sq = (vel_pred - vel_gt) ** 2
    vel_mse_per_axis = np.mean(vel_err_sq, axis=0)
    vel_mse_total = float(np.mean(vel_err_sq))
    vel_mae_total = float(np.mean(np.abs(vel_pred - vel_gt)))

    return {
        "n_frames": n,
        "pos_loss_mae_total": mae_total,
        "pos_loss_mae_pitch": float(mae_per_axis[0]),
        "pos_loss_mae_yaw": float(mae_per_axis[1]),
        "pos_loss_mae_roll": float(mae_per_axis[2]),
        "pos_loss_mse_total": mse_total,
        "pos_loss_mse_pitch": float(mse_per_axis[0]),
        "pos_loss_mse_yaw": float(mse_per_axis[1]),
        "pos_loss_mse_roll": float(mse_per_axis[2]),
        "temporal_consistency_vel_mse_total": vel_mse_total,
        "temporal_consistency_vel_mse_pitch": float(vel_mse_per_axis[0]),
        "temporal_consistency_vel_mse_yaw": float(vel_mse_per_axis[1]),
        "temporal_consistency_vel_mse_roll": float(vel_mse_per_axis[2]),
        "temporal_consistency_vel_mae_total": vel_mae_total,
    }


def compute_mesh_metrics(pred_verts, gt_verts):
    """Calcola le metriche di posizione (MVE) e consistenza temporale (Jitter) sui vertici 3D."""
    n = min(len(pred_verts), len(gt_verts))
    pred_verts = pred_verts[:n]
    gt_verts = gt_verts[:n]

    # ── PosLoss (Mean Vertex Error - MVE L2) ───────────────────
    per_frame_mve = np.mean(np.linalg.norm(pred_verts - gt_verts, axis=-1), axis=-1)
    mve_mean_m = float(np.mean(per_frame_mve))
    mve_mean_mm = mve_mean_m * 1000.0
    mve_std_mm = float(np.std(per_frame_mve)) * 1000.0

    # ── Temporal Consistency (Mesh Velocity MSE - Jittering) ─
    vel_pred_verts = np.diff(pred_verts, axis=0)
    vel_gt_verts = np.diff(gt_verts, axis=0)
    mesh_vel_mse = float(np.mean((vel_pred_verts - vel_gt_verts) ** 2))
    mesh_vel_mae = float(np.mean(np.abs(vel_pred_verts - vel_gt_verts)))

    return {
        "n_frames": n,
        "pos_loss_mve_mm": mve_mean_mm,
        "pos_loss_mve_std_mm": mve_std_mm,
        "temporal_consistency_mesh_vel_mse": mesh_vel_mse,
        "temporal_consistency_mesh_vel_mae": mesh_vel_mae,
    }


def evaluate_single_sample(pred_pose, sample_name, dataset_dir, label="Sample"):
    """Esegue la valutazione completa su un singolo esempio e ne stampa il report."""
    stem = Path(sample_name).stem
    pose_gt_path = os.path.join(dataset_dir, "pose", f"{stem}.npy")
    vert_static_path = os.path.join(dataset_dir, "vertices", f"{stem}.npy")
    vert_gt_path = os.path.join(dataset_dir, "vertices_pose", f"{stem}.npy")

    if not os.path.exists(pose_gt_path):
        print(f"[ERRORE] GT pose non trovato in: {pose_gt_path}")
        return None

    gt_pose = np.load(pose_gt_path)
    pose_res = compute_pose_metrics(pred_pose, gt_pose)
    n_frames = pose_res["n_frames"]

    mesh_res = None
    if os.path.exists(vert_static_path) and os.path.exists(vert_gt_path):
        vert_static = np.load(vert_static_path)[:n_frames]
        vert_gt = np.load(vert_gt_path)[:n_frames]
        
        # Applica le pose predette ai vertici statici
        pred_verts = np.zeros_like(vert_static)
        for i in range(n_frames):
            pred_verts[i] = apply_rotation_to_vertices(vert_static[i], pred_pose[i], use_origin=True)
            
        mesh_res = compute_mesh_metrics(pred_verts, vert_gt)

    # ── Stampa Report Formattato ───────────────────────────────
    print("\n" + "=" * 70)
    print(f"  VALUTAZIONE QUANTITATIVA — {label}: {stem}")
    print("=" * 70)
    print(f"  • Frame presi in considerazione: {n_frames}")
    print("\n  [1] HEAD POSE (Orientamento Angolare):")
    print(f"      • PosLoss - MAE Totale:        {pose_res['pos_loss_mae_total']:.6f} rad ({np.degrees(pose_res['pos_loss_mae_total']):.2f}°)")
    print(f"        - Pitch (Su/Giù):           {pose_res['pos_loss_mae_pitch']:.6f} rad ({np.degrees(pose_res['pos_loss_mae_pitch']):.2f}°)")
    print(f"        - Yaw (Destra/Sinistra):    {pose_res['pos_loss_mae_yaw']:.6f} rad ({np.degrees(pose_res['pos_loss_mae_yaw']):.2f}°)")
    print(f"        - Roll (Inclinazione):      {pose_res['pos_loss_mae_roll']:.6f} rad ({np.degrees(pose_res['pos_loss_mae_roll']):.2f}°)")
    print(f"      • PosLoss - MSE Totale:        {pose_res['pos_loss_mse_total']:.6f} rad²")
    print(f"      • Temporal Consistency Metric (VelLoss):")
    print(f"        - Angular VelMSE:           {pose_res['temporal_consistency_vel_mse_total']:.3e} (rad/f)²")
    print(f"        - Angular VelMAE:           {pose_res['temporal_consistency_vel_mae_total']:.6f} rad/f")

    if mesh_res:
        print("\n  [2] 3D FACE MESH (Vertici FLAME):")
        print(f"      • PosLoss - Mean Vertex Error (MVE):  {mesh_res['pos_loss_mve_mm']:.4f} mm (std: {mesh_res['pos_loss_mve_std_mm']:.4f} mm)")
        print(f"      • Temporal Consistency Metric (Mesh VelLoss):")
        print(f"        - Mesh Velocity MSE (Jitter):       {mesh_res['temporal_consistency_mesh_vel_mse']:.3e} (m/f)²")
        print(f"        - Mesh Velocity MAE:                {mesh_res['temporal_consistency_mesh_vel_mae']:.6f} m/f")
    else:
        print("\n  [2] 3D FACE MESH: File vertici non trovati nel dataset.")

    print("=" * 70 + "\n")

    full_results = {
        "sample": stem,
        "n_frames": n_frames,
        "head_pose": pose_res,
        "mesh_3d": mesh_res
    }
    return full_results


def main():
    parser = argparse.ArgumentParser(description="Valutazione Quantitativa S2P (Head Pose & Mesh)")
    parser.add_argument("--pred_pose", type=str, default=None,
                        help="Path al file .npy della posa predetta")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Cartella contenente predicted_pose.npy o multipli file .npy")
    parser.add_argument("--sample_name", type=str, default=None,
                        help="Nome del campione (es. M003_contempt_2_022)")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR,
                        help=f"Path alla cartella base del dataset (default: {DEFAULT_DATASET_DIR})")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path opzionale dove salvare il report JSON")
    args = parser.parse_args()

    # Caso 1: Path esplicito a predicted_pose
    if args.pred_pose:
        if not os.path.exists(args.pred_pose):
            print(f"[ERRORE] File predizione non trovato: {args.pred_pose}")
            sys.exit(1)
        pred_pose = np.load(args.pred_pose)
        sample_name = args.sample_name or Path(args.pred_pose).stem
        res = evaluate_single_sample(pred_pose, sample_name, args.dataset_dir)
        if args.output_json and res:
            os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(res, f, indent=4)
            print(f"Risultati salvati in: {args.output_json}")
        return

    # Caso 2: Cartella di risultati
    if args.results_dir:
        pred_pose_single = os.path.join(args.results_dir, "predicted_pose.npy")
        if os.path.exists(pred_pose_single):
            pred_pose = np.load(pred_pose_single)
            sample_name = args.sample_name or Path(args.results_dir).name
            res = evaluate_single_sample(pred_pose, sample_name, args.dataset_dir)
            if args.output_json and res:
                os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
                with open(args.output_json, "w") as f:
                    json.dump(res, f, indent=4)
                print(f"Risultati salvati in: {args.output_json}")
            return

        # Multipli file .npy nella cartella
        files = sorted([f for f in os.listdir(args.results_dir) if f.endswith(".npy")])
        if not files:
            print(f"[ERRORE] Nessun file .npy trovato in: {args.results_dir}")
            sys.exit(1)

        all_res = []
        for f in tqdm(files, desc="Valutazione Batch"):
            pred_p = np.load(os.path.join(args.results_dir, f))
            s_name = f.replace(".npy", "")
            r = evaluate_single_sample(pred_p, s_name, args.dataset_dir, label=f)
            if r:
                all_res.append(r)

        if args.output_json and all_res:
            os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
            with open(args.output_json, "w") as out_f:
                json.dump(all_res, out_f, indent=4)
            print(f"Report batch salvato in: {args.output_json}")
        return

    print("Specifica --pred_pose oppure --results_dir. Usa -h per visualizzare la guida.")


if __name__ == "__main__":
    main()
