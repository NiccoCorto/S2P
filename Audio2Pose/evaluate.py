"""
evaluate.py - Script di valutazione metriche per S2P (Speech-to-Pose)
Calcola MAE, MSE e Velocity Error tra le predizioni e il ground truth.
"""
import os
import sys
import csv
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_metrics(pred, gt):
    """Calcola metriche tra una singola predizione e il ground truth.
    
    Args:
        pred: np.ndarray shape (N_frames, 3) — Predizione (Pitch, Yaw, Roll)
        gt:   np.ndarray shape (N_frames, 3) — Ground truth (Pitch, Yaw, Roll)
    
    Returns:
        dict con le metriche calcolate
    """
    # Allinea le lunghezze
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]

    # MSE (Mean Squared Error) — complessivo
    mse = np.mean((pred - gt) ** 2)

    # MAE (Mean Absolute Error) — per asse e complessivo
    mae_per_axis = np.mean(np.abs(pred - gt), axis=0)  # shape (3,)
    mae_total = np.mean(np.abs(pred - gt))

    # Velocity Error — misura la fluidità del movimento
    # Confronta le "derivate" (velocità frame-to-frame)
    pred_vel = pred[1:] - pred[:-1]
    gt_vel = gt[1:] - gt[:-1]
    vel_error = np.mean(np.abs(pred_vel - gt_vel))

    return {
        "mse": mse,
        "mae_total": mae_total,
        "mae_pitch": mae_per_axis[0],
        "mae_yaw": mae_per_axis[1],
        "mae_roll": mae_per_axis[2],
        "vel_error": vel_error,
        "n_frames": min_len
    }


def evaluate_results(results_dir, gt_dir, output_csv=None):
    """Valuta tutte le predizioni in una cartella rispetto al ground truth.
    
    Args:
        results_dir: Cartella con i file .npy predetti dal modello
        gt_dir: Cartella con i file .npy del ground truth (pose/)
        output_csv: Path opzionale per salvare i risultati in CSV
    
    Returns:
        dict con metriche aggregate
    """
    pred_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.npy')])

    if len(pred_files) == 0:
        print(f"Nessun file .npy trovato in {results_dir}")
        return None

    all_metrics = []
    per_file_results = []

    for f in tqdm(pred_files, desc="Valutazione"):
        pred_path = os.path.join(results_dir, f)
        gt_path = os.path.join(gt_dir, f)

        if not os.path.exists(gt_path):
            print(f"  [WARN] GT non trovato per {f}, skip")
            continue

        pred = np.load(pred_path)
        gt = np.load(gt_path)

        metrics = compute_metrics(pred, gt)
        metrics["file"] = f
        all_metrics.append(metrics)
        per_file_results.append(metrics)

    if len(all_metrics) == 0:
        print("Nessun file valutato!")
        return None

    # --- Metriche aggregate ---
    agg = {
        "n_files": len(all_metrics),
        "mse_mean": np.mean([m["mse"] for m in all_metrics]),
        "mse_std": np.std([m["mse"] for m in all_metrics]),
        "mae_mean": np.mean([m["mae_total"] for m in all_metrics]),
        "mae_std": np.std([m["mae_total"] for m in all_metrics]),
        "mae_pitch_mean": np.mean([m["mae_pitch"] for m in all_metrics]),
        "mae_yaw_mean": np.mean([m["mae_yaw"] for m in all_metrics]),
        "mae_roll_mean": np.mean([m["mae_roll"] for m in all_metrics]),
        "vel_error_mean": np.mean([m["vel_error"] for m in all_metrics]),
        "vel_error_std": np.std([m["vel_error"] for m in all_metrics]),
    }

    # --- Stampa risultati ---
    print(f"\n{'='*60}")
    print(f"  RISULTATI VALUTAZIONE S2P")
    print(f"{'='*60}")
    print(f"  File valutati: {agg['n_files']}")
    print(f"")
    print(f"  MSE complessivo:  {agg['mse_mean']:.6f} ± {agg['mse_std']:.6f}")
    print(f"  MAE complessivo:  {agg['mae_mean']:.6f} ± {agg['mae_std']:.6f}")
    print(f"")
    print(f"  MAE per asse:")
    print(f"    Pitch: {agg['mae_pitch_mean']:.6f}")
    print(f"    Yaw:   {agg['mae_yaw_mean']:.6f}")
    print(f"    Roll:  {agg['mae_roll_mean']:.6f}")
    print(f"")
    print(f"  Velocity Error:   {agg['vel_error_mean']:.6f} ± {agg['vel_error_std']:.6f}")
    print(f"{'='*60}\n")

    # --- Salva CSV se richiesto ---
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "file", "mse", "mae_total", "mae_pitch", "mae_yaw",
                "mae_roll", "vel_error", "n_frames"
            ])
            writer.writeheader()
            for m in per_file_results:
                writer.writerow(m)

        # Aggiungi riga di riepilogo
        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(["MEDIA", f"{agg['mse_mean']:.6f}",
                             f"{agg['mae_mean']:.6f}",
                             f"{agg['mae_pitch_mean']:.6f}",
                             f"{agg['mae_yaw_mean']:.6f}",
                             f"{agg['mae_roll_mean']:.6f}",
                             f"{agg['vel_error_mean']:.6f}", ""])

        print(f"Risultati salvati in: {output_csv}")

    return agg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Valutazione metriche S2P")
    parser.add_argument("--results_dir", type=str, default="Results",
                        help="Cartella con le predizioni .npy")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="Cartella con il ground truth .npy (pose/)")
    parser.add_argument("--output_csv", type=str, default="Logs/evaluation_results.csv",
                        help="Path dove salvare i risultati CSV")
    args = parser.parse_args()

    # Se gt_dir non specificato, usa il path SSH di default
    if args.gt_dir is None:
        args.gt_dir = "/mnt/diskone-first/Mead_EMOTE_Elaborated_Pose/pose"

    evaluate_results(args.results_dir, args.gt_dir, args.output_csv)
