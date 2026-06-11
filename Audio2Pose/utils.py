"""
utils.py - Funzioni di utilità condivise per S2P
"""
import os
import sys
import csv
import numpy as np
import torch
import cv2
import librosa
from transformers import Wav2Vec2Processor

# Aggiungi il path del progetto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model(args, checkpoint_path=None):
    """Carica il modello HeadPosePredictor con i pesi da checkpoint.
    
    Args:
        args: Namespace con gli iperparametri del modello
        checkpoint_path: Path al file .pth. Se None, usa best_audio2pose.pth
    
    Returns:
        model: Modello pronto per l'inferenza
    """
    from model import HeadPosePredictor

    model = HeadPosePredictor(args)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(args.save_path, "best_audio2pose.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=args.device)
    )
    model = model.to(torch.device(args.device))
    model.eval()

    print(f"Modello caricato da: {checkpoint_path}")
    return model


def predict_pose_from_audio(model, audio_path, device="cpu", target_frames=None):
    """Predice la posa della testa (Pitch, Yaw, Roll) da un file audio.
    
    Pipeline completa: carica audio → Wav2Vec2 features → modello → 3 angoli per frame.
    
    Args:
        model: HeadPosePredictor già caricato
        audio_path: Path al file .wav
        device: "cuda" o "cpu"
        target_frames: Numero di frame da generare. Se None, il modello decide.
    
    Returns:
        np.ndarray shape (N_frames, 3) con Pitch, Yaw, Roll per frame
    """
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    # Carica audio a 16kHz
    speech_array, _ = librosa.load(audio_path, sr=16000)
    input_values = processor(speech_array, sampling_rate=16000).input_values
    audio_tensor = torch.FloatTensor(input_values).to(device)

    # Predizione
    with torch.no_grad():
        predictions = model(audio_tensor, target_seq_len=target_frames)

    return predictions.squeeze().cpu().numpy()


def create_rotation_matrix(angles):
    """Crea una matrice di rotazione 3x3 da un vettore di 3 angoli (Rodrigues).
    
    Args:
        angles: np.ndarray shape (3,) — Pitch, Yaw, Roll
    
    Returns:
        R: np.ndarray shape (3, 3) — Matrice di rotazione
    """
    R, _ = cv2.Rodrigues(angles.astype(np.float64))
    return R


def apply_rotation_to_vertices(vertices, angles, center=None, use_origin=True):
    """Applica una rotazione (Rodrigues) a un set di vertici 3D.
    
    Per mesh FLAME (usate da ScanTalk), l'origine (0,0,0) è posizionata
    al perno anatomico del collo/base cranio. Ruotare attorno all'origine
    produce movimenti naturali della testa.
    
    Args:
        vertices: np.ndarray shape (N, 3) — Vertici della mesh
        angles: np.ndarray shape (3,) — Pitch, Yaw, Roll (vettore Rodrigues)
        center: np.ndarray shape (3,) — Centro di rotazione esplicito.
                Sovrascrive use_origin se specificato.
        use_origin: Se True (default) e center è None, ruota attorno a (0,0,0).
                    Se False e center è None, ruota attorno al baricentro.
    
    Returns:
        np.ndarray shape (N, 3) — Vertici ruotati
    """
    if center is None:
        if use_origin:
            center = np.zeros(3)
        else:
            center = np.mean(vertices, axis=0)

    R = create_rotation_matrix(angles)
    rotated = R.dot((vertices - center).T).T + center
    return rotated


def plot_training_curves(csv_path, output_path=None):
    """Plotta le curve di training e validation loss da un file CSV di log.
    
    Args:
        csv_path: Path al file training_log.csv
        output_path: Path dove salvare il grafico. Se None, mostra a schermo.
    """
    try:
        import matplotlib
        if output_path:
            matplotlib.use('Agg')  # Backend non interattivo per salvare
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib non installato, impossibile plottare")
        return

    epochs = []
    train_losses = []
    val_losses = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2, color="#2196F3")
    ax.plot(epochs, val_losses, label="Validation Loss", linewidth=2, color="#F44336")
    ax.set_xlabel("Epoca", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("S2P Training Curves", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Grafico salvato in: {output_path}")
    else:
        plt.show()

    plt.close(fig)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="S2P Utilities")
    parser.add_argument("--action", type=str, choices=["plot", "predict"],
                        required=True, help="Azione da eseguire")
    parser.add_argument("--csv", type=str, help="Path al CSV di log (per plot)")
    parser.add_argument("--output", type=str, help="Path output grafico")
    parser.add_argument("--audio", type=str, help="Path audio WAV (per predict)")

    cli_args = parser.parse_args()

    if cli_args.action == "plot":
        if not cli_args.csv:
            print("Serve --csv per plottare!")
        else:
            plot_training_curves(cli_args.csv, cli_args.output)

    elif cli_args.action == "predict":
        print("Per predict, usa demo_nicco.py con --audio")
