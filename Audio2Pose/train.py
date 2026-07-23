"""
train.py - Script di addestramento per Audio2Pose (S2P)
Segue la struttura di train_S2L.py di s2l-s2d.
Features: early stopping, best model saving, LR fisso, logging CSV separato per pos/vel loss.
"""
import numpy as np
import csv
import os
import sys

import comet_ml
import torch
import torch.nn as nn
from tqdm import tqdm

# aggiungi la cartella principale al path per trovare il dataloader e config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import get_dataloaders
from config import get_args
from model import HeadPosePredictor


class PoseLoss(nn.Module):
    """
    Funzione di errore personalizzata per la testa.
    Calcola l'errore sulla posizione esatta (MSE) + l'errore sulla velocità
    del movimento (per renderlo fluido e naturale).
    """
    def __init__(self, vel_weight=2.0):
        super(PoseLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.vel_weight = vel_weight

    def forward(self, predictions, target):
        # errore di posizione (quanto siamo lontani dagli angoli reali)
        pos_loss = self.mse(predictions, target)

        # errore di velocità (differenza tra il frame attuale e il precedente)
        # questo costringe il modello a non fare movimenti "a scatti"
        prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_shift = target[:, 1:, :] - target[:, :-1, :]
        vel_loss = self.mse(prediction_shift, target_shift)

        # sommiamo gli errori con il peso configurato
        # vel_weight=0 disabilita la velocity loss (solo pos_loss)
        # vel_weight=2044 standardizza la vel_loss per avere la stessa magnitudo di pos_loss
        total_loss = pos_loss + (self.vel_weight * vel_loss)

        # ritorniamo anche i valori grezzi (non pesati) per il logging
        return total_loss, pos_loss.detach().item(), vel_loss.detach().item()


def setup_logging(log_path):
    """Crea la cartella di log e il file CSV per le loss."""
    os.makedirs(log_path, exist_ok=True)
    csv_path = os.path.join(log_path, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_pos_loss", "train_vel_loss",
                         "val_loss", "val_pos_loss", "val_vel_loss", "lr", "best"])
    return csv_path


def log_epoch(csv_path, epoch, train_loss, train_pos, train_vel,
              val_loss, val_pos, val_vel, lr, is_best):
    """Scrive una riga nel log CSV con le loss separate."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{train_loss:.8f}", f"{train_pos:.8f}", f"{train_vel:.8f}",
            f"{val_loss:.8f}", f"{val_pos:.8f}", f"{val_vel:.8f}",
            f"{lr:.8f}", "best" if is_best else ""
        ])


def trainer(args, train_loader, dev_loader, model, optimizer, criterion, experiment):
    """Loop di addestramento con early stopping e best model saving."""
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    csv_path = setup_logging(args.log_path)

    print(f"\n{'='*60}")
    print(f"Inizio addestramento su {args.device}")
    print(f"  Epoche max:      {args.max_epoch}")
    print(f"  Learning rate:   {args.lr} (FISSO, no scheduler)")
    print(f"  Early stopping:  {args.patience} epoche senza miglioramento")
    print(f"  Hidden dim:      {args.hidden_dim}")
    print(f"  LSTM layers:     {args.num_layers}")
    print(f"  Dropout:         {args.dropout}")
    print(f"  Vel loss weight: {args.vel_loss_weight}"
          f" ({'disabilitata' if args.vel_loss_weight == 0 else 'standardizzata' if args.vel_loss_weight > 100 else 'normale'})")
    print(f"  Checkpoint ogni: 20 epoche")
    print(f"{'='*60}\n")

    best_val_loss = float("inf")
    patience_counter = 0

    for e in range(args.max_epoch):
        loss_log = []
        pos_loss_log = []
        vel_loss_log = []

        # fase di training
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {e+1}/{args.max_epoch} [TRAIN]")

        for i, (audio, pose_target, file_name) in pbar:
            audio = audio.to(device=args.device)
            pose_target = pose_target.to(device=args.device)

            optimizer.zero_grad()

            # passiamo la lunghezza reale delle pose al modello
            predictions = model(audio, target_seq_len=pose_target.size(1))

            # allinea la lunghezza (a volte differiscono di 1 frame)
            min_seq_len = min(predictions.size(1), pose_target.size(1))
            predictions = predictions[:, :min_seq_len, :]
            pose_target_aligned = pose_target[:, :min_seq_len, :]

            loss, pos_l, vel_l = criterion(predictions, pose_target_aligned)
            loss.backward()
            optimizer.step()

            loss_log.append(loss.item())
            pos_loss_log.append(pos_l)
            vel_loss_log.append(vel_l)
            pbar.set_postfix({"Loss": f"{np.mean(loss_log):.6f}",
                              "Pos": f"{np.mean(pos_loss_log):.6f}",
                              "Vel": f"{np.mean(vel_loss_log):.8f}"})

        train_loss = np.mean(loss_log)
        train_pos_loss = np.mean(pos_loss_log)
        train_vel_loss = np.mean(vel_loss_log)

        # fase di validation
        valid_loss_log = []
        valid_pos_loss_log = []
        valid_vel_loss_log = []
        model.eval()

        with torch.no_grad():
            for audio, pose_target, file_name in dev_loader:
                audio = audio.to(device=args.device)
                pose_target = pose_target.to(device=args.device)

                # passare target_seq_len anche in validation
                predictions = model(audio, target_seq_len=pose_target.size(1))

                min_seq_len = min(predictions.size(1), pose_target.size(1))
                predictions = predictions[:, :min_seq_len, :]
                pose_target_aligned = pose_target[:, :min_seq_len, :]

                loss, pos_l, vel_l = criterion(predictions, pose_target_aligned)
                valid_loss_log.append(loss.item())
                valid_pos_loss_log.append(pos_l)
                valid_vel_loss_log.append(vel_l)

        val_loss = np.mean(valid_loss_log) if valid_loss_log else float("inf")
        val_pos_loss = np.mean(valid_pos_loss_log) if valid_pos_loss_log else float("inf")
        val_vel_loss = np.mean(valid_vel_loss_log) if valid_vel_loss_log else float("inf")

        # learning rate FISSO — nessuno scheduler
        current_lr = optimizer.param_groups[0]["lr"]

        # best model & early stopping
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(save_path, "best_audio2pose.pth"))
        else:
            patience_counter += 1

        # Log
        log_epoch(csv_path, e + 1,
                  train_loss, train_pos_loss, train_vel_loss,
                  val_loss, val_pos_loss, val_vel_loss,
                  current_lr, is_best)
        experiment.log_metrics({
            "train_loss": train_loss,
            "train_pos_loss": train_pos_loss,
            "train_vel_loss": train_vel_loss,
            "val_loss": val_loss,
            "val_pos_loss": val_pos_loss,
            "val_vel_loss": val_vel_loss,
            "learning_rate": current_lr
        }, step=e + 1)

        # print riepilogo epoca
        best_marker = " ★ BEST" if is_best else ""
        print(f"  Epoca {e+1}/{args.max_epoch} | "
              f"Train: {train_loss:.6f} (Pos: {train_pos_loss:.6f} | Vel: {train_vel_loss:.8f}) | "
              f"Val: {val_loss:.6f} (Pos: {val_pos_loss:.6f} | Vel: {val_vel_loss:.8f}) | "
              f"LR: {current_lr:.2e} | "
              f"Patience: {patience_counter}/{args.patience}{best_marker}")

        # salva checkpoint periodici ogni 20 epoche
        if (e + 1) % 20 == 0:
            torch.save(model.state_dict(),
                       os.path.join(save_path, f"audio2pose_epoch_{e+1}.pth"))

        # early stopping
        if patience_counter >= args.patience:
            print(f"\n Early stopping. Nessun miglioramento per {args.patience} epoche.")
            print(f"   Miglior Val Loss: {best_val_loss:.6f}")
            break

    # salva il modello finale
    torch.save(model.state_dict(),
               os.path.join(save_path, f"audio2pose_final_epoch_{e+1}.pth"))

    print(f"\n Training completato!")
    print(f"   Miglior modello: {os.path.join(save_path, 'best_audio2pose.pth')}")
    print(f"   Log training:    {csv_path}")

    return model


@torch.no_grad()
def test(args, model, test_loader):
    """Genera predizioni sui dati di test e le salva come .npy."""
    print(f"\n{'='*60}")
    print("Inizio fase di Test sui dati mai visti...")
    print(f"{'='*60}")

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)

    # Carica i pesi del miglior modello
    best_path = os.path.join(args.save_path, "best_audio2pose.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=args.device))
        print(f"  Caricato il miglior modello: {best_path}")
    else:
        print(f"  best_audio2pose.pth non trovato, uso il modello corrente")

    model = model.to(torch.device(args.device))
    model.eval()

    for audio, pose_target, file_name in tqdm(test_loader, desc="Testing"):
        audio = audio.to(device=args.device)

        # Genera predizione con la lunghezza del target (per confronto)
        predictions = model(audio, target_seq_len=pose_target.size(1))
        predictions = predictions.squeeze()  # Rimuove la dimensione del batch

        # Salva la predizione come file .npy
        save_name = os.path.join(result_path, file_name[0].replace(".wav", ".npy"))
        np.save(save_name, predictions.detach().cpu().numpy())

    print(f"\n Test completato! Pose generate salvate in: {result_path}")


def main():
    args = get_args()

    # Inizializza Comet ML (l'API key verrà letta dalla variabile d'ambiente COMET_API_KEY)
    experiment = comet_ml.Experiment(
        project_name="audio2pose",
        auto_metric_logging=True,
        auto_param_logging=True,
        auto_histogram_weight_logging=True,
        auto_histogram_gradient_logging=True,
        auto_histogram_activation_logging=True,
    )
    # Rinomina l'esperimento per identificarlo facilmente sulla dashboard
    experiment.set_name(f"vel_loss_{args.vel_loss_weight}_epochs_{args.max_epoch}")
    # Logga tutti gli argomenti (iperparametri, path, etc.)
    experiment.log_parameters(vars(args))

    print(f"\n S2P — Speech-to-Pose")
    print(f"   Modalità: {args.mode}")
    print(f"   Device:   {args.device}")
    print(f"   Dati:     {args.data_dir}\n")

    # 1. Costruisci il modello
    model = HeadPosePredictor(args)
    model = model.to(torch.device(args.device))

    # Conta parametri addestrabili
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Parametri: {trainable:,} addestrabili / {total:,} totali\n")

    # 2. Loss e Ottimizzatore
    criterion = PoseLoss(vel_weight=args.vel_loss_weight)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    # 3. (nessuno scheduler — LR fisso a 1e-4 per tutta la durata)

    # 4. Carica i dati
    dataset = get_dataloaders(args)

    # 5. Training
    model = trainer(args, dataset["train"], dataset["valid"],
                    model, optimizer, criterion, experiment)

    # 6. Test
    test(args, model, dataset["test"])


if __name__ == "__main__":
    main()