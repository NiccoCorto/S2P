"""
train.py - Script di addestramento per Audio2Pose (S2P)
Segue la struttura di train_S2L.py di s2l-s2d.
Features: early stopping, best model saving, LR scheduler, logging CSV.
"""
import numpy as np
import csv
import os
import sys

import torch
import torch.nn as nn
from tqdm import tqdm

# Aggiungiamo la cartella principale al path per trovare il dataloader e config
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
        # 1. Errore di posizione (quanto siamo lontani dagli angoli reali)
        pos_loss = self.mse(predictions, target)

        # 2. Errore di velocità (differenza tra il frame attuale e il precedente)
        # Questo costringe il modello a non fare movimenti "a scatti"
        prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_shift = target[:, 1:, :] - target[:, :-1, :]
        vel_loss = self.mse(prediction_shift, target_shift)

        # Sommiamo gli errori (diamo più peso alla velocità per movimenti fluidi)
        return pos_loss + (self.vel_weight * vel_loss)


def setup_logging(log_path):
    """Crea la cartella di log e il file CSV per le loss."""
    os.makedirs(log_path, exist_ok=True)
    csv_path = os.path.join(log_path, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "best"])
    return csv_path


def log_epoch(csv_path, epoch, train_loss, val_loss, lr, is_best):
    """Scrive una riga nel log CSV."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}",
                         f"{lr:.8f}", "★" if is_best else ""])


def trainer(args, train_loader, dev_loader, model, optimizer, criterion, scheduler):
    """Loop di addestramento con early stopping e best model saving."""
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    csv_path = setup_logging(args.log_path)

    print(f"\n{'='*60}")
    print(f"Inizio addestramento su {args.device}")
    print(f"  Epoche max:      {args.max_epoch}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  Early stopping:  {args.patience} epoche senza miglioramento")
    print(f"  Hidden dim:      {args.hidden_dim}")
    print(f"  LSTM layers:     {args.num_layers}")
    print(f"  Dropout:         {args.dropout}")
    print(f"  Vel loss weight: {args.vel_loss_weight}")
    print(f"{'='*60}\n")

    best_val_loss = float("inf")
    patience_counter = 0

    for e in range(args.max_epoch):
        loss_log = []

        # --- FASE DI TRAINING ---
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {e+1}/{args.max_epoch} [TRAIN]")

        for i, (audio, pose_target, file_name) in pbar:
            audio = audio.to(device=args.device)
            pose_target = pose_target.to(device=args.device)

            optimizer.zero_grad()

            # Passiamo la lunghezza reale delle pose al modello
            predictions = model(audio, target_seq_len=pose_target.size(1))

            # Allinea la lunghezza (a volte differiscono di 1 frame)
            min_seq_len = min(predictions.size(1), pose_target.size(1))
            predictions = predictions[:, :min_seq_len, :]
            pose_target_aligned = pose_target[:, :min_seq_len, :]

            loss = criterion(predictions, pose_target_aligned)
            loss.backward()
            optimizer.step()

            loss_log.append(loss.item())
            pbar.set_postfix({"Loss": f"{np.mean(loss_log):.6f}"})

        train_loss = np.mean(loss_log)

        # --- FASE DI VALIDATION ---
        valid_loss_log = []
        model.eval()

        with torch.no_grad():
            for audio, pose_target, file_name in dev_loader:
                audio = audio.to(device=args.device)
                pose_target = pose_target.to(device=args.device)

                # FIX: passare target_seq_len anche in validation!
                predictions = model(audio, target_seq_len=pose_target.size(1))

                min_seq_len = min(predictions.size(1), pose_target.size(1))
                predictions = predictions[:, :min_seq_len, :]
                pose_target_aligned = pose_target[:, :min_seq_len, :]

                loss = criterion(predictions, pose_target_aligned)
                valid_loss_log.append(loss.item())

        val_loss = np.mean(valid_loss_log) if valid_loss_log else float("inf")

        # --- Learning Rate Scheduler ---
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != current_lr:
            print(f"  📉 LR ridotto: {current_lr:.2e} → {new_lr:.2e}")

        # --- Best Model & Early Stopping ---
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(save_path, "best_audio2pose.pth"))
        else:
            patience_counter += 1

        # Log
        log_epoch(csv_path, e + 1, train_loss, val_loss, new_lr, is_best)

        # Print riepilogo epoca
        best_marker = " ★ BEST" if is_best else ""
        print(f"  Epoca {e+1}/{args.max_epoch} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {new_lr:.2e} | "
              f"Patience: {patience_counter}/{args.patience}{best_marker}")

        # Salva checkpoint periodici
        if (e + 1) % 50 == 0:
            torch.save(model.state_dict(),
                       os.path.join(save_path, f"audio2pose_epoch_{e+1}.pth"))

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⏹  Early stopping! Nessun miglioramento per {args.patience} epoche.")
            print(f"   Miglior Val Loss: {best_val_loss:.6f}")
            break

    # Salva il modello finale
    torch.save(model.state_dict(),
               os.path.join(save_path, f"audio2pose_final_epoch_{e+1}.pth"))

    print(f"\n✅ Training completato!")
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
        print(f"  [WARN] best_audio2pose.pth non trovato, uso il modello corrente")

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

    print(f"\n✅ Test completato! Pose generate salvate in: {result_path}")


def main():
    args = get_args()

    print(f"\n🧠 S2P — Speech-to-Pose")
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

    # 3. Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )

    # 4. Carica i dati
    dataset = get_dataloaders(args)

    # 5. Training
    model = trainer(args, dataset["train"], dataset["valid"],
                    model, optimizer, criterion, scheduler)

    # 6. Test
    test(args, model, dataset["test"])


if __name__ == "__main__":
    main()