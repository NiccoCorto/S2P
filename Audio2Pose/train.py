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
import torch.nn.functional as F
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

    Supporta una padding mask tramite il parametro `lengths`: i frame paddati
    (oltre la lunghezza reale di ciascun campione) vengono completamente ignorati
    sia nella pos_loss che nella vel_loss, evitando che il modello impari a predire
    zero sui frame di padding (loss poisoning).
    """
    def __init__(self, vel_weight=2.0):
        super(PoseLoss, self).__init__()
        self.vel_weight = vel_weight

    def forward(self, predictions, target, lengths=None):
        """
        Args:
            predictions: Tensor (B, T, 3) — predizioni del modello
            target:      Tensor (B, T, 3) — pose target (eventualmente paddate a zero)
            lengths:     LongTensor (B,)  — lunghezza reale di ogni sequenza.
                         Se None, tutti i frame sono considerati reali (no mask).
        """
        B, T, C = target.shape

        # costruisce mask frame-level: True dove il frame è reale (non padding)
        if lengths is not None:
            idx = torch.arange(T, device=target.device).unsqueeze(0)  # (1, T)
            mask_2d = idx < lengths.unsqueeze(1)                       # (B, T)
        else:
            mask_2d = torch.ones(B, T, dtype=torch.bool, device=target.device)

        # --- Position Loss: MSE solo sui frame reali ---
        mask_3d = mask_2d.unsqueeze(-1).expand_as(target)  # (B, T, 3)
        pred_masked = predictions[mask_3d].view(-1, C)
        tgt_masked  = target[mask_3d].view(-1, C)
        pos_loss = F.mse_loss(pred_masked, tgt_masked)

        # --- Velocity Loss: differenze consecutive calcolate prima di appiattire ---
        # (così non si mescolano frame di sequenze diverse)
        pred_vel = predictions[:, 1:, :] - predictions[:, :-1, :]  # (B, T-1, C)
        tgt_vel  = target[:, 1:, :] - target[:, :-1, :]            # (B, T-1, C)
        # valido solo dove ENTRAMBI i frame consecutivi sono reali
        vel_mask    = mask_2d[:, 1:] & mask_2d[:, :-1]             # (B, T-1)
        vel_mask_3d = vel_mask.unsqueeze(-1).expand_as(pred_vel)   # (B, T-1, C)

        if vel_mask_3d.any():
            pred_vel_m = pred_vel[vel_mask_3d].view(-1, C)
            tgt_vel_m  = tgt_vel[vel_mask_3d].view(-1, C)
            vel_loss = F.mse_loss(pred_vel_m, tgt_vel_m)
        else:
            vel_loss = torch.tensor(0.0, device=target.device)

        # sommiamo gli errori con il peso configurato
        # vel_weight=0 disabilita la velocity loss (solo pos_loss)
        # vel_weight=2044 standardizza la vel_loss per avere la stessa magnitudo di pos_loss
        total_loss = pos_loss + (self.vel_weight * vel_loss)

        # ritorniamo anche i valori grezzi (non pesati) per il logging
        return total_loss, pos_loss.detach().item(), vel_loss.detach().item()


def setup_logging(log_path, start_epoch=0):
    """Crea la cartella di log e il file CSV per le loss."""
    os.makedirs(log_path, exist_ok=True)
    csv_path = os.path.join(log_path, "training_log.csv")
    mode = "a" if start_epoch > 0 else "w"
    with open(csv_path, mode, newline="") as f:
        if mode == "w":
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


def trainer(args, train_loader, dev_loader, model, optimizer, criterion, experiment, start_epoch=0):
    """Loop di addestramento con early stopping e best model saving."""
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    csv_path = setup_logging(args.log_path, start_epoch)

    print(f"\n{'='*60}")
    print(f"Inizio addestramento su {args.device}")
    if start_epoch > 0:
        print(f"  Ripresa da epoca: {start_epoch}")
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

    for e in range(start_epoch, args.max_epoch):
        loss_log = []
        pos_loss_log = []
        vel_loss_log = []

        # fase di training
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {e+1}/{args.max_epoch} [TRAIN]")

        for i, (audio, pose_target, pose_lengths, audio_lengths, file_name, speaker_ids) in pbar:
            audio = audio.to(device=args.device)
            pose_target = pose_target.to(device=args.device)
            pose_lengths = pose_lengths.to(device=args.device)
            speaker_ids = speaker_ids.to(device=args.device)

            optimizer.zero_grad()

            # il modello interpola alla lunghezza reale massima del batch (non paddata)
            predictions = model(audio, pose_lengths=pose_lengths, speaker_ids=speaker_ids)

            # la loss usa la mask per ignorare i frame paddati di ogni campione
            loss, pos_l, vel_l = criterion(predictions, pose_target, lengths=pose_lengths)
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
            for audio, pose_target, pose_lengths, audio_lengths, file_name, speaker_ids in dev_loader:
                audio = audio.to(device=args.device)
                pose_target = pose_target.to(device=args.device)
                pose_lengths = pose_lengths.to(device=args.device)
                speaker_ids = speaker_ids.to(device=args.device)

                predictions = model(audio, pose_lengths=pose_lengths, speaker_ids=speaker_ids)
                loss, pos_l, vel_l = criterion(predictions, pose_target, lengths=pose_lengths)
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

    for audio, pose_target, pose_lengths, audio_lengths, file_names, speaker_ids in tqdm(test_loader, desc="Testing"):
        audio = audio.to(device=args.device)
        pose_lengths = pose_lengths.to(device=args.device)
        speaker_ids = speaker_ids.to(device=args.device)

        # Genera predizioni per l'intero batch
        predictions = model(audio, pose_lengths=pose_lengths, speaker_ids=speaker_ids)  # (B, max_real_len, 3)

        # Salva ogni elemento del batch come file .npy separato,
        # tagliando alla lunghezza reale (evita di salvare frame paddati a zero)
        for i, fname in enumerate(file_names):
            real_len = pose_lengths[i].item()
            pred_i = predictions[i, :real_len, :]  # (real_len, 3)
            save_name = os.path.join(result_path, fname.replace(".wav", ".npy"))
            np.save(save_name, pred_i.detach().cpu().numpy())

    print(f"\n Test completato! Pose generate salvate in: {result_path}")


def main():
    args = get_args()

    # Inizializza Comet ML (l'API key verrà letta dalla variabile d'ambiente COMET_API_KEY)
    if args.comet_experiment_key:
        print(f"Ripresa esperimento Comet ML: {args.comet_experiment_key}")
        experiment = comet_ml.ExistingExperiment(
            previous_experiment=args.comet_experiment_key,
            project_name="audio2pose",
            auto_metric_logging=True,
            auto_param_logging=True,
            auto_histogram_weight_logging=True,
            auto_histogram_gradient_logging=True,
            auto_histogram_activation_logging=True,
        )
    else:
        experiment = comet_ml.Experiment(
            project_name="audio2pose",
            auto_metric_logging=True,
            auto_param_logging=True,
            auto_histogram_weight_logging=True,
            auto_histogram_gradient_logging=True,
            auto_histogram_activation_logging=True,
        )
    # Rinomina l'esperimento per identificarlo facilmente sulla dashboard
    auto_name = f"vel{args.vel_loss_weight}_L{args.num_layers}_h{args.hidden_dim}_ep{args.max_epoch}"
    exp_name = args.exp_name if args.exp_name else auto_name
    experiment.set_name(exp_name)
    # Tag con gli iperparametri chiave per filtrare su CometML
    experiment.add_tags([
        f"layers={args.num_layers}",
        f"hidden={args.hidden_dim}",
        f"vel={args.vel_loss_weight}",
        f"lr={args.lr}",
        f"bs={args.batch_size}",
    ])
    # Logga tutti gli argomenti (iperparametri, path, etc.)
    experiment.log_parameters(vars(args))

    print(f"\n S2P — Speech-to-Pose")
    print(f"   Modalità: {args.mode}")
    print(f"   Device:   {args.device}")
    print(f"   Dati:     {args.data_dir}\n")

    import json
    
    # 1. Carica i dati (genererà speaker_mapping.json se non esiste in cache)
    dataset = get_dataloaders(args)

    map_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "speaker_mapping.json")
    if os.path.exists(map_file):
        with open(map_file, "r") as f:
            speaker_map = json.load(f)
        args.num_speakers = len(speaker_map)
        print(f"   One-Hot Encoding attivato per {args.num_speakers} speaker")
    else:
        args.num_speakers = 0

    # 2. Costruisci il modello
    model = HeadPosePredictor(args)
    model = model.to(torch.device(args.device))

    # Conta parametri addestrabili
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Parametri: {trainable:,} addestrabili / {total:,} totali\n")

    # 3. Loss e Ottimizzatore
    criterion = PoseLoss(vel_weight=args.vel_loss_weight)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    # (nessuno scheduler — LR fisso a 1e-4 per tutta la durata)

    start_epoch = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        import re
        print(f"\nCaricamento pesi dal checkpoint: {args.resume_checkpoint}")
        # weights_only rimosso perché la versione py potremme non supportarlo o richiede default false in pytorch vecchi
        model.load_state_dict(torch.load(args.resume_checkpoint, map_location=args.device))
        match = re.search(r"epoch_(\d+)", args.resume_checkpoint)
        if match:
            start_epoch = int(match.group(1))
            print(f"L'allenamento riprenderà dall'epoca {start_epoch}")

    # 5. Training
    model = trainer(args, dataset["train"], dataset["valid"],
                    model, optimizer, criterion, experiment, start_epoch)

    # 6. Test
    test(args, model, dataset["test"])


if __name__ == "__main__":
    main()