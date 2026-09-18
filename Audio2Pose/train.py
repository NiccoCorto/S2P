"""
train.py - Script di addestramento per Audio2Pose (S2P)
Segue la struttura di train_S2L.py di s2l-s2d.
Features: early stopping, best model saving, LR fisso, logging CSV separato per pos/vel/var loss.

Branch VarLoss:
- FaceLoss: invece di MSE sugli angoli, calcola errore (pos + vel) sui 5023 vertici
  della faccia canonica ruotata tramite Forward Kinematics differenziabile.
- VarLoss: penalizza la differenza tra la varianza temporale della GT e della predizione
  (calcolata anch'essa sui vertici 3D). Incoraggia il modello a produrre sequenze con
  la stessa "ampiezza di oscillazione" della ground truth.
- One-Hot encoding: solo identità di train nel mapping; val/test sconosciuti → 1/N.
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


# ---------------------------------------------------------------------------
# Geometria
# ---------------------------------------------------------------------------

def axis_angle_to_matrix(r):
    """
    Converte vettori axis-angle in matrici di rotazione 3x3 (Formula di Rodrigues).
    Differenziabile su PyTorch.

    Args:
        r: Tensor (..., 3)
    Returns:
        R: Tensor (..., 3, 3)
    """
    from Audio2Pose.geometry import axis_angle_to_matrix as _aa2m
    return _aa2m(r)


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class FaceLoss(nn.Module):
    """
    Loss basata sui vertici 3D della faccia (Face Vertex Loss).
    
    Carica un template 3D della faccia (N_points, 3), applica matematicamente
    le rotazioni (Forward Kinematics differenziabile) e calcola:
      - pos_loss:  MSE sui vertici ruotati (posizione spaziale)
      - vel_loss:  MSE sulla velocità dei vertici tra frame consecutivi
    
    Supporta padding mask tramite `lengths`.
    """
    def __init__(self, vel_weight=2.0, canonical_face_path=None):
        super(FaceLoss, self).__init__()
        self.vel_weight = vel_weight

        if canonical_face_path is None:
            # cerca canonical_face.npy nella radice del progetto
            canonical_face_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "canonical_face.npy"
            )
        self.register_buffer(
            "canonical_face",
            torch.tensor(np.load(canonical_face_path), dtype=torch.float32)
        )

    def _rotate_face(self, angles):
        """
        Ruota la faccia canonica con i vettori axis-angle dati.
        
        Args:
            angles: Tensor (B, T, 3) — vettori axis-angle
        Returns:
            V: Tensor (B, T, N, 3) — vertici ruotati
        """
        from Audio2Pose.geometry import axis_angle_to_matrix
        R = axis_angle_to_matrix(angles)            # (B, T, 3, 3)
        # matmul: (B, T, N, 3) = (N, 3) @ (B, T, 3, 3)^T
        V = torch.matmul(self.canonical_face, R.transpose(-1, -2))
        return V

    def forward(self, predictions, target, lengths=None):
        """
        Args:
            predictions: Tensor (B, T, 3) — angoli axis-angle predetti
            target:      Tensor (B, T, 3) — angoli axis-angle target
            lengths:     LongTensor (B,)  — lunghezza reale di ogni sequenza
        Returns:
            total_loss, pos_loss (float), vel_loss (float)
        """
        B, T, C = target.shape

        # maschera frame-level (True = frame reale)
        if lengths is not None:
            idx    = torch.arange(T, device=target.device).unsqueeze(0)
            mask_2d = idx < lengths.unsqueeze(1)  # (B, T)
        else:
            mask_2d = torch.ones(B, T, dtype=torch.bool, device=target.device)

        # Forward Kinematics
        V_pred   = self._rotate_face(predictions)  # (B, T, N, 3)
        V_target = self._rotate_face(target)        # (B, T, N, 3)

        # --- Position Loss: MSE sui vertici sui frame reali ---
        mask_4d     = mask_2d.unsqueeze(-1).unsqueeze(-1).expand_as(V_target)
        pred_masked = V_pred[mask_4d].view(-1, 3)
        tgt_masked  = V_target[mask_4d].view(-1, 3)
        pos_loss    = F.mse_loss(pred_masked, tgt_masked)

        # --- Velocity Loss: differenza di vertici tra frame consecutivi ---
        pred_vel    = V_pred[:, 1:] - V_pred[:, :-1]    # (B, T-1, N, 3)
        tgt_vel     = V_target[:, 1:] - V_target[:, :-1]
        vel_mask    = mask_2d[:, 1:] & mask_2d[:, :-1]  # (B, T-1)
        vel_mask_4d = vel_mask.unsqueeze(-1).unsqueeze(-1).expand_as(pred_vel)

        if vel_mask_4d.any():
            pred_vel_m = pred_vel[vel_mask_4d].view(-1, 3)
            tgt_vel_m  = tgt_vel[vel_mask_4d].view(-1, 3)
            vel_loss   = F.mse_loss(pred_vel_m, tgt_vel_m)
        else:
            vel_loss = torch.tensor(0.0, device=target.device)

        total_loss = pos_loss + (self.vel_weight * vel_loss)

        return total_loss, pos_loss.detach().item(), vel_loss.detach().item()


class VarLoss(nn.Module):
    """
    Variance Loss — penalizza la differenza di varianza temporale tra GT e predizione.

    Misura quanto la sequenza "oscilla nel tempo" (varianza lungo la dimensione T)
    calcolata sui vertici 3D ruotati, coerentemente con FaceLoss.
    
    L'obiettivo è spingere il modello a produrre sequenze con la stessa ampiezza
    di movimento della ground truth, evitando la staticità centrale (predizioni
    costanti o quasi-costanti).
    
    Matematicamente:
        var_pred   = Var_t(V_pred)   shape (B, N, 3)   — varianza lungo T
        var_target = Var_t(V_target)  shape (B, N, 3)
        var_loss   = MSE(var_pred, var_target)
    
    La varianza è calcolata solo sui frame reali (rispettando la padding mask).
    """
    def __init__(self):
        super(VarLoss, self).__init__()

    def forward(self, V_pred, V_target, lengths=None):
        """
        Args:
            V_pred:   Tensor (B, T, N, 3) — vertici predetti ruotati
            V_target: Tensor (B, T, N, 3) — vertici target ruotati
            lengths:  LongTensor (B,)     — lunghezza reale di ogni sequenza

        Returns:
            var_loss: Tensor scalare
        """
        B, T, N, C = V_target.shape

        if lengths is not None:
            idx     = torch.arange(T, device=V_target.device).unsqueeze(0)
            mask_2d = idx < lengths.unsqueeze(1)  # (B, T)
        else:
            mask_2d = torch.ones(B, T, dtype=torch.bool, device=V_target.device)

        # Calcola varianza per ogni campione usando solo i frame reali.
        # Usiamo la formula corretta per la varianza mascherata: E[(x - mean)^2]
        var_pred_list   = []
        var_target_list = []

        for b in range(B):
            real_len = int(lengths[b].item()) if lengths is not None else T
            # (real_len, N, 3)
            vp = V_pred[b, :real_len]
            vt = V_target[b, :real_len]

            # varianza lungo la dimensione temporale: risultato (N, 3)
            var_pred_list.append(vp.var(dim=0, unbiased=False))
            var_target_list.append(vt.var(dim=0, unbiased=False))

        var_pred   = torch.stack(var_pred_list,   dim=0)  # (B, N, 3)
        var_target = torch.stack(var_target_list, dim=0)  # (B, N, 3)

        var_loss = F.mse_loss(var_pred, var_target)
        return var_loss


class CombinedLoss(nn.Module):
    """
    Loss totale = FaceLoss (pos + vel) + var_weight * VarLoss.
    
    Unifica FaceLoss e VarLoss in un unico modulo, condividendo il calcolo
    dei vertici ruotati (Forward Kinematics viene eseguito una sola volta).
    """
    def __init__(self, vel_weight=2.0, var_weight=1.0, canonical_face_path=None):
        super(CombinedLoss, self).__init__()
        self.vel_weight = vel_weight
        self.var_weight = var_weight
        self.face_loss_module = FaceLoss(
            vel_weight=vel_weight,
            canonical_face_path=canonical_face_path
        )
        self.var_loss_module = VarLoss()

    def forward(self, predictions, target, lengths=None):
        """
        Args:
            predictions: Tensor (B, T, 3) — angoli axis-angle predetti
            target:      Tensor (B, T, 3) — angoli axis-angle target
            lengths:     LongTensor (B,)  — lunghezze reali

        Returns:
            total_loss:  Tensor scalare (con grad)
            pos_l:       float — face pos loss (log)
            vel_l:       float — face vel loss (log)
            var_l:       float — var loss (log)
        """
        from Audio2Pose.geometry import axis_angle_to_matrix

        B, T, C = target.shape

        # maschera frame-level
        if lengths is not None:
            idx     = torch.arange(T, device=target.device).unsqueeze(0)
            mask_2d = idx < lengths.unsqueeze(1)
        else:
            mask_2d = torch.ones(B, T, dtype=torch.bool, device=target.device)

        # ---- Forward Kinematics (eseguito una volta sola) ----
        R_pred   = axis_angle_to_matrix(predictions)  # (B, T, 3, 3)
        R_target = axis_angle_to_matrix(target)        # (B, T, 3, 3)

        canonical_face = self.face_loss_module.canonical_face.to(target.device)
        V_pred   = torch.matmul(canonical_face, R_pred.transpose(-1, -2))   # (B, T, N, 3)
        V_target = torch.matmul(canonical_face, R_target.transpose(-1, -2)) # (B, T, N, 3)

        # ---- FaceLoss (pos + vel) ----
        # Position Loss
        mask_4d     = mask_2d.unsqueeze(-1).unsqueeze(-1).expand_as(V_target)
        pred_masked = V_pred[mask_4d].view(-1, 3)
        tgt_masked  = V_target[mask_4d].view(-1, 3)
        pos_loss    = F.mse_loss(pred_masked, tgt_masked)

        # Velocity Loss
        pred_vel    = V_pred[:, 1:] - V_pred[:, :-1]
        tgt_vel     = V_target[:, 1:] - V_target[:, :-1]
        vel_mask    = mask_2d[:, 1:] & mask_2d[:, :-1]
        vel_mask_4d = vel_mask.unsqueeze(-1).unsqueeze(-1).expand_as(pred_vel)

        if vel_mask_4d.any():
            pred_vel_m = pred_vel[vel_mask_4d].view(-1, 3)
            tgt_vel_m  = tgt_vel[vel_mask_4d].view(-1, 3)
            vel_loss   = F.mse_loss(pred_vel_m, tgt_vel_m)
        else:
            vel_loss = torch.tensor(0.0, device=target.device)

        # ---- VarLoss (varianza temporale sui vertici) ----
        var_loss = self.var_loss_module(V_pred, V_target, lengths=lengths)

        # ---- Loss totale ----
        total_loss = pos_loss + (self.vel_weight * vel_loss) + (self.var_weight * var_loss)

        return total_loss, pos_loss.detach().item(), vel_loss.detach().item(), var_loss.detach().item()


# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------

def setup_logging(log_path, start_epoch=0):
    """Crea la cartella di log e il file CSV per le loss."""
    os.makedirs(log_path, exist_ok=True)
    csv_path = os.path.join(log_path, "training_log.csv")
    mode = "a" if start_epoch > 0 else "w"
    with open(csv_path, mode, newline="") as f:
        if mode == "w":
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss", "train_pos_loss", "train_vel_loss", "train_var_loss",
                "val_loss",   "val_pos_loss",   "val_vel_loss",   "val_var_loss",
                "lr", "best"
            ])
    return csv_path


def log_epoch(csv_path, epoch,
              train_loss, train_pos, train_vel, train_var,
              val_loss,   val_pos,   val_vel,   val_var,
              lr, is_best):
    """Scrive una riga nel log CSV con le loss separate (pos, vel, var)."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{train_loss:.8f}", f"{train_pos:.8f}", f"{train_vel:.8e}", f"{train_var:.8e}",
            f"{val_loss:.8f}",   f"{val_pos:.8f}",   f"{val_vel:.8e}",   f"{val_var:.8e}",
            f"{lr:.8f}", "best" if is_best else ""
        ])


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

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
          f" ({'disabilitata' if args.vel_loss_weight == 0 else 'attiva'})")
    print(f"  Var loss weight: {args.var_loss_weight}"
          f" ({'disabilitata' if args.var_loss_weight == 0 else 'attiva'})")
    print(f"  Num speakers:    {args.num_speakers} (solo train)")
    print(f"  Checkpoint ogni: 20 epoche")
    print(f"{'='*60}\n")

    best_val_loss    = float("inf")
    patience_counter = 0

    for e in range(start_epoch, args.max_epoch):
        loss_log     = []
        pos_loss_log = []
        vel_loss_log = []
        var_loss_log = []

        # ---- fase di training ----
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {e+1}/{args.max_epoch} [TRAIN]")

        for i, (audio, pose_target, pose_lengths, audio_lengths, file_name, speaker_ids) in pbar:
            audio       = audio.to(device=args.device)
            pose_target = pose_target.to(device=args.device)
            pose_lengths = pose_lengths.to(device=args.device)
            speaker_ids = speaker_ids.to(device=args.device)

            optimizer.zero_grad()

            predictions = model(audio, pose_lengths=pose_lengths, speaker_ids=speaker_ids)

            loss, pos_l, vel_l, var_l = criterion(predictions, pose_target, lengths=pose_lengths)
            loss.backward()
            optimizer.step()

            loss_log.append(loss.item())
            pos_loss_log.append(pos_l)
            vel_loss_log.append(vel_l)
            var_loss_log.append(var_l)
            pbar.set_postfix({
                "Loss": f"{np.mean(loss_log):.6f}",
                "Pos":  f"{np.mean(pos_loss_log):.6f}",
                "Vel":  f"{np.mean(vel_loss_log):.2e}",
                "Var":  f"{np.mean(var_loss_log):.2e}"
            })

        train_loss     = np.mean(loss_log)
        train_pos_loss = np.mean(pos_loss_log)
        train_vel_loss = np.mean(vel_loss_log)
        train_var_loss = np.mean(var_loss_log)

        # ---- fase di validation ----
        valid_loss_log     = []
        valid_pos_loss_log = []
        valid_vel_loss_log = []
        valid_var_loss_log = []
        model.eval()

        with torch.no_grad():
            for audio, pose_target, pose_lengths, audio_lengths, file_name, speaker_ids in dev_loader:
                audio        = audio.to(device=args.device)
                pose_target  = pose_target.to(device=args.device)
                pose_lengths = pose_lengths.to(device=args.device)
                speaker_ids  = speaker_ids.to(device=args.device)

                predictions = model(audio, pose_lengths=pose_lengths, speaker_ids=speaker_ids)
                loss, pos_l, vel_l, var_l = criterion(predictions, pose_target, lengths=pose_lengths)

                valid_loss_log.append(loss.item())
                valid_pos_loss_log.append(pos_l)
                valid_vel_loss_log.append(vel_l)
                valid_var_loss_log.append(var_l)

        val_loss     = np.mean(valid_loss_log)     if valid_loss_log     else float("inf")
        val_pos_loss = np.mean(valid_pos_loss_log) if valid_pos_loss_log else float("inf")
        val_vel_loss = np.mean(valid_vel_loss_log) if valid_vel_loss_log else float("inf")
        val_var_loss = np.mean(valid_var_loss_log) if valid_var_loss_log else float("inf")

        current_lr = optimizer.param_groups[0]["lr"]

        # best model & early stopping
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(save_path, "best_audio2pose.pth"))
        else:
            patience_counter += 1

        # Log CSV
        log_epoch(csv_path, e + 1,
                  train_loss, train_pos_loss, train_vel_loss, train_var_loss,
                  val_loss,   val_pos_loss,   val_vel_loss,   val_var_loss,
                  current_lr, is_best)

        # Log CometML
        experiment.log_metrics({
            "train_loss":     train_loss,
            "train_pos_loss": train_pos_loss,
            "train_vel_loss": train_vel_loss,
            "train_var_loss": train_var_loss,
            "val_loss":       val_loss,
            "val_pos_loss":   val_pos_loss,
            "val_vel_loss":   val_vel_loss,
            "val_var_loss":   val_var_loss,
            "learning_rate":  current_lr
        }, step=e + 1)

        # print riepilogo epoca
        best_marker = " ★ BEST" if is_best else ""
        print(f"  Epoca {e+1}/{args.max_epoch} | "
              f"Train: {train_loss:.6f} "
              f"(Pos: {train_pos_loss:.6f} | Vel: {train_vel_loss:.2e} | Var: {train_var_loss:.2e}) | "
              f"Val: {val_loss:.6f} "
              f"(Pos: {val_pos_loss:.6f} | Vel: {val_vel_loss:.2e} | Var: {val_var_loss:.2e}) | "
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


# ---------------------------------------------------------------------------
# Test / Inference
# ---------------------------------------------------------------------------

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
        audio        = audio.to(device=args.device)
        pose_lengths = pose_lengths.to(device=args.device)
        speaker_ids  = speaker_ids.to(device=args.device)

        predictions = model(audio, pose_lengths=pose_lengths, speaker_ids=speaker_ids)

        for i, fname in enumerate(file_names):
            real_len = pose_lengths[i].item()
            pred_i   = predictions[i, :real_len, :]
            save_name = os.path.join(result_path, fname.replace(".wav", ".npy"))
            np.save(save_name, pred_i.detach().cpu().numpy())

    print(f"\n Test completato! Pose generate salvate in: {result_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    # Avviso invalidazione cache
    cache_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "s2p_lazy_cache.pkl")
    if os.path.exists(cache_file):
        print(f"\n[WARN] Branch VarLoss: speaker_mapping viene rigenerato SOLO da identità di train.")
        print(f"       La cache esistente ({cache_file}) potrebbe essere incompatibile.")
        print(f"       Se hai problemi, elimina la cache e riesegui.\n")

    # Inizializza Comet ML
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

    auto_name = (f"faceloss_var{args.var_loss_weight}_vel{args.vel_loss_weight}"
                 f"_L{args.num_layers}_h{args.hidden_dim}_ep{args.max_epoch}")
    exp_name  = args.exp_name if args.exp_name else auto_name
    experiment.set_name(exp_name)
    experiment.add_tags([
        "VarLoss", "FaceLoss", "OneHot",
        f"layers={args.num_layers}",
        f"hidden={args.hidden_dim}",
        f"vel={args.vel_loss_weight}",
        f"var={args.var_loss_weight}",
        f"lr={args.lr}",
        f"bs={args.batch_size}",
    ])
    experiment.log_parameters(vars(args))

    print(f"\n S2P — Speech-to-Pose (VarLoss Branch)")
    print(f"   Modalità: {args.mode}")
    print(f"   Device:   {args.device}")
    print(f"   Dati:     {args.data_dir}\n")

    import json

    # 1. Carica i dati (genera speaker_mapping.json solo con identità di train)
    dataset = get_dataloaders(args)

    map_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "speaker_mapping.json")
    if os.path.exists(map_file):
        with open(map_file, "r") as f:
            speaker_map = json.load(f)
        args.num_speakers = len(speaker_map)
        print(f"   One-Hot Encoding attivato per {args.num_speakers} speaker (solo train)")
        print(f"   Speaker di val/test non visti → vettore uniforme 1/{args.num_speakers}")
    else:
        args.num_speakers = 0
        print(f"   One-Hot Encoding disabilitato (speaker_mapping.json non trovato)")

    # 2. Costruisci il modello
    model = HeadPosePredictor(args)
    model = model.to(torch.device(args.device))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"   Parametri: {trainable:,} addestrabili / {total:,} totali\n")

    # 3. Loss combinata (FaceLoss + VarLoss)
    canonical_face_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "canonical_face.npy"
    )
    criterion = CombinedLoss(
        vel_weight=args.vel_loss_weight,
        var_weight=args.var_loss_weight,
        canonical_face_path=canonical_face_path
    )

    # 4. Ottimizzatore
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    start_epoch = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        import re
        print(f"\nCaricamento pesi dal checkpoint: {args.resume_checkpoint}")
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