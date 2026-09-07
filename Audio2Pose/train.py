"""
train.py - Script di addestramento per Audio2Pose VAE (S2P)
Implementa il Variational Autoencoder, KLD Loss e meccanismo di KL Annealing
per prevenire il Posterior Collapse.
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
from model import HeadPosePredictor  # Alias di HeadPoseVAE


class PoseLoss(nn.Module):
    """
    Funzione di errore VAE per la testa.
    Calcola l'errore sulla posizione (MSE), sulla velocità (MSE) e 
    la KL Divergence per lo spazio latente.
    """
    def __init__(self, vel_weight=2.0):
        super(PoseLoss, self).__init__()
        self.vel_weight = vel_weight

    def forward(self, predictions, target, mu, logvar, current_kld_weight=0.0, lengths=None):
        B, T, C = target.shape

        if lengths is not None:
            idx = torch.arange(T, device=target.device).unsqueeze(0)
            mask_2d = idx < lengths.unsqueeze(1)
        else:
            mask_2d = torch.ones(B, T, dtype=torch.bool, device=target.device)

        # --- Position Loss ---
        mask_3d = mask_2d.unsqueeze(-1).expand_as(target)
        pred_masked = predictions[mask_3d].view(-1, C)
        tgt_masked  = target[mask_3d].view(-1, C)
        pos_loss = F.mse_loss(pred_masked, tgt_masked)

        # --- Velocity Loss ---
        pred_vel = predictions[:, 1:, :] - predictions[:, :-1, :]
        tgt_vel  = target[:, 1:, :] - target[:, :-1, :]
        vel_mask    = mask_2d[:, 1:] & mask_2d[:, :-1]
        vel_mask_3d = vel_mask.unsqueeze(-1).expand_as(pred_vel)

        if vel_mask_3d.any():
            pred_vel_m = pred_vel[vel_mask_3d].view(-1, C)
            tgt_vel_m  = tgt_vel[vel_mask_3d].view(-1, C)
            vel_loss = F.mse_loss(pred_vel_m, tgt_vel_m)
        else:
            vel_loss = torch.tensor(0.0, device=target.device)

        # --- KLD Loss ---
        # kld_loss = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = kld_loss / B  # normalizza sulla batch size

        total_loss = pos_loss + (self.vel_weight * vel_loss) + (current_kld_weight * kld_loss)

        return total_loss, pos_loss.detach().item(), vel_loss.detach().item(), kld_loss.detach().item()


def setup_logging(log_path, start_epoch=0):
    os.makedirs(log_path, exist_ok=True)
    csv_path = os.path.join(log_path, "training_log.csv")
    mode = "a" if start_epoch > 0 else "w"
    with open(csv_path, mode, newline="") as f:
        if mode == "w":
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_pos", "train_vel", "train_kld",
                             "val_loss", "val_pos", "val_vel", "val_kld", "lr", "best"])
    return csv_path


def log_epoch(csv_path, epoch, train_loss, train_pos, train_vel, train_kld,
              val_loss, val_pos, val_vel, val_kld, lr, is_best):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{train_loss:.8f}", f"{train_pos:.8f}", f"{train_vel:.8f}", f"{train_kld:.8f}",
            f"{val_loss:.8f}", f"{val_pos:.8f}", f"{val_vel:.8f}", f"{val_kld:.8f}",
            f"{lr:.8f}", "best" if is_best else ""
        ])


def trainer(args, train_loader, dev_loader, model, optimizer, criterion, experiment, start_epoch=0):
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    csv_path = setup_logging(args.log_path, start_epoch)

    print(f"\n{'='*60}")
    print(f"Inizio addestramento VAE su {args.device}")
    print(f"  Epoche max:      {args.max_epoch}")
    print(f"  Latent dim:      {args.latent_dim}")
    print(f"  Max KLD weight:  {args.kld_weight}")
    print(f"  Anneal epochs:   {args.anneal_epochs}")
    print(f"{'='*60}\n")

    best_val_metric = float("inf")
    patience_counter = 0

    for e in range(start_epoch, args.max_epoch):
        # KL Annealing Logic
        if args.anneal_epochs > 0:
            kld_ratio = min(1.0, (e + 1) / args.anneal_epochs)
            current_kld_weight = args.kld_weight * kld_ratio
        else:
            current_kld_weight = args.kld_weight

        loss_log, pos_log, vel_log, kld_log = [], [], [], []

        # Train
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {e+1}/{args.max_epoch} [TRAIN]")

        for i, (audio, pose_target, pose_lengths, audio_lengths, file_name) in pbar:
            audio, pose_target, pose_lengths = audio.to(args.device), pose_target.to(args.device), pose_lengths.to(args.device)
            optimizer.zero_grad()

            # Forward passando la GT per estrarre lo stile
            predictions, mu, logvar = model(audio, pose_lengths=pose_lengths, target_pose=pose_target)

            loss, pos_l, vel_l, kld_l = criterion(
                predictions, pose_target, mu, logvar, current_kld_weight, lengths=pose_lengths
            )
            loss.backward()
            optimizer.step()

            loss_log.append(loss.item())
            pos_log.append(pos_l)
            vel_log.append(vel_l)
            kld_log.append(kld_l)
            
            pbar.set_postfix({
                "L": f"{np.mean(loss_log):.4f}",
                "Pos": f"{np.mean(pos_log):.4f}",
                "KLD": f"{np.mean(kld_log):.4f} (w={current_kld_weight:.4f})"
            })

        train_loss = np.mean(loss_log)
        train_pos_loss = np.mean(pos_log)
        train_vel_loss = np.mean(vel_log)
        train_kld_loss = np.mean(kld_log)

        # Validation
        val_loss_log, val_pos_log, val_vel_log, val_kld_log = [], [], [], []
        model.eval()

        with torch.no_grad():
            for audio, pose_target, pose_lengths, audio_lengths, file_name in dev_loader:
                audio, pose_target, pose_lengths = audio.to(args.device), pose_target.to(args.device), pose_lengths.to(args.device)

                predictions, mu, logvar = model(audio, pose_lengths=pose_lengths, target_pose=pose_target)
                loss, pos_l, vel_l, kld_l = criterion(
                    predictions, pose_target, mu, logvar, current_kld_weight, lengths=pose_lengths
                )
                val_loss_log.append(loss.item())
                val_pos_log.append(pos_l)
                val_vel_log.append(vel_l)
                val_kld_log.append(kld_l)

        val_loss = np.mean(val_loss_log) if val_loss_log else float("inf")
        val_pos = np.mean(val_pos_log) if val_pos_log else float("inf")
        val_vel = np.mean(val_vel_log) if val_vel_log else float("inf")
        val_kld = np.mean(val_kld_log) if val_kld_log else float("inf")

        current_lr = optimizer.param_groups[0]["lr"]

        # Early stopping e best model basati su pos+vel, ignorando KLD che cambia col tempo
        val_metric = val_pos + val_vel
        is_best = val_metric < best_val_metric
        if is_best:
            best_val_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, "best_audio2pose.pth"))
        else:
            patience_counter += 1

        log_epoch(csv_path, e + 1,
                  train_loss, train_pos_loss, train_vel_loss, train_kld_loss,
                  val_loss, val_pos, val_vel, val_kld,
                  current_lr, is_best)
        
        experiment.log_metrics({
            "train_loss": train_loss, "train_pos": train_pos_loss, "train_vel": train_vel_loss, "train_kld": train_kld_loss,
            "val_loss": val_loss, "val_pos": val_pos, "val_vel": val_vel, "val_kld": val_kld,
            "kld_weight": current_kld_weight
        }, step=e + 1)

        print(f"  Epoca {e+1}/{args.max_epoch} | T_Loss: {train_loss:.4f} (KLD: {train_kld_loss:.4f}) | V_Loss: {val_loss:.4f} | Pat: {patience_counter}/{args.patience}")

        if (e + 1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"audio2pose_epoch_{e+1}.pth"))

        if patience_counter >= args.patience:
            print("Early stopping!")
            break

    torch.save(model.state_dict(), os.path.join(save_path, f"audio2pose_final_epoch_{e+1}.pth"))
    return model


@torch.no_grad()
def test(args, model, test_loader):
    print("\nInizio fase di Test (Inferenza casuale dal VAE)...")
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)

    best_path = os.path.join(args.save_path, "best_audio2pose.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=args.device))

    model = model.to(torch.device(args.device))
    model.eval()

    for audio, pose_target, pose_lengths, audio_lengths, file_names in tqdm(test_loader, desc="Testing"):
        audio, pose_lengths = audio.to(args.device), pose_lengths.to(args.device)

        # Generazione VAE: NON passiamo target_pose per forzare il campionamento da N(0,1)
        predictions, _, _ = model(audio, pose_lengths=pose_lengths, target_pose=None)

        for i, fname in enumerate(file_names):
            real_len = pose_lengths[i].item()
            pred_i = predictions[i, :real_len, :]
            np.save(os.path.join(result_path, fname.replace(".wav", ".npy")), pred_i.cpu().numpy())

def main():
    args = get_args()

    if args.comet_experiment_key:
        experiment = comet_ml.ExistingExperiment(
            previous_experiment=args.comet_experiment_key, project_name="audio2pose"
        )
    else:
        experiment = comet_ml.Experiment(project_name="audio2pose")
        
    exp_name = args.exp_name if args.exp_name else f"VAE_Z{args.latent_dim}_vel{args.vel_loss_weight}"
    experiment.set_name(exp_name)
    experiment.add_tags(["VAE", f"Z={args.latent_dim}", f"KLw={args.kld_weight}"])
    experiment.log_parameters(vars(args))

    model = HeadPosePredictor(args).to(args.device)
    criterion = PoseLoss(vel_weight=args.vel_loss_weight)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr
    )

    dataset = get_dataloaders(args)
    start_epoch = 0

    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        model.load_state_dict(torch.load(args.resume_checkpoint, map_location=args.device))
        import re
        match = re.search(r"epoch_(\d+)", args.resume_checkpoint)
        if match: start_epoch = int(match.group(1))

    model = trainer(args, dataset["train"], dataset["valid"], model, optimizer, criterion, experiment, start_epoch)
    # Il test automatico a fine addestramento è stato rimosso. 
    # Le inferenze sul test set andranno eseguite tramite script separati.


if __name__ == "__main__":
    main()