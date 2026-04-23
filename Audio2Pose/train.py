import numpy as np
import argparse
from tqdm import tqdm
import os
import sys

import torch
import torch.nn as nn

# Aggiungiamo la cartella principale al path per trovare il dataloader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import get_dataloaders
from model import HeadPosePredictor

class PoseLoss(nn.Module):
    """
    Funzione di errore personalizzata per la testa.
    Calcola l'errore sulla posizione esatta (MSE) + l'errore sulla velocità del movimento (per renderlo fluido).
    """
    def __init__(self):
        super(PoseLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, predictions, target):
        # 1. Errore di posizione (quanto siamo lontani dagli angoli reali)
        pos_loss = self.mse(predictions, target)
        
        # 2. Errore di velocità (differenza tra il frame attuale e il precedente)
        # Questo costringe il modello a non fare movimenti "a scatti"
        prediction_shift = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_shift = target[:, 1:, :] - target[:, :-1, :]
        vel_loss = self.mse(prediction_shift, target_shift)

        # Sommiamo gli errori (diamo più peso alla posizione)
        return pos_loss + (2.0 * vel_loss)


def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch):
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f"Inizio addestramento su {args.device} per {epoch} epoche...")

    for e in range(epoch):
        loss_log = []
        
        # --- FASE DI TRAINING ---
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {e+1}/{epoch} [TRAIN]")
        
        for i, (audio, landmarks_target, file_name) in pbar:
            # Sposta i dati sulla scheda video (GPU)
            audio = audio.to(device=args.device)
            landmarks_target = landmarks_target.to(device=args.device)
            
            optimizer.zero_grad()
            
            # Passiamo la lunghezza reale delle pose (landmarks_target.size(1)) al modello!
            predictions = model(audio, target_seq_len=landmarks_target.size(1))
            
            # Allinea la lunghezza di audio e video (a volte differiscono di 1 frame)
            min_seq_len = min(predictions.size(1), landmarks_target.size(1))
            predictions = predictions[:, :min_seq_len, :]
            landmarks_target = landmarks_target[:, :min_seq_len, :]
            
            # Calcola l'errore e aggiorna i pesi
            loss = criterion(predictions, landmarks_target)
            loss.backward()
            optimizer.step()
            
            loss_log.append(loss.item())
            pbar.set_postfix({"Loss": f"{np.mean(loss_log):.6f}"})

        # --- FASE DI VALIDATION ---
        valid_loss_log = []
        model.eval()
        
        with torch.no_grad(): # Non calcoliamo i gradienti per risparmiare memoria
            for audio, landmarks_target, file_name in dev_loader:
                audio = audio.to(device=args.device)
                landmarks_target = landmarks_target.to(device=args.device)

                predictions = model(audio)
                
                min_seq_len = min(predictions.size(1), landmarks_target.size(1))
                predictions = predictions[:, :min_seq_len, :]
                landmarks_target = landmarks_target[:, :min_seq_len, :]

                loss = criterion(predictions, landmarks_target)
                valid_loss_log.append(loss.item())

        current_val_loss = np.mean(valid_loss_log)
        print(f"--> Fine Epoca {e+1} | Train Loss: {np.mean(loss_log):.6f} | Val Loss: {current_val_loss:.6f}")

        # Salva il modello ogni 50 epoche o all'ultima epoca
        if (e + 1) == epoch or (e + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'audio2pose_epoch_{e+1}.pth'))

    return model


@torch.no_grad()
def test(args, model, test_loader, epoch):
    print("\nInizio fase di Test sui dati mai visti...")
    result_path = args.result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Carica i pesi del modello appena addestrato
    model.load_state_dict(torch.load(os.path.join(args.save_path, f'audio2pose_epoch_{epoch}.pth')))
    model = model.to(torch.device(args.device))
    model.eval()

    for audio, landmarks_target, file_name in tqdm(test_loader, desc="Testing"):
        audio = audio.to(device=args.device)
        
        predictions = model(audio)
        predictions = predictions.squeeze() # Rimuove la dimensione del batch
        
        # Salva la predizione come file .npy per poterla usare dopo con ScanTalk!
        save_name = os.path.join(result_path, file_name[0].replace(".wav", ".npy"))
        np.save(save_name, predictions.detach().cpu().numpy())
        
    print(f"Test completato! Le pose generate sono salvate in: {result_path}")


def main():
    parser = argparse.ArgumentParser(description='Audio2Pose: Generazione movimenti testa da Audio')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default="Saves", help='Cartella salvataggio modello')
    parser.add_argument("--result_path", type=str, default="Results", help='Cartella salvataggio predizioni .npy')
    args = parser.parse_args()

    # 1. Costruisci il modello
    model = HeadPosePredictor()
    model = model.to(torch.device(args.device))

    # 2. Definisci la funzione di Loss e l'Ottimizzatore
    criterion = PoseLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # 3. Carica i dati usando il tuo dataloader modulare
    dataset = get_dataloaders(args)

    # 4. Avvia il Training
    model = trainer(args, dataset["train"], dataset["valid"], model, optimizer, criterion, epoch=args.max_epoch)

    # 5. Avvia il Test alla fine
    test(args, model, dataset["test"], epoch=args.max_epoch)

if __name__ == "__main__":
    main()