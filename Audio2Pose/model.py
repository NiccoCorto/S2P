"""
model.py - Rete neurale Audio2Pose per S2P
Architettura: Wav2Vec2 (completamente frozen) → LSTM bidirezionale → FC → 3 angoli (Pitch, Yaw, Roll)
Ispirata a Speech2Land di s2l-s2d, adattata per predire rotazioni della testa.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class HeadPosePredictor(nn.Module):
    """Predice i 3 angoli di rotazione della testa (Pitch, Yaw, Roll) da audio.
    
    Args:
        args: Namespace con hidden_dim, num_layers, dropout.
              Se None, usa i default.
    """

    def __init__(self, args=None):
        super(HeadPosePredictor, self).__init__()

        # iperparametri (da args o default)
        hidden_dim = getattr(args, "hidden_dim", 256) if args else 256
        # num_layers: default 2. Aumentare a 3-4 per catturare pattern temporali più complessi
        # (consigliato con dataset grande come EMOTE — sperimentare con --num_layers 3 o 4)
        num_layers = getattr(args, "num_layers", 2) if args else 2
        # dropout al valore PyTorch di default (0.1) — allineato con config.py
        dropout = getattr(args, "dropout", 0.1) if args else 0.1

        # Audio Encoder (Wav2Vec2) — completamente frozen (no fine-tuning)
        # tutti i parametri (CNN feature extractor + tutti i transformer blocks)
        # vengono congelati: il modello impara solo a partire dalle feature fisse.
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        self.num_speakers = getattr(args, "num_speakers", 0) if args else 0
        input_size = 768 + self.num_speakers

        # Layer Normalization sull'input (stabilizza il training)
        self.layer_norm = nn.LayerNorm(768)

        # Layer temporale (LSTM bidirezionale)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # dropout prima dell'output
        self.dropout = nn.Dropout(p=dropout)

        # output: 3 valori (Pitch, Yaw, Roll) — compatibili con cv2.Rodrigues()
        self.fc = nn.Linear(hidden_dim * 2, 3)  # *2 perché bidirezionale

    def forward(self, audio_input, pose_lengths=None, speaker_ids=None):
        """
        Args:
            audio_input:  Tensor audio raw (Batch, Samples) pre-processato da Wav2Vec2Processor
            pose_lengths: LongTensor (Batch,) con la lunghezza reale (pre-padding) di ogni
                          sequenza pose nel batch. Se specificato, le feature audio vengono
                          interpolate alla lunghezza del campione più lungo (max reale, non padded),
                          garantendo che frame paddati non vengano mai predetti né inclusi in loss.
            speaker_ids:  LongTensor (Batch,) con gli ID univoci dei parlanti per il one-hot.

        Returns:
            pose_pred: (Batch, max_real_len, 3) — Pitch, Yaw, Roll per frame
        """
        # estrai le feature dall'audio (escono a ~50 FPS)
        with torch.no_grad():
            features = self.audio_encoder(audio_input).last_hidden_state  # (B, Seq_Audio, 768)

        # layer normalization
        features = self.layer_norm(features)

        # interpolazione lineare verso la lunghezza reale massima del batch
        # pose_lengths.max() = lunghezza del campione più lungo (nessun padding aggiunto sopra)
        if pose_lengths is not None:
            target_len = int(pose_lengths.max().item())
        else:
            target_len = features.size(1)

        if features.size(1) != target_len:
            features = features.transpose(1, 2)  # (B, 768, Seq_Audio)
            features = F.interpolate(
                features, size=target_len, mode='linear', align_corners=True
            )
            features = features.transpose(1, 2)  # (B, target_len, 768)

        # Aggiunta One-Hot Encoder (se num_speakers > 0 e speaker_ids forniti)
        if self.num_speakers > 0 and speaker_ids is not None:
            # Crea vettore one-hot di dimensione (B, num_speakers)
            one_hot = F.one_hot(speaker_ids, num_classes=self.num_speakers).float()
            # Espandi per la dimensione temporale: (B, 1, num_speakers) -> (B, target_len, num_speakers)
            one_hot_expanded = one_hot.unsqueeze(1).expand(-1, target_len, -1)
            # Concatena lungo l'ultima dimensione
            features = torch.cat([features, one_hot_expanded], dim=-1)

        # lstm
        lstm_out, _ = self.lstm(features)

        # dropout
        lstm_out = self.dropout(lstm_out)

        # output: 3 angoli per frame
        pose_pred = self.fc(lstm_out)  # (B, max_real_len, 3)

        return pose_pred