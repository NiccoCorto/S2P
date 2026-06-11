"""
model.py - Rete neurale Audio2Pose per S2P
Architettura: Wav2Vec2 (frozen) → LSTM bidirezionale → FC → 3 angoli (Pitch, Yaw, Roll)
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

        # Iperparametri (da args o default)
        hidden_dim = getattr(args, "hidden_dim", 256) if args else 256
        num_layers = getattr(args, "num_layers", 2) if args else 2
        dropout = getattr(args, "dropout", 0.2) if args else 0.2

        # 1. Audio Encoder (Wav2Vec2) — pesi congelati
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        # 2. Layer Normalization sull'input (stabilizza il training)
        self.layer_norm = nn.LayerNorm(768)

        # 3. Layer temporale (LSTM bidirezionale)
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 4. Dropout prima dell'output
        self.dropout = nn.Dropout(p=dropout)

        # 5. Output: 3 valori (Pitch, Yaw, Roll) — compatibili con cv2.Rodrigues()
        self.fc = nn.Linear(hidden_dim * 2, 3)  # *2 perché bidirezionale

    def forward(self, audio_input, target_seq_len=None):
        """
        Args:
            audio_input: Tensor audio raw (Batch, Samples) pre-processato da Wav2Vec2Processor
            target_seq_len: Numero di frame del video target. Se specificato,
                           interpola le features audio per allinearsi esattamente.
        
        Returns:
            pose_pred: (Batch, target_seq_len, 3) — Pitch, Yaw, Roll per frame
        """
        # 1. Estrai le feature dall'audio (escono a ~50 FPS)
        with torch.no_grad():
            features = self.audio_encoder(audio_input).last_hidden_state  # (B, Seq_Audio, 768)

        # 2. Layer Normalization
        features = self.layer_norm(features)

        # 3. Interpolazione Lineare (la "magia di Federico")
        # Se sappiamo quanti frame ha il video, "stiriamo" l'audio per renderlo identico
        if target_seq_len is not None and features.size(1) != target_seq_len:
            features = features.transpose(1, 2)  # (B, 768, Seq_Audio)
            features = F.interpolate(
                features, size=target_seq_len, mode='linear', align_corners=True
            )
            features = features.transpose(1, 2)  # (B, target_seq_len, 768)

        # 4. LSTM
        lstm_out, _ = self.lstm(features)

        # 5. Dropout
        lstm_out = self.dropout(lstm_out)

        # 6. Output: 3 angoli per frame
        pose_pred = self.fc(lstm_out)  # (B, Seq, 3)

        return pose_pred