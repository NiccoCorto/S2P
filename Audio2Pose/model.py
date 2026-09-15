"""
model.py - Rete neurale Audio2Pose per S2P
Architettura: Wav2Vec2 (completamente frozen) → LSTM bidirezionale → FC → 3 angoli (Pitch, Yaw, Roll)
Ispirata a Speech2Land di s2l-s2d, adattata per predire rotazioni della testa.

Branch VarLoss:
- One-Hot encoding degli speaker (solo identità di train nel mapping)
- speaker_id == -1 → vettore uniforme 1/N (prior uniforme su tutti i soggetti noti)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class HeadPosePredictor(nn.Module):
    """Predice i 3 angoli di rotazione della testa (Pitch, Yaw, Roll) da audio.
    
    Supporta One-Hot conditioning per identità speaker.
    I soggetti sconosciuti (speaker_id == -1, es. val/test non visti in train)
    ricevono un vettore uniforme 1/N al posto del one-hot puntuale, equivalente
    a un prior uniforme che non favorisce nessuna identità in particolare.

    Args:
        args: Namespace con hidden_dim, num_layers, dropout, num_speakers.
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

        # numero di speaker noti (solo quelli di train)
        # se 0, il one-hot viene disabilitato
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

    def _build_speaker_vector(self, speaker_ids, target_len):
        """
        Costruisce il vettore di condizionamento speaker (B, target_len, num_speakers).

        - speaker_id >= 0 → one-hot standard sull'ID conosciuto
        - speaker_id == -1 → vettore uniforme 1/N (prior uniforme, nessuna identità favorita)
          Corrisponde a val/test con identità non viste durante il training.
        """
        B = speaker_ids.size(0)
        N = self.num_speakers
        # inizializza con vettore uniforme 1/N (default per sconosciuti)
        speaker_vec = torch.full((B, N), fill_value=1.0 / N,
                                 device=speaker_ids.device, dtype=torch.float32)

        # sostituisci con one-hot per gli ID validi (>= 0)
        known_mask = speaker_ids >= 0
        if known_mask.any():
            known_ids  = speaker_ids[known_mask]
            one_hot_known = F.one_hot(known_ids, num_classes=N).float()
            speaker_vec[known_mask] = one_hot_known

        # espandi temporalmente: (B, N) → (B, target_len, N)
        return speaker_vec.unsqueeze(1).expand(-1, target_len, -1)

    def forward(self, audio_input, pose_lengths=None, speaker_ids=None):
        """
        Args:
            audio_input:  Tensor audio raw (Batch, Samples) pre-processato da Wav2Vec2Processor
            pose_lengths: LongTensor (Batch,) con la lunghezza reale (pre-padding) di ogni
                          sequenza pose nel batch. Se specificato, le feature audio vengono
                          interpolate alla lunghezza del campione più lungo (max reale, non padded),
                          garantendo che frame paddati non vengano mai predetti né inclusi in loss.
            speaker_ids:  LongTensor (Batch,) con gli ID dei parlanti.
                          -1 = identità sconosciuta (val/test non visti in train).

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

        # Aggiunta condizionamento speaker tramite One-Hot / Vettore Uniforme
        if self.num_speakers > 0 and speaker_ids is not None:
            speaker_vec = self._build_speaker_vector(speaker_ids, target_len)
            # Concatena lungo l'ultima dimensione: (B, target_len, 768 + num_speakers)
            features = torch.cat([features, speaker_vec], dim=-1)

        # lstm
        lstm_out, _ = self.lstm(features)

        # dropout
        lstm_out = self.dropout(lstm_out)

        # output: 3 angoli per frame
        pose_pred = self.fc(lstm_out)  # (B, max_real_len, 3)

        return pose_pred