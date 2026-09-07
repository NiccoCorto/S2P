"""
model.py - Rete neurale Audio2Pose VAE per S2P
Architettura: 
- Pose Encoder: LSTM per mappare la GT pose nello spazio latente Z (stile globale)
- Audio Encoder: Wav2Vec2 (completamente frozen)
- Decoder: LSTM bidirezionale che prende [Audio Features, Z] → 3 angoli (Pitch, Yaw, Roll)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class HeadPoseVAE(nn.Module):
    """Variational Autoencoder per la predizione probabilistica della posa."""

    def __init__(self, args=None):
        super(HeadPoseVAE, self).__init__()

        # iperparametri (da args o default)
        hidden_dim = getattr(args, "hidden_dim", 256) if args else 256
        latent_dim = getattr(args, "latent_dim", 64) if args else 64
        num_layers = getattr(args, "num_layers", 2) if args else 2
        dropout = getattr(args, "dropout", 0.1) if args else 0.1

        # ---------------------------------------------------------
        # 1. Audio Encoder (Wav2Vec2) - FROZEN
        # ---------------------------------------------------------
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        self.layer_norm = nn.LayerNorm(768)

        # ---------------------------------------------------------
        # 2. Pose Encoder (GT -> Latent Space)
        # ---------------------------------------------------------
        self.pose_encoder_lstm = nn.LSTM(
            input_size=3,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # ---------------------------------------------------------
        # 3. Decoder
        # ---------------------------------------------------------
        self.decoder_lstm = nn.LSTM(
            input_size=768 + latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc_out = nn.Linear(hidden_dim * 2, 3)

    def encode_pose(self, pose, lengths=None):
        """
        Codifica l'intera sequenza di Ground Truth in una distribuzione latente globale.
        Args:
            pose: (Batch, Seq_Pose, 3)
            lengths: (Batch,) con la lunghezza reale di ogni sequenza
        Returns:
            mu, logvar: (Batch, latent_dim)
        """
        out, _ = self.pose_encoder_lstm(pose)  # (B, T, hidden_dim*2)
        
        if lengths is not None:
            # Masked average pooling
            B, T, C = out.shape
            mask = torch.arange(T, device=out.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T)
            mask = mask.unsqueeze(-1).float()  # (B, T, 1)
            summed = torch.sum(out * mask, dim=1)  # (B, C)
            pooled = summed / lengths.unsqueeze(1).float()  # (B, C)
        else:
            pooled = out.mean(dim=1)  # (B, hidden_dim*2)
        
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Trick di riparametrizzazione: z = mu + std * epsilon"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # In validation/test, se forniamo mu, usiamo solo mu (niente rumore)
            # per avere un comportamento deterministico condizionato,
            # anche se in test reale non avremo proprio il target.
            return mu

    def forward(self, audio_input, pose_lengths=None, target_pose=None):
        """
        Args:
            audio_input:  Tensor audio raw (Batch, Samples)
            pose_lengths: Lunghezza reale (pre-padding) di ogni sequenza pose
            target_pose:  Tensor (Batch, max_real_len, 3) della Ground Truth. 
                          Se None (inferenza pura), si campiona z casualmente.
        
        Returns:
            pose_pred: (Batch, max_real_len, 3)
            mu:        (Batch, latent_dim)
            logvar:    (Batch, latent_dim)
        """
        B = audio_input.size(0)

        # ==========================================
        # 1. Feature Audio
        # ==========================================
        with torch.no_grad():
            audio_feat = self.audio_encoder(audio_input).last_hidden_state  # (B, Seq_Audio, 768)
        audio_feat = self.layer_norm(audio_feat)

        if pose_lengths is not None:
            target_len = int(pose_lengths.max().item())
        else:
            target_len = audio_feat.size(1)

        if audio_feat.size(1) != target_len:
            audio_feat = audio_feat.transpose(1, 2)
            audio_feat = F.interpolate(
                audio_feat, size=target_len, mode='linear', align_corners=True
            )
            audio_feat = audio_feat.transpose(1, 2)  # (B, target_len, 768)

        # ==========================================
        # 2. VAE - Spazio Latente (Z)
        # ==========================================
        if target_pose is not None:
            # Training: il Pose Encoder sputa la distribuzione reale dello stile
            mu, logvar = self.encode_pose(target_pose, lengths=pose_lengths)
            z = self.reparameterize(mu, logvar)
        else:
            # Inference pura (senza target): 
            # peschiamo uno stile casuale da una normale standard N(0,1)
            mu = torch.zeros(B, self.fc_mu.out_features, device=audio_feat.device)
            logvar = torch.zeros(B, self.fc_logvar.out_features, device=audio_feat.device)
            z = torch.randn(B, self.fc_mu.out_features, device=audio_feat.device)

        # ==========================================
        # 3. Decoder
        # ==========================================
        # z: (B, latent_dim) -> (B, target_len, latent_dim)
        z_expanded = z.unsqueeze(1).expand(-1, target_len, -1)
        
        # Input decoder: [Audio, Z]
        decoder_input = torch.cat([audio_feat, z_expanded], dim=-1)  # (B, target_len, 768 + latent_dim)

        lstm_out, _ = self.decoder_lstm(decoder_input)
        lstm_out = self.dropout(lstm_out)
        pose_pred = self.fc_out(lstm_out)  # (B, target_len, 3)

        return pose_pred, mu, logvar

# Mantengo il vecchio nome HeadPosePredictor come alias per non rompere l'import negli script
HeadPosePredictor = HeadPoseVAE