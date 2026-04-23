import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

class HeadPosePredictor(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=2):
        super(HeadPosePredictor, self).__init__()
        
        # 1. Audio Encoder (Wav2Vec2)
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
            
        # 2. Layer temporale
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # 3. Output: 3 valori (Pitch, Yaw, Roll)
        self.fc = nn.Linear(hidden_dim * 2, 3)

    def forward(self, audio_input, target_seq_len=None):
        # 1. Estrai le feature dall'audio (escono a 50 FPS)
        features = self.audio_encoder(audio_input).last_hidden_state # shape: (Batch, Seq_Audio, 768)
        
        # 2. LA MAGIA DI FEDERICO: Interpolazione Lineare
        # Se sappiamo quanti frame ha il video (target_seq_len), "stiriamo" l'audio per renderlo identico
        if target_seq_len is not None and features.size(1) != target_seq_len:
            # F.interpolate lavora sulla 3a dimensione, quindi dobbiamo scambiare le dimensioni
            features = features.transpose(1, 2) # (Batch, 768, Seq_Audio)
            # Interpoliamo alla lunghezza esatta del video
            features = F.interpolate(features, size=target_seq_len, mode='linear', align_corners=True)
            # Rimettiamo a posto le dimensioni
            features = features.transpose(1, 2) # (Batch, target_seq_len, 768)

        # 3. Passiamo i dati allineati alla LSTM
        lstm_out, _ = self.lstm(features) 
        
        # 4. Sputiamo fuori i 3 assi
        pose_pred = self.fc(lstm_out) 
        
        return pose_pred