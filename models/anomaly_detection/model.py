import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """
    Lightweight LSTM Autoencoder for edge-based anomaly detection.
    Reconstructs time-series sequence window; high reconstruction loss indicates anomalies.
    """
    def __init__(self, seq_len=30, input_dim=23, hidden_dim=16, latent_dim=8):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_linear = nn.Linear(hidden_dim, latent_dim)
        self.decoder_linear = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (hn, _) = self.encoder_lstm(x)
        latent = self.encoder_linear(hn.squeeze(0))
        decoded_hidden = self.decoder_linear(latent)
        decoded_hidden_repeated = decoded_hidden.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        reconstructed, _ = self.decoder_lstm(decoded_hidden_repeated)
        return reconstructed
