
import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM Autoencoder for 1D risk signals (or small feature vectors).
    input_size: dimensionality of each timestep's vector (default 1 for scalar risk).
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        # x: (B, T, D)
        _, (h_n, _) = self.encoder(x)
        # Repeat hidden state T times as input to decoder
        B, T, D = x.size()
        dec_in = h_n[-1].unsqueeze(1).repeat(1, T, 1)  # (B, T, H)
        out, _ = self.decoder(dec_in)
        return out

    def reconstruction_loss(self, x, y):
        return torch.mean((x - y) ** 2)