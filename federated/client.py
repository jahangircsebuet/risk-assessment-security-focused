
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from ..drift.ae_model import LSTMAutoencoder
from .privacy import clip_gradients, add_gaussian_noise_to_grads

class LocalClient:
    """
    Simulated FL client: trains an LSTM-AE on local windows of risk scores.
    """
    def __init__(self, client_id: str, data_windows: np.ndarray, epochs: int = 1, batch_size: int = 32, device: str = "cpu"):
        self.client_id = client_id
        self.device = device
        X = torch.tensor(data_windows, dtype=torch.float32, device=device)  # (N, W, 1)
        self.loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)
        self.epochs = epochs

    def train(self, global_model: LSTMAutoencoder, lr: float = 1e-3, max_grad_norm: float = 1.0, noise_multiplier: float = 0.0) -> Dict[str, Any]:
        model = LSTMAutoencoder(input_size=1, hidden_size=global_model.hidden_size, num_layers=1).to(self.device)
        model.load_state_dict(global_model.state_dict())
        model.train()

        opt = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(self.epochs):
            for (batch,) in self.loader:
                opt.zero_grad()
                recon = model(batch)
                loss = model.reconstruction_loss(batch, recon)
                loss.backward()
                clip_gradients(model, max_grad_norm)
                add_gaussian_noise_to_grads(model, noise_multiplier=noise_multiplier, max_norm=max_grad_norm)
                opt.step()

        # Return updated weights
        return {"client_id": self.client_id, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}