
from typing import List, Dict, Any
import random
import numpy as np
from ..drift.ae_model import LSTMAutoencoder
from .client import LocalClient
from .privacy import PrivacyLedger

def create_windows(risk_series: np.ndarray, window: int = 10) -> np.ndarray:
    risk_series = risk_series.astype(np.float32)
    if len(risk_series) < window:
        return np.zeros((0, window, 1), dtype=np.float32)
    X = []
    for i in range(len(risk_series) - window + 1):
        X.append(risk_series[i:i+window])
    X = np.stack(X)[:, :, None]
    return X

class FederatedServer:
    """
    Simulated FedAvg for an LSTM-AE model across local clients.
    """
    def __init__(self, hidden_size: int = 32, device: str = "cpu", noise_multiplier: float = 0.5, max_grad_norm: float = 1.0):
        self.device = device
        self.global_model = LSTMAutoencoder(input_size=1, hidden_size=hidden_size).to(device)
        self.ledger = PrivacyLedger(noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm)

    def aggregate(self, client_updates: List[Dict[str, Any]]):
        # Simple average of state_dicts
        state_dicts = [u["state_dict"] for u in client_updates if "state_dict" in u]
        if not state_dicts:
            return
        avg = {}
        for k in state_dicts[0].keys():
            avg[k] = sum(sd[k] for sd in state_dicts) / len(state_dicts)
        self.global_model.load_state_dict(avg)

    def train_round(
        self,
        clients: List[LocalClient],
        sample_frac: float = 1.0,
        lr: float = 1e-3
    ):
        m = max(1, int(len(clients) * sample_frac))
        selected = random.sample(clients, m)
        updates = []
        for c in selected:
            upd = c.train(self.global_model, lr=lr, max_grad_norm=self.ledger.max_grad_norm, noise_multiplier=self.ledger.noise_multiplier)
            updates.append(upd)
        self.aggregate(updates)
        self.ledger.log_round()

    def summary(self) -> Dict[str, Any]:
        s = self.ledger.summary()
        s["global_params"] = sum(p.numel() for p in self.global_model.parameters())
        return s