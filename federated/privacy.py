
from typing import Dict, Any
import torch

def clip_gradients(model, max_norm: float):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def add_gaussian_noise_to_grads(model, noise_multiplier: float, max_norm: float):
    if noise_multiplier <= 0:
        return
    for p in model.parameters():
        if p.grad is None:
            continue
        noise = torch.normal(
            mean=0.0, std=noise_multiplier * max_norm, size=p.grad.shape, device=p.grad.device
        )
        p.grad.add_(noise)

class PrivacyLedger:
    """
    Minimal tracker for DP hyperparameters used across rounds.
    This is NOT a rigorous accountant. For formal guarantees, integrate Opacus/TF-Privacy.
    """
    def __init__(self, noise_multiplier: float, max_grad_norm: float):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.rounds = 0

    def log_round(self):
        self.rounds += 1

    def summary(self) -> Dict[str, Any]:
        return {
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "rounds": self.rounds,
            "note": "This is a placeholder ledger; use a formal DP accountant for epsilon."
        }