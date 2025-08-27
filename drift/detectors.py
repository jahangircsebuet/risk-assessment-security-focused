
from typing import List
import numpy as np
import pandas as pd
import torch
from drift.ae_model import LSTMAutoencoder

def rolling_zscore(x: np.ndarray, window: int = 30) -> np.ndarray:
    z = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window)
        segment = x[start:i+1]
        mu = segment.mean()
        sigma = segment.std() + 1e-8
        z[i] = (x[i] - mu) / sigma
    return z

def cusum_detect(x: np.ndarray, k: float = 0.5, h: float = 8.0) -> List[int]:
    """
    Simple CUSUM detector for upward shifts.
    Returns indices where the statistic exceeds threshold h.
    """
    g = 0.0
    alarms = []
    for i, xi in enumerate(x):
        g = max(0.0, g + xi - k)
        if g > h:
            alarms.append(i)
            g = 0.0  # reset after alarm
    return alarms

def ae_anomaly_scores(x: np.ndarray, window: int = 10, epochs: int = 10, hidden: int = 32, device: str = "cpu") -> np.ndarray:
    """
    Train a tiny LSTM-AE on sliding windows and compute reconstruction error per center point.
    """
    x = x.astype(np.float32)
    T = len(x)
    if T < window:
        return np.zeros(T, dtype=np.float32)

    # Build dataset of windows
    X = []
    for i in range(T - window + 1):
        X.append(x[i:i+window])
    X = np.stack(X)[:, :, None]  # (N, W, 1)
    tensor = torch.tensor(X, device=device)

    model = LSTMAutoencoder(input_size=1, hidden_size=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        recon = model(tensor)
        loss = model.reconstruction_loss(tensor, recon)
        loss.backward()
        opt.step()

    # Inference: reconstruction error per window, center mapped back
    model.eval()
    with torch.no_grad():
        recon = model(tensor)
        err = torch.mean((tensor - recon) ** 2, dim=(1,2)).detach().cpu().numpy()

    scores = np.zeros(T, dtype=np.float32)
    half = window // 2
    for i, e in enumerate(err):
        center = i + half
        if center < T:
            scores[center] = max(scores[center], e)
    return scores

def detect_drifts_series(
    series: pd.Series,
    method: str = "hybrid",
    z_window: int = 30,
    cusum_k: float = 0.5,
    cusum_h: float = 8.0,
    ae_window: int = 10,
    ae_epochs: int = 10,
    ae_hidden: int = 32,
    device: str = "cpu",
    z_threshold: float = 3.0,
    ae_threshold: float = None
) -> pd.DataFrame:
    """
    Detect drift indices in a single user time series of risk scores.
    method: 'z', 'cusum', 'ae', or 'hybrid' (OR of z>thr, cusum alarm, ae>thr)
    """
    x = series.values.astype(float)
    out = pd.DataFrame({"t": series.index, "risk_score": x})

    # Rolling z-score
    z = rolling_zscore(x, window=z_window)
    out["zscore"] = z
    z_alarms = set(np.where(z >= z_threshold)[0].tolist())

    # CUSUM on z-scores (stabilized)
    cusum_alarms = set(cusum_detect(z, k=cusum_k, h=cusum_h))

    # AE scores
    ae_scores = ae_anomaly_scores(x, window=ae_window, epochs=ae_epochs, hidden=ae_hidden, device=device)
    out["ae_score"] = ae_scores
    if ae_threshold is None:
        # heuristic: 95th percentile of AE scores as threshold
        ae_threshold = float(np.percentile(ae_scores, 95)) if len(ae_scores) else 0.0
    ae_alarms = set(np.where(ae_scores >= ae_threshold)[0].tolist())

    if method == "z":
        alarms = z_alarms
    elif method == "cusum":
        alarms = cusum_alarms
    elif method == "ae":
        alarms = ae_alarms
    else:
        alarms = z_alarms | cusum_alarms | ae_alarms  # hybrid

    out["drift"] = [int(i in alarms) for i in range(len(out))]
    out["method"] = method
    out["thresholds"] = [ {"z": z_threshold, "cusum": {"k": cusum_k, "h": cusum_h}, "ae": ae_threshold} ] * len(out)
    return out