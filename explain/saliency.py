
from typing import List, Dict, Any
import pandas as pd

def top_feature_deviations(
    df_window: pd.DataFrame,
    df_baseline: pd.DataFrame,
    feature_cols: List[str],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Compute top-k features that deviated in the window relative to baseline mean.
    """
    win_mean = df_window[feature_cols].mean()
    base_mean = df_baseline[feature_cols].mean() + 1e-8
    delta = (win_mean - base_mean).sort_values(ascending=False)
    out = []
    for feat, val in delta.iloc[:top_k].items():
        out.append({"feature": feat, "delta": float(val)})
    return out