
from typing import List, Dict

def explanation_text(user_id: str, t_str: str, top_feats: List[Dict]) -> str:
    points = ", ".join([f"{f['feature']} (↑{f['delta']:.2f})" for f in top_feats])
    return (
        f"User {user_id}: Around {t_str}, your privacy exposure increased. "
        f"Top changes: {points}. Consider adjusting audience/geotags or removing sensitive details."
    )