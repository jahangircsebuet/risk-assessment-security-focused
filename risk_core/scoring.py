
from typing import Dict, Any, Optional
import pandas as pd
import yaml

DEFAULT_WEIGHTS = {
    # content features
    "n_emails": 3.0,
    "n_phones": 3.0,
    "n_urls": 0.5,
    "health_hits": 2.0,
    "finance_hits": 2.5,
    "address_hits": 1.5,
    "ner_person": 0.8,
    "ner_gpe": 0.8,
    "ner_org": 0.6,
    # context multipliers
    "audience_public": 1.5,
    "audience_friends": 0.6,
    "audience_private": 0.2,
    "geotag_on": 1.2,
    "has_media": 0.4,
}

class RiskScorer:
    def __init__(self, weight_config: Optional[str] = None):
        if weight_config:
            with open(weight_config, "r") as f:
                self.weights = yaml.safe_load(f) or DEFAULT_WEIGHTS
        else:
            self.weights = DEFAULT_WEIGHTS.copy()

    def score_row(self, row: Dict[str, Any]) -> float:
        score = 0.0
        for k, w in self.weights.items():
            v = float(row.get(k, 0.0))
            score += w * v
        return float(score)

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["risk_score"] = df.apply(lambda r: self.score_row(r.to_dict()), axis=1)
        return df

def aggregate_daily(df: pd.DataFrame, time_col: str = "timestamp", user_col: str = "user_id") -> pd.DataFrame:
    """
    Optional daily aggregation of post-level risk to a time series per user.
    Assumes `timestamp` is pandas datetime64.
    """
    g = df.copy()
    g["day"] = pd.to_datetime(g[time_col]).dt.floor("D")
    agg = (
        g.groupby([user_col, "day"])
         .agg(risk_score=("risk_score", "sum"))
         .reset_index()
         .rename(columns={"day": "t"})
    )
    return agg