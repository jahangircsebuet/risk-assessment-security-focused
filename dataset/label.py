from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import pandas as pd


# ----------------------------
# Mappings / taxonomy helpers
# ----------------------------
PII_TYPES = {"email", "phone", "address", "id", "coordinate"}
SENS_TYPES = {"health", "finance", "employment", "legal_immigration", "relationship"}

VIS_SCORE = {"private": 0.0, "friends": 0.5, "public": 1.0}
PII_SEVERITY = {"email": 0.6, "phone": 0.8, "address": 0.9, "id": 1.0, "coordinate": 1.0}


def _norm_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def _norm_str(x: Any) -> str:
    return "" if x is None else str(x).strip().lower()


# ----------------------------
# Channel construction (E) + V scalar
# ----------------------------
def build_channels_from_row(row: pd.Series) -> Dict[str, float]:
    """
    Builds the 5 exposure channels used in z_E:
      spii, ssens, sloc, svis, slink in [0,1]
    Aligned with Sec. 4.2.1 + Table 1. :contentReference[oaicite:2]{index=2}
    """
    contains_pii = _norm_bool(row.get("contains_pii"))
    pii_type = _norm_str(row.get("pii_type"))
    sens_type = _norm_str(row.get("sensitive_disclosure_type"))
    loc = _norm_bool(row.get("location_exposure"))
    vis = _norm_str(row.get("visibility"))
    ext = _norm_bool(row.get("external_surface"))

    # spii: explicit PII exposure
    spii = PII_SEVERITY.get(pii_type, 0.0) if contains_pii else 0.0

    # ssens: sensitive self-disclosure (health/finance/employment/legal_immigration/relationship)
    ssens = 0.0
    if sens_type in SENS_TYPES:
        # you can tune these values; keep simple + monotone
        ssens = {
            "health": 0.8,
            "finance": 0.6,
            "employment": 0.55,
            "legal_immigration": 0.75,
            "relationship": 0.45,
        }.get(sens_type, 0.0)

    # sloc: location exposure (real-time presence / identifiable place)
    sloc = 0.6 if loc else 0.0

    # svis: visibility amplifier (public > friends > private)
    svis = VIS_SCORE.get(vis, 0.25)

    # slink: external surface (URLs, contact requests, suspicious domains)
    slink = 0.6 if ext else 0.0

    return {"spii": spii, "ssens": ssens, "sloc": sloc, "svis": svis, "slink": slink}


def build_V_from_row(row: pd.Series) -> float:
    """
    Builds V(pt) ∈ [0,1] from contextual cue list.
    In your taxonomy, these cues are bounded modifiers (Sec. 4.2.2; Table 1). :contentReference[oaicite:3]{index=3}
    Here: simple fraction of cues present.
    """
    cues = row.get("contextual_cues_present", [])
    if cues is None:
        return 0.0
    if isinstance(cues, str):
        # if it came in as a stringified list, keep conservative
        cues = [c.strip() for c in cues.strip("[]").replace('"', "").split(",") if c.strip()]
    if not isinstance(cues, list):
        return 0.0

    allowed = {
        "distress_cues",
        "urgency_panic_cues",
        "crisis_impulsivity_cues",
        "coercion_solicitation_cues",
        "conflict_harassment_cues",
        "oversharing_regret_boundary_cues",
    }
    present = [c for c in cues if _norm_str(c) in allowed]
    return min(1.0, len(present) / 6.0)


# ----------------------------
# Labeling policy (silver labels from metadata)
# ----------------------------
@dataclass(frozen=True)
class LabelPolicy:
    """
    Rule-based label policy for generating y for calibration.
    - y_binary: 1 if "unacceptable exposure" (your y=1 definition in Sec. 4.2.1) :contentReference[oaicite:4]{index=4}
    - severity_0_2: 0 benign, 1 moderate, 2 severe
    """
    # threshold on exposure-only "severity proxy"
    moderate_threshold: float = 0.75
    severe_threshold: float = 1.35
    # visibility boosts (public posts are more actionable)
    public_boost: float = 0.20
    friends_boost: float = 0.10


def label_row(row: pd.Series, policy: LabelPolicy) -> Tuple[int, int, str, Dict[str, float], float]:
    ch = build_channels_from_row(row)
    V = build_V_from_row(row)

    vis = _norm_str(row.get("visibility"))
    pii_type = _norm_str(row.get("pii_type"))
    contains_pii = _norm_bool(row.get("contains_pii"))
    ext = _norm_bool(row.get("external_surface"))
    loc = _norm_bool(row.get("location_exposure"))
    sens = _norm_str(row.get("sensitive_disclosure_type"))

    # Exposure severity proxy (E-only): emphasize PII, then location/external, then sensitive
    # (This is NOT your calibrated RE; it’s just to auto-label y.)
    severity_proxy = (
        1.4 * ch["spii"] +
        0.8 * ch["sloc"] +
        0.7 * ch["slink"] +
        0.9 * ch["ssens"] +
        0.4 * ch["svis"]
    )

    # Visibility actionability boost for labeling
    if vis == "public":
        severity_proxy += policy.public_boost
    elif vis == "friends":
        severity_proxy += policy.friends_boost

    reasons: List[str] = []

    # Strong, direct triggers for y=1
    if contains_pii and pii_type in PII_TYPES:
        reasons.append(f"explicit_pii:{pii_type}")
    if sens in SENS_TYPES and sens != "none":
        reasons.append(f"sensitive_disclosure:{sens}")
    if loc:
        reasons.append("location_exposure")
    if ext:
        reasons.append("external_surface")
    if vis:
        reasons.append(f"visibility:{vis}")

    # Severity bucket
    if severity_proxy >= policy.severe_threshold:
        sev = 2
    elif severity_proxy >= policy.moderate_threshold:
        sev = 1
    else:
        sev = 0

    # Binary label y (for logistic calibration): unacceptable exposure if sev >= 1
    y = 1 if sev >= 1 else 0

    # For category C and E, enforce consistency rules as in your generator prompt :contentReference[oaicite:5]{index=5}
    cat = _norm_str(row.get("category"))
    if cat == "e":
        y, sev = 0, 0
        reasons = ["benign_control"]
    elif cat == "c":
        # contextual-only should not be labeled high-risk (exposure-first semantics)
        # unless metadata contradicts and contains_pii is actually true (shouldn’t happen per rules)
        if not contains_pii and sens == "none" and not loc and not ext:
            y, sev = 0, 0
            reasons = ["context_only_no_exposure"]

    reason_str = ";".join(reasons) if reasons else "low_exposure"
    return y, sev, reason_str, ch, V


def label_dataframe(df: pd.DataFrame, policy: LabelPolicy | None = None) -> pd.DataFrame:
    policy = policy or LabelPolicy()

    ys: List[int] = []
    sevs: List[int] = []
    rs: List[str] = []
    Vs: List[float] = []
    ch_cols = {k: [] for k in ("spii", "ssens", "sloc", "svis", "slink")}

    for _, row in df.iterrows():
        y, sev, reason, ch, V = label_row(row, policy)
        ys.append(y)
        sevs.append(sev)
        rs.append(reason)
        Vs.append(V)
        for k in ch_cols:
            ch_cols[k].append(ch[k])

    out = df.copy()
    out["y_binary"] = ys
    out["severity_0_2"] = sevs
    out["label_reason"] = rs
    out["V_context"] = Vs

    # add exposure channels
    for k, vals in ch_cols.items():
        out[k] = vals

    return out


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # df = pd.read_json("your_posts.jsonl", lines=True)
    # or df = pd.DataFrame(list_of_dicts)

    df_labeled = label_dataframe(df)
    df_labeled.to_csv("labeled_posts.csv", index=False)
    print(df_labeled[["category", "y_binary", "severity_0_2", "label_reason", "spii", "ssens", "sloc", "svis", "slink", "V_context"]].head())