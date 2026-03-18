# from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ----------------------------
# Config
# ----------------------------
EXPOSURE_KEYS = ("spii", "ssens", "sloc", "svis", "slink")


# ----------------------------
# Validation
# ----------------------------
@dataclass(frozen=True)
class WeightValidationReport:
    ok: bool
    errors: List[str]
    warnings: List[str]


def validate_weights(
    weights_E: Mapping[str, float],
    *,
    require_all_keys: bool = True,
    non_negative: bool = True,
    max_abs_weight: Optional[float] = 50.0,
    allowed_keys: Sequence[str] = EXPOSURE_KEYS,
) -> WeightValidationReport:
    errors: List[str] = []
    warnings: List[str] = []

    allowed = set(allowed_keys)
    present = set(weights_E.keys())

    missing = [k for k in allowed_keys if k not in present]
    extra = [k for k in present if k not in allowed]

    if require_all_keys and missing:
        errors.append(f"Missing required exposure weight keys: {missing}")
    elif missing:
        warnings.append(f"Missing exposure weight keys (treated as 0): {missing}")

    if extra:
        warnings.append(f"Extra weight keys not used by scorer (ignored): {extra}")

    for k in allowed_keys:
        if k not in weights_E:
            continue
        try:
            v = float(weights_E[k])
        except Exception:
            errors.append(f"Weight '{k}' is not a valid number: {weights_E[k]!r}")
            continue

        if non_negative and v < 0.0:
            errors.append(f"Weight '{k}' is negative ({v}); violates monotonicity (w_i >= 0).")

        if max_abs_weight is not None and abs(v) > max_abs_weight:
            warnings.append(f"Weight '{k}' magnitude is large ({v}); check scaling / overfit risk.")

        if math.isnan(v) or math.isinf(v):
            errors.append(f"Weight '{k}' is NaN/Inf: {v}")

    return WeightValidationReport(ok=(len(errors) == 0), errors=errors, warnings=warnings)


# ----------------------------
# IO: Load JSONL
# ----------------------------
def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
    return rows


# ----------------------------
# Feature extraction (CBPRS-C exposure channels)
# ----------------------------
def svis_from_visibility(vis: str) -> float:
    # As in your draft: public=1, friends=0.5, private=0
    v = (vis or "").lower()
    if v == "public":
        return 1.0
    if v == "friends":
        return 0.5
    return 0.0


def spii_from_record(rec: dict) -> float:
    # Binary PII signal; you can refine by pii_type severity if you want.
    return 1.0 if rec["post"].get("contains_pii", False) else 0.0


def ssens_from_record(rec: dict) -> float:
    # Binary sensitive disclosure signal
    sdt = rec["post"].get("sensitive_disclosure_type", "none") or "none"
    return 1.0 if sdt != "none" else 0.0


def sloc_from_record(rec: dict) -> float:
    # Binary location exposure signal
    return 1.0 if rec["post"].get("location_exposure", False) else 0.0


def slink_from_record(rec: dict) -> float:
    # External surface (URLs / outbound links)
    return 1.0 if rec["post"].get("external_surface", False) else 0.0


def extract_exposure_row(rec: dict) -> Dict[str, float]:
    post = rec.get("post", {})
    # Consistency: if external_surface=false, urls should be empty; still safe.
    return {
        "spii": spii_from_record(rec),
        "ssens": ssens_from_record(rec),
        "sloc": sloc_from_record(rec),
        "svis": svis_from_visibility(post.get("visibility", "private")),
        "slink": slink_from_record(rec),
    }


def extract_label(rec: dict) -> int:
    y = rec.get("labels", {}).get("risk_label", None)
    if y not in (0, 1):
        raise ValueError(f"Missing/invalid labels.risk_label (expected 0/1): {y!r}")
    return int(y)


def make_Xy(records: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    rows = [extract_exposure_row(r) for r in records]
    X = np.zeros((len(rows), len(EXPOSURE_KEYS)), dtype=float)
    for i, row in enumerate(rows):
        for j, k in enumerate(EXPOSURE_KEYS):
            X[i, j] = float(row.get(k, 0.0))
    y = np.array([extract_label(r) for r in records], dtype=int)
    return X, y


# ----------------------------
# Logistic calibration fit (recommended for binary risk_label)
# ----------------------------
@dataclass(frozen=True)
class LogisticFitResult:
    weights_E: Dict[str, float]  # w· for spii, ssens, sloc, svis, slink
    bE: float                    # operating-point term in form (w^T x - bE)
    train_metrics: Dict[str, float]
    val_metrics: Optional[Dict[str, float]]


def fit_logistic_weights(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str] = EXPOSURE_KEYS,
    l2_C: float = 1.0,
    enforce_nonneg: bool = True,
    val_split: float = 0.2,
    random_state: int = 42,
) -> LogisticFitResult:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    if X.ndim != 2 or X.shape[1] != len(feature_names):
        raise ValueError(f"X must be shape (n_samples, {len(feature_names)}), got {X.shape}")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("y must be binary {0,1}")

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=val_split, random_state=random_state, stratify=y
    )

    lr = LogisticRegression(
        penalty="l2",
        C=float(l2_C),
        solver="lbfgs",
        max_iter=5000,
        fit_intercept=True,
    )
    lr.fit(Xtr, ytr)

    coef = lr.coef_.reshape(-1).copy()
    intercept = float(lr.intercept_.reshape(-1)[0])

    if enforce_nonneg:
        # clip negative weights to 0 to preserve monotonicity
        coef = np.maximum(coef, 0.0)

        # refit intercept only (1D) with fixed non-negative coef
        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        b = intercept
        for _ in range(60):
            z = Xtr @ coef + b
            p = sigmoid(z)
            g = float(np.sum(p - ytr))
            h = float(np.sum(p * (1 - p)) + 1e-9)
            step = g / h
            b -= step
            if abs(step) < 1e-7:
                break
        intercept = float(b)

    def _metrics(Xm, ym):
        probs = 1.0 / (1.0 + np.exp(-(Xm @ coef + intercept)))
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(ym, probs) if len(np.unique(ym)) > 1 else float("nan")
        prec, rec, f1, _ = precision_recall_fscore_support(
            ym, preds, average="binary", zero_division=0
        )
        acc = accuracy_score(ym, preds)
        return {
            "auc": float(auc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "acc": float(acc),
        }

    train_metrics = _metrics(Xtr, ytr)
    val_metrics = _metrics(Xva, yva) if len(yva) > 0 else None

    weights_E = {feature_names[i]: float(coef[i]) for i in range(len(feature_names))}
    bE = -intercept  # keeps your form: w^T x - bE

    rep = validate_weights(weights_E, require_all_keys=True, non_negative=enforce_nonneg)
    if not rep.ok:
        raise RuntimeError("Fitted weights violate constraints:\n" + "\n".join(rep.errors))

    return LogisticFitResult(
        weights_E=weights_E,
        bE=float(bE),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )


# ----------------------------
# Save weights/bias to JSON
# ----------------------------
def save_weights_json(out_path: str, fit: LogisticFitResult) -> None:
    payload = {
        "schema_version": "cbprs-c.weights.v1",
        "feature_names": list(EXPOSURE_KEYS),
        "weights_E": fit.weights_E,
        "bE": fit.bE,
        "train_metrics": fit.train_metrics,
        "val_metrics": fit.val_metrics,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ----------------------------
# Main
# ----------------------------
def main(
    dataset_jsonl_path: str,
    out_weights_json_path: str = "cbprs_c_weights.json",
) -> None:
    records = load_jsonl(dataset_jsonl_path)
    X, y = make_Xy(records)

    fit = fit_logistic_weights(
        X, y,
        l2_C=1.0,
        enforce_nonneg=True,
        val_split=0.2,
        random_state=42,
    )

    save_weights_json(out_weights_json_path, fit)

    print("Saved:", out_weights_json_path)
    print("Weights:", fit.weights_E)
    print("bE:", fit.bE)
    print("Train metrics:", fit.train_metrics)
    print("Val metrics:", fit.val_metrics)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to cbprs-c.v2.1 JSONL dataset")
    ap.add_argument("--out", default="cbprs_c_weights.json", help="Output JSON path for weights/bias")
    args = ap.parse_args()

    main(args.data, args.out)