#!/usr/bin/env python3
import argparse, pandas as pd
from risk_core.scoring import aggregate_daily
from drift.detectors import detect_drifts_series

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="Input scored.parquet from Step 1")
    ap.add_argument("--out", required=True, help="Output drifts.parquet")
    ap.add_argument("--z_window", type=int, default=14)
    ap.add_argument("--ae_window", type=int, default=10)
    ap.add_argument("--ae_epochs", type=int, default=10)
    args = ap.parse_args()

    scored = pd.read_parquet(args.scored)
    daily = aggregate_daily(scored, time_col="timestamp", user_col="user_id")

    outs = []
    for uid, grp in daily.groupby("user_id"):
        ser = grp.set_index("t")["risk_score"].sort_index()
        det = detect_drifts_series(
            ser, method="hybrid",
            z_window=args.z_window,
            ae_window=args.ae_window,
            ae_epochs=args.ae_epochs
        )
        det["user_id"] = uid
        outs.append(det)

    out = pd.concat(outs, ignore_index=True)
    out.to_parquet(args.out, index=False)
    print(f"Wrote {args.out} with {len(out)} rows.")

if __name__ == "__main__":
    # Step 2: Detect drift per user
    main()
