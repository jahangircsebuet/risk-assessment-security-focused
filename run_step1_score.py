#!/usr/bin/env python3
import argparse, glob
import pandas as pd
from risk_core.extractors import build_feature_vector
from risk_core.scoring import RiskScorer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_glob", required=True, help="Parquet glob (collects all the files matching that pattern into a list, then loads them.), e.g., data/parquet/*.parquet")
    ap.add_argument("--out", required=True, help="Output scored Parquet")
    args = ap.parse_args()

    print("args: ", args)

    files = sorted(glob.glob(args.in_glob))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    feats = []
    for _, r in df.iterrows():
        fv = build_feature_vector(
            r.get("text",""),
            audience=r.get("audience","public"),
            geotag=bool(r.get("geotag", False)),
            has_media=bool(r.get("has_media", False)),
        )
        fv.update({"user_id": r["user_id"], "timestamp": r["timestamp"], "text": r["text"]})
        feats.append(fv)

    feat_df = pd.DataFrame(feats)
    scorer = RiskScorer()
    scored = scorer.score_dataframe(feat_df)
    scored.to_parquet(args.out, index=False)
    print(f"Wrote {args.out} with {len(scored)} rows.")

if __name__ == "__main__":
    main()
