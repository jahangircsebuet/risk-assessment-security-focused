
# End-to-end smoke test for the 4 steps.
# It makes a tiny synthetic dataset with 2 users, computes features & risk (Step 1),
# detects drifts (Step 2), simulates a federated training round (Step 3),
# and prints explanations (Step 4).

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from risk_core.extractors import build_feature_vector
from risk_core.scoring import RiskScorer, aggregate_daily
from drift.detectors import detect_drifts_series
from federated.server import FederatedServer, create_windows
from federated.client import LocalClient
from explain.saliency import top_feature_deviations
from explain.templates import explanation_text
from explain.misinfo import misinfo_context

def synth_data():
    rows = []
    base = datetime(2025, 1, 1)
    # Two users; user_1 will spike later
    for u in ["user_1", "user_2"]:
        for d in range(60):
            ts = base + timedelta(days=d)
            # Synthetic behavior
            if u == "user_1" and d > 40:
                text = f"My salary is 90k, hospital visit at day {d}. https://badnews.example/news"
                audience = "public"
                geotag = True
                has_media = True
            else:
                text = "Lovely day with friends at the park."
                audience = "friends"
                geotag = False
                has_media = False
            feats = build_feature_vector(text, audience=audience, geotag=geotag, has_media=has_media)
            feats.update({"user_id": u, "timestamp": ts, "text": text})
            rows.append(feats)
    return pd.DataFrame(rows)

def step1_score(df_posts: pd.DataFrame) -> pd.DataFrame:
    scorer = RiskScorer()
    scored = scorer.score_dataframe(df_posts)
    return scored

def step2_detect(scored: pd.DataFrame) -> pd.DataFrame:
    # Aggregate daily risk per user (optional)
    daily = aggregate_daily(scored, time_col="timestamp", user_col="user_id")
    out = []
    for uid, grp in daily.groupby("user_id"):
        ser = grp.set_index("t")["risk_score"].sort_index()
        det = detect_drifts_series(ser, method="hybrid", z_window=14, ae_window=10, ae_epochs=15)
        det["user_id"] = uid
        out.append(det)
    return pd.concat(out, ignore_index=True)

def step3_federated(scored: pd.DataFrame) -> dict:
    # Build windows per user and simulate a single FL round
    server = FederatedServer(hidden_size=16, noise_multiplier=0.5, max_grad_norm=1.0)
    clients = []
    for uid, grp in scored.groupby("user_id"):
        # Build scalar risk series
        ser = grp.sort_values("timestamp")["risk_score"].values.astype(np.float32)
        X = create_windows(ser, window=10)
        clients.append(LocalClient(uid, X, epochs=1, batch_size=8, device="cpu"))
    server.train_round(clients, sample_frac=1.0, lr=1e-3)
    return server.summary()

def step4_explain(scored: pd.DataFrame, det_df: pd.DataFrame):
    feature_cols = [c for c in scored.columns if c not in {"user_id", "timestamp", "text", "risk_score"}]
    msgs = []
    for uid, grp in det_df.groupby("user_id"):
        user_posts = scored[scored["user_id"] == uid].sort_values("timestamp")
        for _, row in grp[grp["drift"] == 1].iterrows():
            # Window around drift time
            t = row["t"]
            win = user_posts[(user_posts["timestamp"] >= (t - pd.Timedelta(days=3))) &
                             (user_posts["timestamp"] <= (t + pd.Timedelta(days=3)))]
            base = user_posts[user_posts["timestamp"] < (t - pd.Timedelta(days=14))]
            if len(win) == 0 or len(base) == 0:
                continue
            top_feats = top_feature_deviations(win, base, feature_cols, top_k=3)
            msg = explanation_text(uid, t.strftime("%Y-%m-%d"), top_feats)
            ctx = misinfo_context(win["text"].tolist())
            if ctx["engaged"]:
                msg += f" (Context: topics={ctx['topics']}, bad_domains={ctx['bad_domain_hits']})"
            msgs.append(msg)
    return msgs

def main():
    print("Generating synthetic posts...")
    df_posts = synth_data()
    print("Step 1: Scoring...")
    scored = step1_score(df_posts)
    print(scored.head())

    print("Step 2: Drift detection...")
    det_df = step2_detect(scored)
    print(det_df.tail())

    print("Step 3: Federated simulation summary...")
    summary = step3_federated(scored)
    print(summary)

    print("Step 4: Explanations...")
    msgs = step4_explain(scored, det_df)
    for m in msgs[:5]:
        print("-", m)

if __name__ == "__main__":
    main()