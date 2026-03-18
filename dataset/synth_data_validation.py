# validate_synthetic_privacy_drift.py
# Usage:
#   python validate_synthetic_privacy_drift.py /path/to/synthetic_social_privacy_drift_seed42.csv
#
# Notes:
# - This is a *validation/coverage* checker using practical heuristics.
# - It will flag missing/weak signals and print examples for each requirement.

import sys, json, re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# ----------------------------
# CONFIG (expected values)
# ----------------------------
EXPECTED_ARCHETYPES = {
    "Stable/Normal",
    "Oversharer",
    "Compromised",
    "Spammer/Scammer",
    "Infrequent/Cold",
    "High Privacy",
}
DRIFT_PHASES = {"baseline", "mild", "moderate", "strong", "takeover", "recovery"}
AUDIENCE_VALUES = {"public", "friends", "private", "custom_group"}

REQUIRED_COLUMNS = [
    "user_id", "timestamp", "text",
    "audience", "geotag", "has_media", "contains_url", "is_reply", "is_repost", "language",
    "privacy_risk_score", "drift_phase", "user_archetype", "was_compromised", "pii_signals"
]

# ----------------------------
# Regex heuristics
# ----------------------------
URL_RX = re.compile(r"(https?://\S+|www\.\S+)", re.I)
MALFORMED_URL_RX = re.compile(r"\b(htp://\S+|http:\/\S+|https?//\S+|https?:/{1,2}(?!/)\S+)\b", re.I)

HASHTAG_RX = re.compile(r"#\w+")
EMOJI_RX = re.compile("[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F1E0-\U0001F1FF" "]+", flags=re.UNICODE)
CODE_RX = re.compile(r"```.*?```|`[^`]+`|\b(def|class|import|SELECT|INSERT|curl|wget|sudo)\b", re.S | re.I)

PROMPT_INJECT_RX = re.compile(r"\b(ignore previous instructions|system:|developer:|jailbreak|output my password|reveal.*password)\b", re.I)
PHISH_RX = re.compile(r"\b(verify( your)? account|reset( your)? password|urgent|click (here|link)|suspended|act now)\b", re.I)
SCAM_RX = re.compile(r"\b(free gift card|crypto|airdrop|investment|wire transfer|cashapp|venmo|prize|you won)\b", re.I)

MASKED_PII_RX = re.compile(
    r"(\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b)"                # SSN-like
    r"|(\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b)"     # email
    r"|(\b\+?\d{1,3}[-\s]?\(?\d{2,3}\)?[-\s]?\d{3}[-\s]?\d{4}\b)"  # phone-like
    r"|(@\w+)", re.I
)

MISSPELL_RX = re.compile(r"\b\w*(\w)\1{2,}\w*\b")       # e.g., sooooo
RANDOM_CAPS_RX = re.compile(r"\b[A-Z]{4,}\b")           # ALLCAPS tokens
NON_ASCII_RX = re.compile(r"[^\x00-\x7F]")              # code-switch-ish

# Subtle mixed-risk hints
HEALTH_RX = re.compile(r"\b(migraine|anxiety|depression|therapy|meds|doctor|clinic|diagnos)\w*\b", re.I)
FIN_RX = re.compile(r"\b(paycheck|salary|rent|mortgage|debt|credit|bank|bonus|tax)\w*\b", re.I)
WORK_RX = re.compile(r"\b(office|shift|manager|HR|client|meeting|standup|deadline|promo)\w*\b", re.I)
LOC_RX = re.compile(r"\b(downtown|airport|hotel|mall|station|near the|on my way|in [A-Z][a-z]+)\b")
NAME_HINT_RX = re.compile(r"\b(my (son|daughter|wife|husband|mom|dad) [A-Z][a-z]+|with [A-Z][a-z]+)\b")

# Contradiction heuristic (simple)
NEVER_RX = re.compile(r"\b(i\s+never|never\s+been|i\s+don'?t)\b", re.I)
DO_RX = re.compile(r"\b(i\s+do|i\s+did|i\s+have|i\s+am)\b", re.I)


# ----------------------------
# Helpers
# ----------------------------
def parse_list_cell(x):
    if pd.isna(x): return []
    if isinstance(x, list): return x
    s = str(x).strip()
    if not s or s.lower() == "nan": return []
    # json list?
    try:
        v = json.loads(s)
        if isinstance(v, list): return v
    except Exception:
        pass
    # split fallback
    s = s.strip("[]")
    return [p.strip().strip("'\"") for p in s.split(",") if p.strip()]

def pct(x): return round(100.0 * float(x), 2)

def ok_fail(ok): return "OK" if ok else "FAIL"

def sample_examples(df, mask, n=5):
    sub = df[mask]
    if sub.empty: return []
    cols = ["user_id", "timestamp", "drift_phase", "privacy_risk_score", "text"]
    return sub[cols].head(n).to_dict(orient="records")

def print_section(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))


# ----------------------------
# Main validation
# ----------------------------
def validate_synth_data(path):
    print("validation started...")
    df = pd.read_csv(path)

    # normalize timestamp col name
    if "timestamp" not in df.columns and "timestamp (YYYY-MM-DD HH:MM:SS)" in df.columns:
        df = df.rename(columns={"timestamp (YYYY-MM-DD HH:MM:SS)": "timestamp"})

    # schema
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    print_section("0) SCHEMA + SIZE")
    print(f"rows={len(df)}  (expected 500) -> {ok_fail(len(df)==500)}")
    print(f"missing_columns={missing} -> {ok_fail(len(missing)==0)}")
    if len(df) != 500:
        print("  WARNING: row count mismatch.")
    if missing:
        print("  WARNING: missing required columns; remaining checks may error.")

    # parse types
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["pii_signals_list"] = df["pii_signals"].apply(parse_list_cell)

    # ------------------------------------------------------------------
    print_section("1) USER ARCHETYPES")
    archetypes = set(df["user_archetype"].dropna().astype(str).unique())
    missing_arch = sorted(list(EXPECTED_ARCHETYPES - archetypes))
    print(f"unique_archetypes={sorted(list(archetypes))}")
    print(f"expected_subset -> {ok_fail(len(missing_arch)==0)}  missing_expected={missing_arch}")
    print("counts:", df["user_archetype"].value_counts(dropna=False).to_dict())
    n_users = df["user_id"].nunique()
    print(f"distinct users={n_users} (expected 8-12+) -> {ok_fail(n_users>=8)}")

    # ------------------------------------------------------------------
    print_section("2) TEMPORAL REALISM")
    min_ts, max_ts = df["timestamp"].min(), df["timestamp"].max()
    if pd.isna(min_ts) or pd.isna(max_ts):
        print("timestamp parse failed -> FAIL")
    else:
        span_days = (max_ts.date() - min_ts.date()).days + 1
        print(f"min_ts={min_ts}  max_ts={max_ts}  span_days={span_days} (expected 90-120) -> {ok_fail(90<=span_days<=120)}")

        all_days = pd.date_range(min_ts.normalize(), max_ts.normalize(), freq="D")
        missing_days = [d.date().isoformat() for d in all_days if d.date() not in set(df["date"])]
        print(f"missing_days_count={len(missing_days)} (want >0) -> {ok_fail(len(missing_days)>0)}  sample={missing_days[:10]}")

        daily_counts = df.groupby("date").size()
        burst = daily_counts[daily_counts >= 10]
        print(f"burst_days>=10 posts/day overall -> {ok_fail(len(burst)>0)}  n_burst_days={len(burst)}  top={burst.sort_values(ascending=False).head(5).to_dict()}")

        night_pct = (df["hour"].between(1,5)).mean()
        print(f"night_posts_pct(01:00-05:59)={pct(night_pct)}% (heuristic <=20%) -> {ok_fail(night_pct<=0.20)}")

        per_user_day = df.groupby(["user_id","date"]).size()
        viol = per_user_day[per_user_day > 5]
        print(f"per-user max posts/day <=5 -> {ok_fail(len(viol)==0)}  violations={len(viol)}  sample={viol.head(5).to_dict()}")

    # ------------------------------------------------------------------
    print_section("3) CONTENT DIVERSITY")
    dup = df["text"].duplicated().sum()
    print(f"no identical text duplicates -> {ok_fail(dup==0)}  dup_count={dup}")

    wc = df["text"].astype(str).apply(lambda t: len(t.split()))
    print(f"word_count range [{wc.min()}, {wc.max()}] (expected within 5-80) -> {ok_fail((wc.min()>=5) and (wc.max()<=80))}")

    # diversity markers
    has_emoji = df["text"].astype(str).apply(lambda t: bool(EMOJI_RX.search(t))).mean()
    has_hashtag = df["text"].astype(str).apply(lambda t: bool(HASHTAG_RX.search(t))).mean()
    has_code = df["text"].astype(str).apply(lambda t: bool(CODE_RX.search(t))).mean()
    has_url = df["text"].astype(str).apply(lambda t: bool(URL_RX.search(t))).mean()
    has_typos = df["text"].astype(str).apply(lambda t: bool(MISSPELL_RX.search(t))).mean()
    print("diversity marker rates (%):",
          {"emoji": pct(has_emoji), "hashtag": pct(has_hashtag), "code": pct(has_code), "url_in_text": pct(has_url), "typo_like": pct(has_typos)})
    print("marker presence ->",
          ok_fail(all(x > 0 for x in [has_emoji, has_hashtag, has_code, has_url])))

    # topic mixture (heuristic)
    health = df["text"].astype(str).apply(lambda t: bool(HEALTH_RX.search(t))).mean()
    fin = df["text"].astype(str).apply(lambda t: bool(FIN_RX.search(t))).mean()
    work = df["text"].astype(str).apply(lambda t: bool(WORK_RX.search(t))).mean()
    loc = df["text"].astype(str).apply(lambda t: bool(LOC_RX.search(t))).mean()
    print("topic-signal rates (%):",
          {"work": pct(work), "health": pct(health), "finance": pct(fin), "location_hint": pct(loc)})
    print("topic mixture ->", ok_fail(all(x > 0 for x in [work, health, fin, loc])))

    # ------------------------------------------------------------------
    print_section("4) DRIFT PATTERNS (heuristic checks)")
    # per-user weekly trend slope
    df2 = df.sort_values(["user_id","timestamp"]).copy()
    df2["risk"] = pd.to_numeric(df2["privacy_risk_score"], errors="coerce")

    def weekly_slope(user_df):
        user_df = user_df.dropna(subset=["timestamp","risk"]).set_index("timestamp").sort_index()
        wk = user_df["risk"].resample("7D").mean().dropna()
        if len(wk) < 3:
            return None
        x = np.arange(len(wk))
        slope = np.polyfit(x, wk.values, 1)[0]
        return slope, float(wk.min()), float(wk.max())

    gradual = []
    multiphase = []
    ambiguous = []

    for uid, g in df2.groupby("user_id"):
        ws = weekly_slope(g)
        if ws:
            slope, wkmin, wkmax = ws
            if slope > 0.02 and (wkmax - wkmin) > 0.25:
                gradual.append(uid)

        phases = g["drift_phase"].astype(str).tolist()
        transitions = sum(1 for i in range(1, len(phases)) if phases[i] != phases[i-1])
        has_takeover = (g["drift_phase"] == "takeover").any()
        has_recovery = (g["drift_phase"] == "recovery").any()
        if has_takeover and has_recovery and transitions >= 4:
            multiphase.append(uid)

        g_tmp = g.dropna(subset=["timestamp"]).copy()
        g_tmp["week"] = g_tmp["timestamp"].dt.to_period("W").astype(str)
        if (g_tmp.groupby("week")["drift_phase"].nunique() >= 3).any():
            ambiguous.append(uid)

    print(f"gradual drift users (risk rises slowly): {gradual} -> {ok_fail(len(gradual)>0)}")
    print(f"multi-phase drift users (takeover+recovery): {multiphase} -> {ok_fail(len(multiphase)>0)}")
    print(f"ambiguous drift users (>=3 phases in a week): {ambiguous} -> {ok_fail(len(ambiguous)>0)}")

    # ------------------------------------------------------------------
    print_section("5) MIXED-RISK SCENARIOS")
    pii_pct = df["pii_signals_list"].apply(len).gt(0).mean()
    name_pct = df["text"].astype(str).apply(lambda t: bool(NAME_HINT_RX.search(t))).mean()
    print("mixed-risk hint rates (%):",
          {"pii_signals_nonempty": pct(pii_pct), "health_hint": pct(health), "finance_hint": pct(fin),
           "location_hint": pct(loc), "name_hint": pct(name_pct)})
    print("mixed-risk presence ->", ok_fail(all(x > 0 for x in [pii_pct, health, fin, loc, name_pct])))

    risk_windows = df[df["drift_phase"].isin(["strong","takeover"])]
    benign_inside = (risk_windows["privacy_risk_score"] < 0.3).mean() if len(risk_windows) else 0.0
    print(f"benign posts inside strong/takeover windows (risk<0.3): {pct(benign_inside)}% -> {ok_fail(len(risk_windows)>0 and benign_inside>0)}")

    # ------------------------------------------------------------------
    print_section("6) ADVERSARIAL MANIPULATIONS")
    inj = df["text"].astype(str).apply(lambda t: bool(PROMPT_INJECT_RX.search(t))).mean()
    phish = df["text"].astype(str).apply(lambda t: bool(PHISH_RX.search(t))).mean()
    scam = df["text"].astype(str).apply(lambda t: bool(SCAM_RX.search(t))).mean()
    mal = df["text"].astype(str).apply(lambda t: bool(MALFORMED_URL_RX.search(t))).mean()
    masked = df["text"].astype(str).apply(lambda t: bool(MASKED_PII_RX.search(t))).mean()

    print("adversarial rates (%):",
          {"prompt_injection": pct(inj), "phishing": pct(phish), "scam": pct(scam), "malformed_url": pct(mal), "masked_pii": pct(masked)})
    print("prompt injection present ->", ok_fail(inj > 0))
    print("phishing present ->", ok_fail(phish > 0))
    print("scam present ->", ok_fail(scam > 0))
    print("malformed urls present ->", ok_fail(mal > 0))

    print("example prompt-injection posts:", sample_examples(df, df["text"].astype(str).str.contains(PROMPT_INJECT_RX), 3))
    print("example phishing posts:", sample_examples(df, df["text"].astype(str).str.contains(PHISH_RX), 3))
    print("example scam posts:", sample_examples(df, df["text"].astype(str).str.contains(SCAM_RX), 3))
    print("example malformed url posts:", sample_examples(df, df["text"].astype(str).str.contains(MALFORMED_URL_RX), 3))

    # ------------------------------------------------------------------
    print_section("7) MULTI-MODAL FLAGS + METADATA")
    aud_ok = df["audience"].isin(AUDIENCE_VALUES).mean()
    print(f"audience valid pct={pct(aud_ok)}% -> {ok_fail(aud_ok==1.0)}  counts={df['audience'].value_counts(dropna=False).to_dict()}")

    bool_cols = ["geotag","has_media","contains_url","is_reply","is_repost","was_compromised"]
    for c in bool_cols:
        ok = df[c].dropna().isin([True, False, 0, 1, "True", "False", "true", "false"]).mean()
        print(f"{c} boolean-like pct={pct(ok)}% -> {ok_fail(ok>=0.99)}")

    # contains_url consistency (heuristic)
    contains_url_text = df["text"].astype(str).apply(lambda t: bool(URL_RX.search(t)))
    contains_url_flag = df["contains_url"].astype(str).str.lower().isin(["true","1"])
    agree = (contains_url_text == contains_url_flag).mean()
    print(f"contains_url flag agreement with text-url heuristic={pct(agree)}% (target >=80%) -> {ok_fail(agree>=0.80)}")

    # ------------------------------------------------------------------
    print_section("8) NOISE / RANDOMNESS SIGNALS")
    caps = df["text"].astype(str).apply(lambda t: bool(RANDOM_CAPS_RX.search(t))).mean()
    codeswitch = ((df["language"].astype(str) != "en") | df["text"].astype(str).apply(lambda t: bool(NON_ASCII_RX.search(t)))).mean()
    print("noise rates (%):",
          {"random_caps": pct(caps), "emoji": pct(has_emoji), "misspell_like": pct(has_typos), "code_switch": pct(codeswitch), "malformed_url": pct(mal)})

    # contradictions (simple): "I never..." followed by "I did/I have/I am..." within 14 days
    contrad = []
    df3 = df2.sort_values(["user_id","timestamp"])
    for uid, g in df3.groupby("user_id"):
        t = g["text"].astype(str).tolist()
        ts = g["timestamp"].tolist()
        for i in range(len(t)-1):
            if NEVER_RX.search(t[i]) and DO_RX.search(t[i+1]):
                if (ts[i+1] - ts[i]) <= pd.Timedelta(days=14):
                    contrad.append((uid, str(ts[i])[:19], t[i][:80], str(ts[i+1])[:19], t[i+1][:80]))
                    break
    print(f"contradiction-heuristic found -> {ok_fail(len(contrad)>0)}  count={len(contrad)} sample={contrad[:3]}")

    # ------------------------------------------------------------------
    print_section("9) EXTRA VALIDATIONS I RECOMMEND")
    # drift_phase allowed values
    bad_phases = sorted(list(set(df["drift_phase"].dropna().astype(str).unique()) - DRIFT_PHASES))
    print(f"drift_phase allowed values -> {ok_fail(len(bad_phases)==0)}  bad={bad_phases}")

    # risk bounds
    oob = df[(df["privacy_risk_score"] < 0) | (df["privacy_risk_score"] > 1)]
    print(f"privacy_risk_score in [0,1] -> {ok_fail(len(oob)==0)}  oob_rows={len(oob)}")

    # takeover should correlate with was_compromised (soft)
    takeover = df[df["drift_phase"] == "takeover"]
    if len(takeover):
        takeover_comp = takeover["was_compromised"].astype(str).str.lower().isin(["true","1"]).mean()
        print(f"takeover rows={len(takeover)}; takeover marked compromised pct={pct(takeover_comp)}% (suggest >=70%) -> {ok_fail(takeover_comp>=0.70)}")
    else:
        print("no takeover rows found; if you expected takeover behavior, regenerate or adjust generator.")

    # distribution sanity: at least one post per drift_phase?
    phase_counts = df["drift_phase"].value_counts(dropna=False).to_dict()
    print("drift_phase counts:", phase_counts)

    # per-user style diversity: average word count per user should vary (basic sanity)
    user_wc = df.groupby("user_id")["text"].apply(lambda s: s.astype(str).apply(lambda t: len(t.split())).mean())
    print("avg wordcount per user (sample):", user_wc.sort_values().head(5).round(2).to_dict(), "...", user_wc.sort_values(ascending=False).head(5).round(2).to_dict())

    # PII signal coverage
    pii_counter = Counter([sig for L in df["pii_signals_list"] for sig in L])
    print("top pii_signals:", pii_counter.most_common(15))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "synthetic_social_privacy_drift_seed42.csv"
    validate_synth_data(path)
