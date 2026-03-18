import random
import uuid
from datetime import date, timedelta

# ----------------------------
# Config
# ----------------------------
USERS = [
  {"user_id":"user_01","archetype":"student","privacy_awareness":"medium"},
  {"user_id":"user_02","archetype":"professional","privacy_awareness":"low"},
  {"user_id":"user_03","archetype":"older_adult","privacy_awareness":"high"},
  {"user_id":"user_04","archetype":"business_owner","privacy_awareness":"medium"},
  {"user_id":"user_05","archetype":"activist","privacy_awareness":"low"},
]

THREATS = ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12"]

START_DATE = date(2026, 3, 1)
DAYS = 30
MIN_POSTS, MAX_POSTS = 60, 100
HOURS = list(range(0,24))

# Keep uniqueness globally (or per user)
seen_texts = set()

# ----------------------------
# Helper: phase by day_index
# ----------------------------
def phase(day_idx):
    if day_idx <= 9: return "baseline"
    if day_idx <= 19: return "escalation"
    if day_idx <= 24: return "spike"
    return "recovery"

# ----------------------------
# Helper: sample posts per day (0-5) with bursts + silence
# ----------------------------
def sample_posts_today(user, threat, day_idx):
    # You can condition on threat; this is simple:
    # base distribution with occasional zeros and occasional 4-5 bursts
    r = random.random()
    if r < 0.20: return 0
    if r < 0.70: return random.randint(1,2)
    if r < 0.90: return random.randint(3,4)
    return 5

# ----------------------------
# Generate daily platform events + telemetry using threat schedules
# ----------------------------
def sample_daily_events_and_telemetry(user, threat, ph):
    # Start with baselines
    events = {
      "friend_requests_received": 0,
      "friend_requests_accepted": 0,
      "follower_additions": 0,
      "follower_removals": 0,
      "profile_edits": [{"field":"bio","count":0},{"field":"work","count":0},
                        {"field":"education","count":0},{"field":"location","count":0},
                        {"field":"relationship","count":0},{"field":"photo","count":0}],
      "oauth_logins": 0,
      "third_party_apps_used": ["none"],
      "messages_from_unknown": 0,
      "reports_or_blocks": 0
    }
    telemetry = {
      "impressions_total": random.randint(50, 400),
      "impressions_nonfollowers": random.randint(0, 120),
      "profile_views": random.randint(0, 30),
      "link_clicks": 0
    }

    # Override by threat + phase (illustrative patterns)
    if threat == "T6":
        if ph == "escalation":
            events["friend_requests_received"] = random.randint(2,6)
            events["friend_requests_accepted"] = random.randint(1,3)
        if ph == "spike":
            events["friend_requests_received"] = random.randint(8,14)
            events["friend_requests_accepted"] = random.randint(4,8)
            events["messages_from_unknown"] = random.randint(1,5)
            # profile edits not user-announced; platform metadata
            events["profile_edits"][0]["count"] = random.randint(0,1)  # bio
            events["profile_edits"][4]["count"] = random.randint(0,1)  # relationship

    if threat == "T7":
        if ph == "escalation":
            events["follower_additions"] = random.randint(3,12)
        if ph == "spike":
            events["follower_additions"] = random.randint(15,40)
            telemetry["impressions_nonfollowers"] += random.randint(80,200)
            telemetry["profile_views"] += random.randint(10,50)

    if threat == "T11":
        if ph in ["escalation","spike"]:
            events["oauth_logins"] = random.randint(1,5)
            events["third_party_apps_used"] = [random.choice(["spotify","strava","canva","game_app","other"])]
            events["messages_from_unknown"] = random.randint(1,4)

    if threat == "T12":
        # surveillance-passivity = low posts + high nonfollower views (later combined)
        if ph in ["escalation","spike"]:
            telemetry["impressions_nonfollowers"] += random.randint(100,250)
            telemetry["profile_views"] += random.randint(10,40)

    return events, telemetry

# ----------------------------
# LLM call stub: generate one post JSON "post" block + exposure labels
# (You will implement using your prompt)
# ----------------------------
def llm_generate_post_text_and_labels(context):
    """
    Returns dict for `post` with:
    text, visibility, category, contains_pii, pii_type, sensitive_disclosure_type,
    location_exposure, location_hint, external_surface, urls, contextual_cues_present,
    cross_platform_markers
    """
    raise NotImplementedError

# ----------------------------
# Main generation
# ----------------------------
def generate_dataset():
    dataset = []
    for u in USERS:
        threat = random.choice(THREATS)
        target_posts = random.randint(MIN_POSTS, MAX_POSTS)
        produced = 0

        for day_idx in range(1, DAYS+1):
            ph = phase(day_idx)
            d = START_DATE + timedelta(days=day_idx-1)

            daily_events, daily_tel = sample_daily_events_and_telemetry(u, threat, ph)
            n_posts = sample_posts_today(u, threat, day_idx)

            for _ in range(n_posts):
                if produced >= target_posts:
                    break

                hour = random.choice(HOURS)
                context = {
                  "user": u,
                  "threat": threat,
                  "drift_phase": ph,
                  "day_index": day_idx,
                  "date": d.isoformat(),
                  "hour": hour,
                  "platform_events_daily": daily_events,
                  "telemetry_daily": daily_tel,
                  # optionally pass rolling history summaries for novelty, coherence, etc.
                }

                post_block = llm_generate_post_text_and_labels(context)

                # enforce uniqueness (retry or regenerate)
                if post_block["text"] in seen_texts:
                    continue
                seen_texts.add(post_block["text"])

                record = {
                  "schema_version":"cbprs-c.v2",
                  "record_id": str(uuid.uuid4()),
                  "user": {**u, "assigned_threat": threat},
                  "time": {"day_index": day_idx, "date": d.isoformat(), "hour": hour},
                  "post": post_block,
                  "platform_events_daily": daily_events,
                  "telemetry_daily": daily_tel,
                  "labels": {"drift_phase": ph, "drift_event": False, "threat_type": threat}
                }

                dataset.append(record)
                produced += 1

            if produced >= target_posts:
                break

    return dataset