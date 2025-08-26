
from typing import List, Dict, Any
import re

DEFAULT_BAD_DOMAINS = {"badnews.example", "clickbait.example"}
TOPIC_KEYWORDS = {"politics", "election", "hoax", "rumor", "scam", "vaccine"}

URL_RE = re.compile(r"https?://([^/\s]+)")

def extract_domains(text: str) -> List[str]:
    return URL_RE.findall(text or "")

def misinfo_context(texts_in_window: List[str]) -> Dict[str, Any]:
    domains = []
    topics = set()
    for tx in texts_in_window:
        domains += extract_domains(tx)
        low = (tx or "").lower()
        for kw in TOPIC_KEYWORDS:
            if kw in low:
                topics.add(kw)
    bad_hits = [d for d in domains if any(bad in d for bad in DEFAULT_BAD_DOMAINS)]
    return {
        "bad_domain_hits": bad_hits,
        "topics": sorted(list(topics)),
        "engaged": bool(bad_hits or topics)
    }