
import re
from typing import Dict, Any, Optional
import spacy

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s-])?(?:\(\d{3}\)|\d{3})[\s-]?)?\d{3}[\s-]?\d{4}")
URL_RE = re.compile(r"https?://\S+")

# Lightweight keyword lists (extend as needed)
HEALTH_TERMS = {"diagnosed", "hospital", "prescription", "therapy", "covid", "depression", "anxiety"}
FINANCE_TERMS = {"salary", "ssn", "bank", "debit", "credit card", "loan", "tax", "bitcoin", "wallet"}
ADDRESS_TERMS = {"street", "avenue", "road", "boulevard", "apt", "apartment", "zipcode", "zip", "postal"}

# Try to load a spaCy pipeline; fall back to a blank English model if model missing
def _load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
        ruler = nlp.add_pipe("entity_ruler")
        ruler.add_patterns([
            {"label": "PERSON", "pattern": [{"IS_TITLE": True, "OP": "+"}]},
            {"label": "GPE", "pattern": [{"IS_ALPHA": True, "OP": "+"}]},
        ])
        return nlp

_NLP = _load_nlp()

def extract_text_features(text: str) -> Dict[str, Any]:
    """
    Extracts privacy-relevant features from free text.
    Returns a dictionary of counts/booleans.
    """
    text_lower = text.lower() if text else ""

    emails = EMAIL_RE.findall(text or "")
    phones = PHONE_RE.findall(text or "")
    urls = URL_RE.findall(text or "")

    # Keyword hits
    health_hits = sum(1 for w in HEALTH_TERMS if w in text_lower)
    finance_hits = sum(1 for w in FINANCE_TERMS if w in text_lower)
    address_hits = sum(1 for w in ADDRESS_TERMS if w in text_lower)

    # NER
    doc = _NLP(text or "")
    n_person = sum(1 for e in doc.ents if e.label_ == "PERSON")
    n_gpe = sum(1 for e in doc.ents if e.label_ in {"GPE", "LOC"})
    n_org = sum(1 for e in doc.ents if e.label_ == "ORG")

    return {
        "n_emails": len(emails),
        "n_phones": len(phones),
        "n_urls": len(urls),
        "health_hits": int(health_hits),
        "finance_hits": int(finance_hits),
        "address_hits": int(address_hits),
        "ner_person": int(n_person),
        "ner_gpe": int(n_gpe),
        "ner_org": int(n_org),
    }

def build_feature_vector(
    text: str,
    audience: Optional[str] = None,
    geotag: Optional[bool] = False,
    has_media: Optional[bool] = False
) -> Dict[str, Any]:
    """
    Compose the full per-post feature vector, including contextual multipliers.
    `audience` in {"public", "friends", "private"} is recommended.
    """
    feats = extract_text_features(text or "")
    # Contextual flags
    feats["audience_public"] = 1 if (audience or "").lower() == "public" else 0
    feats["audience_friends"] = 1 if (audience or "").lower() == "friends" else 0
    feats["audience_private"] = 1 if (audience or "").lower() == "private" else 0
    feats["geotag_on"] = 1 if geotag else 0
    feats["has_media"] = 1 if has_media else 0
    return feats