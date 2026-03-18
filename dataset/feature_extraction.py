from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from tqdm.auto import tqdm
import time
import json
from typing import Optional, List, Dict, Any
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
import json
from typing import Any, Dict, Optional, List, Iterable, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig



ANNOTATION_PROMPT = """
    You are an annotation assistant for privacy-risk research aligned with the CBPRS-C formulation.

    IMPORTANT CONSTRAINTS:
    - Do NOT diagnose mental health.
    - Do NOT infer hidden personality traits.
    - Label ONLY observable textual signals.
    - If a signal is not explicitly present in the text, mark it as present: false.
    - Base intensity ONLY on linguistic strength and clarity.
    - Do NOT fabricate missing information.
    - Return ONLY valid JSON.

    --------------------------------------------------
    TASK
    --------------------------------------------------

    Given a social media post, extract structured signals required for CBPRS-C risk computation and drift modeling.

    We distinguish:

    1) Direct Exposure Signals x(E)  → linear risk channels
    2) Exposure Amplifiers            → visibility and external reach
    3) Contextual Vulnerability x(V) → multiplicative modifiers

    --------------------------------------------------
    DIRECT EXPOSURE x(E)
    --------------------------------------------------

    A. Explicit PII (spii)
    Observable presence of:
    - email
    - phone number
    - physical address
    - ID fragment
    - geographic coordinate

    B. Sensitive Disclosure (ssens)
    Explicit personal disclosure in:
    - health
    - finance
    - employment
    - legal_immigration
    - relationship

    C. Location Exposure (sloc)
    Real-time presence or identifiable venue/city.

    --------------------------------------------------
    EXPOSURE AMPLIFIERS
    --------------------------------------------------

    D. Visibility Amplifier (svis)
    If the text explicitly indicates:
    - public sharing
    - “everyone can see this”
    - “posting publicly”
    - “friends only”
    - “private”
    Otherwise value = "unknown".

    E. External Surface (slink)
    - Presence of URL
    - Link shortener
    - Explicit contact request
    - “click here”
    - “DM me”
    - “message me at”
    - Any outbound surface increasing exposure

    --------------------------------------------------
    CONTEXTUAL VULNERABILITY x(V)
    --------------------------------------------------

    Label ONLY if linguistically observable:

    - distress_cues
    - urgency_panic_cues
    - crisis_impulsivity_cues
    - coercion_solicitation_cues
    - conflict_harassment_cues
    - oversharing_regret_boundary_cues

    Intensity definition:
    - low: mild mention
    - medium: clearly expressed
    - high: strong, urgent, emotionally escalated language
    - none: if not present

    --------------------------------------------------
    OUTPUT FORMAT
    --------------------------------------------------

    Return ONLY valid JSON matching exactly:

    {
    "direct_exposure": {
        "explicit_pii": {
        "present": true/false,
        "pii_type": "email/phone/address/ID/coordinate/none",
        "intensity": "low/medium/high/none"
        },
        "sensitive_disclosure": {
        "present": true/false,
        "type": "none/health/finance/employment/legal_immigration/relationship",
        "intensity": "low/medium/high/none"
        },
        "location_exposure": {
        "present": true/false,
        "intensity": "low/medium/high/none"
        }
    },
    "exposure_amplifiers": {
        "visibility_amplifier": {
        "present": true/false,
        "value": "public/friends/private/unknown",
        "intensity": "low/medium/high/none"
        },
        "external_surface": {
        "present": true/false,
        "intensity": "low/medium/high/none"
        }
    },
    "contextual_vulnerability": {
        "distress_cues": {"present": true/false, "intensity": "low/medium/high/none"},
        "urgency_panic_cues": {"present": true/false, "intensity": "low/medium/high/none"},
        "crisis_impulsivity_cues": {"present": true/false, "intensity": "low/medium/high/none"},
        "coercion_solicitation_cues": {"present": true/false, "intensity": "low/medium/high/none"},
        "conflict_harassment_cues": {"present": true/false, "intensity": "low/medium/high/none"},
        "oversharing_regret_boundary_cues": {"present": true/false, "intensity": "low/medium/high/none"}
    }
    }
    """




# Optional 4-bit quant
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BNB = False


# ----------------------------
# Updated schema validator (matches NEW prompt output)
# ----------------------------
REQUIRED_TOP_KEYS = ("direct_exposure", "exposure_amplifiers", "contextual_vulnerability")

REQUIRED_DIRECT_EXPOSURE = ("explicit_pii", "sensitive_disclosure", "location_exposure")
REQUIRED_EXPOSURE_AMPLIFIERS = ("visibility_amplifier", "external_surface")
REQUIRED_CONTEXTUAL_KEYS = (
    "distress_cues",
    "urgency_panic_cues",
    "crisis_impulsivity_cues",
    "coercion_solicitation_cues",
    "conflict_harassment_cues",
    "oversharing_regret_boundary_cues",
)

VALID_INTENSITY = {"low", "medium", "high", "none"}
VALID_VIS = {"public", "friends", "private", "unknown"}
VALID_PII = {"email", "phone", "address", "ID", "coordinate", "none"}
VALID_SENS = {"none", "health", "finance", "employment", "legal_immigration", "relationship"}


def _is_obj(x) -> bool:
    return isinstance(x, dict)


def validate_annotation_schema(obj: Dict[str, Any]) -> bool:
    if not _is_obj(obj):
        return False
    if any(k not in obj for k in REQUIRED_TOP_KEYS):
        return False

    de = obj.get("direct_exposure")
    ea = obj.get("exposure_amplifiers")
    cv = obj.get("contextual_vulnerability")
    if not (_is_obj(de) and _is_obj(ea) and _is_obj(cv)):
        return False

    if any(k not in de for k in REQUIRED_DIRECT_EXPOSURE):
        return False
    if any(k not in ea for k in REQUIRED_EXPOSURE_AMPLIFIERS):
        return False
    if any(k not in cv for k in REQUIRED_CONTEXTUAL_KEYS):
        return False

    # lightweight value checks (don’t be too strict; keep robust)
    try:
        # explicit_pii
        ep = de["explicit_pii"]
        if not _is_obj(ep): return False
        if not isinstance(ep.get("present"), bool): return False
        if ep.get("pii_type") not in VALID_PII: return False
        if ep.get("intensity") not in VALID_INTENSITY: return False

        # sensitive_disclosure
        sd = de["sensitive_disclosure"]
        if not _is_obj(sd): return False
        if not isinstance(sd.get("present"), bool): return False
        if sd.get("type") not in VALID_SENS: return False
        if sd.get("intensity") not in VALID_INTENSITY: return False

        # location_exposure
        le = de["location_exposure"]
        if not _is_obj(le): return False
        if not isinstance(le.get("present"), bool): return False
        if le.get("intensity") not in VALID_INTENSITY: return False

        # visibility_amplifier
        va = ea["visibility_amplifier"]
        if not _is_obj(va): return False
        if not isinstance(va.get("present"), bool): return False
        if va.get("value") not in VALID_VIS: return False
        if va.get("intensity") not in VALID_INTENSITY: return False

        # external_surface
        es = ea["external_surface"]
        if not _is_obj(es): return False
        if not isinstance(es.get("present"), bool): return False
        if es.get("intensity") not in VALID_INTENSITY: return False

        # contextual keys
        for k in REQUIRED_CONTEXTUAL_KEYS:
            ck = cv[k]
            if not _is_obj(ck): return False
            if not isinstance(ck.get("present"), bool): return False
            if ck.get("intensity") not in VALID_INTENSITY: return False

    except Exception:
        return False

    return True


# ----------------------------
# Annotator
# ----------------------------
class FastLLMAnnotator:
    """
    Updated for:
    - NEW annotation JSON schema (direct_exposure / exposure_amplifiers / contextual_vulnerability)
    - robust JSON extraction (handles extra chatter)
    - single _build_prompts definition (your code had duplicates)
    - better generation slicing (decode ONLY generated continuation when possible)
    - optional metadata injection (visibility, urls) to reduce LLM confusion
    """

    def __init__(
        self,
        model_name: str,
        prompt: str,
        batch_size: int = 16,
        max_new_tokens: int = 512,
        max_length: int = 1536,
        retries: int = 0,
        verbose: bool = True,
        use_4bit: bool = True,
        force_cuda: bool = True,
        attn_implementation: str = "sdpa",  # "flash_attention_2" if installed
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.retries = retries
        self.verbose = verbose
        self.use_4bit = use_4bit
        self.force_cuda = force_cuda
        self.attn_implementation = attn_implementation

        self.tokenizer = None
        self.model = None

    # ---------- JSON helpers ----------
    @staticmethod
    def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract first balanced {...} JSON object from a string.
        Works even if the model prepends explanations (which it shouldn't).
        """
        start = text.find("{")
        if start == -1:
            return None
        stack = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                stack += 1
            elif ch == "}":
                stack -= 1
                if stack == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        return None
        return None

    # ---------- Load model ----------
    def load(self):
        t0 = time.time()
        if self.verbose:
            print(f"[FastLLMAnnotator] Loading tokenizer: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # device_map
        if self.force_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("force_cuda=True but CUDA not available.")
            device_map = {"": 0}  # avoids slow auto/offload behavior
        else:
            device_map = "auto"

        quant_config = None
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

        if self.use_4bit:
            if not _HAS_BNB:
                raise RuntimeError(
                    "use_4bit=True but BitsAndBytesConfig not available. "
                    "Install: pip install bitsandbytes"
                )
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        if self.verbose:
            print(
                f"[FastLLMAnnotator] Loading model (device_map={device_map}, "
                f"use_4bit={self.use_4bit}, attn={self.attn_implementation})..."
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            torch_dtype=None if self.use_4bit else torch_dtype,
            quantization_config=quant_config,
            attn_implementation=self.attn_implementation,
            low_cpu_mem_usage=True,
        )

        # deterministic
        self.model.generation_config = GenerationConfig(
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        self.model.config.use_cache = True

        if self.verbose:
            print("[FastLLMAnnotator] Loaded in %.2fs" % (time.time() - t0))
            print("[FastLLMAnnotator] hf_device_map:", getattr(self.model, "hf_device_map", None))

        # optional warmup
        # self._warmup()

    def _ensure_loaded(self):
        if self.model is None or self.tokenizer is None:
            self.load()

    # ---------- Prompt building ----------
    def _build_prompts(
        self,
        posts: List[str],
        *,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Optionally include metadata per post to reduce ambiguity.
        metadata[i] may include fields like:
          - visibility
          - urls
          - external_surface
          - location_exposure
        The prompt should still instruct: label only observable text, but metadata helps
        the model avoid contradictions if you want (set include_metadata=False otherwise).
        """
        prompts: List[str] = []
        for i, t in enumerate(posts):
            meta_block = ""
            if metadata is not None:
                m = metadata[i] if i < len(metadata) else {}
                # keep metadata minimal and explicit
                meta_block = "\n\nMETADATA (may help disambiguate visibility/urls; do not invent):\n" + json.dumps(
                    m, ensure_ascii=False
                )

            messages = [
                {"role":"system","content":"Return ONLY valid JSON in ONE line (no pretty print). No markdown. No extra text."},
                {"role": "user", "content": self.prompt + meta_block + "\n\nPOST:\n" + str(t)},
            ]

            if hasattr(self.tokenizer, "apply_chat_template"):
                prompts.append(
                    self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
            else:
                prompts.append(
                    "SYSTEM: Return ONLY valid JSON in ONE line (no pretty print). No markdown. No extra text.\n"
                    "USER: " + (self.prompt + meta_block) + "\n\nPOST:\n" + str(t) + "\nASSISTANT:"
                )
        return prompts

    # ---------- Batched generate ----------
    @torch.inference_mode()
    def annotate_batch(
        self,
        posts: List[str],
        *,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Optional[Dict[str, Any]]], List[str], float]:
        """
        Returns:
        objs: parsed annotation dicts (or None)
        decoded: decoded generated continuations (one per input)
        avg_inference_time_sec: average (per-post) inference time for this batch,
                                including tokenization + generate + decode + parse.
        """
        self._ensure_loaded()

        # (Optional but recommended for accurate GPU timing)
        if torch.cuda.is_available() and getattr(self.model, "device", None) is not None and "cuda" in str(self.model.device):
            torch.cuda.synchronize()
        t0 = time.time()

        prompts = self._build_prompts(posts, metadata=metadata)

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.model.device)

        out_ids = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode ONLY the generated continuation
        input_lens = enc["input_ids"].shape[1]
        gen_only = out_ids[:, input_lens:]
        decoded = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        objs: List[Optional[Dict[str, Any]]] = []
        for d in decoded:
            obj = self.extract_json_object(d)
            if obj and validate_annotation_schema(obj):
                objs.append(obj)
            else:
                objs.append(None)

        if torch.cuda.is_available() and getattr(self.model, "device", None) is not None and "cuda" in str(self.model.device):
            torch.cuda.synchronize()
        t1 = time.time()

        batch_time = t1 - t0
        avg_time = batch_time / max(1, len(posts))

        return objs, decoded, float(avg_time)



    def annotate_dataframe(
        self,
        df: pd.DataFrame,
        *,
        text_col: str = "post_text",
        include_metadata: bool = True,
        meta_cols: Optional[List[str]] = None,
        out_ann_col: str = "llm_annotation",
        out_raw_col: str = "llm_raw_output",
    ) -> pd.DataFrame:

        self._ensure_loaded()
        df = df.copy()

        print("annotate_dataframe...")
        print("text_col: ", text_col)

        texts = df[text_col].astype(str).tolist()

        if include_metadata:
            if meta_cols is None:
                cand = ["visibility", "external_surface", "urls", "location_exposure", "location_hint"]
                meta_cols = [c for c in cand if c in df.columns]

            metas = [
                {c: df.iloc[i][c] for c in meta_cols}
                for i in range(len(df))
            ]
        else:
            metas = None

        ann_list: List[Optional[Dict[str, Any]]] = [None] * len(df)
        raw_list: List[str] = [""] * len(df)
        time_list: List[float] = [0.0] * len(df)

        t0 = time.time()

        # -----------------------------
        # TQDM Progress Bar
        # -----------------------------
        pbar = tqdm(
            total=len(texts),
            desc="Annotating",
            unit="rows",
            dynamic_ncols=True
        )

        for start in range(0, len(texts), self.batch_size):
            end = min(start + self.batch_size, len(texts))
            batch = texts[start:end]
            batch_meta = metas[start:end] if metas is not None else None

            objs, raws, avg_time = self.annotate_batch(batch, metadata=batch_meta)
            time_list[start:end] = [avg_time] * (end - start)

            # Retry failures
            if self.retries > 0 and any(o is None for o in objs):
                retry_texts = [t for t, o in zip(batch, objs) if o is None]
                retry_meta = (
                    [m for m, o in zip(batch_meta or [None]*len(batch), objs) if o is None]
                    if batch_meta else None
                )
                retry_pos = [i for i, o in enumerate(objs) if o is None]

                if retry_texts:
                    ro, rr, _ = self.annotate_batch(retry_texts, metadata=retry_meta)
                    for p, o2, r2 in zip(retry_pos, ro, rr):
                        if o2 is not None:
                            objs[p] = o2
                            raws[p] = r2

            ann_list[start:end] = objs
            raw_list[start:end] = raws

            # Update progress bar
            pbar.update(end - start)

            # Optional: show speed in postfix
            elapsed = time.time() - t0
            rate = (end) / max(elapsed, 1e-9)
            pbar.set_postfix({"rows/s": f"{rate:.2f}"})

        pbar.close()

        df["inference_time_sec"] = time_list
        df[out_ann_col] = ann_list
        df[out_raw_col] = raw_list
        df["llm_annotation_json"] = [
            json.dumps(x, ensure_ascii=False) if x is not None else None
            for x in ann_list
        ]
        df["llm_annotation_ok"] = [x is not None for x in ann_list]

        return df

    @staticmethod
    def read_jsonl_robust(path: str) -> pd.DataFrame:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rows.append(json.loads(s))
                except Exception:
                    continue
        return pd.DataFrame(rows)


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    """
    Example:
      1) Flatten your JSONL into a DataFrame with at least:
         - post_text column (or rename)
         - optionally: visibility, urls, external_surface, location_exposure
      2) Run annotator and save results.
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to cbprs-c.v2.1 JSONL")
    ap.add_argument("--model", required=True, help="HF model name, e.g., meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--out_csv", default="annotated.csv")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no_meta", action="store_true", help="Do not pass metadata into the prompt")
    args = ap.parse_args()

    df = FastLLMAnnotator.read_jsonl_robust(args.jsonl)

    # Flatten post.text into a working column
    if "post_text" not in df.columns:
        df["post_text"] = df["post"].apply(lambda p: (p or {}).get("text", ""))

    # Optional flatten metadata columns (only if not already flattened)
    if "visibility" not in df.columns:
        df["visibility"] = df["post"].apply(lambda p: (p or {}).get("visibility", "unknown"))
    if "external_surface" not in df.columns:
        df["external_surface"] = df["post"].apply(lambda p: bool((p or {}).get("external_surface", False)))
    if "urls" not in df.columns:
        df["urls"] = df["post"].apply(lambda p: (p or {}).get("urls", []))
    if "location_exposure" not in df.columns:
        df["location_exposure"] = df["post"].apply(lambda p: bool((p or {}).get("location_exposure", False)))
    if "location_hint" not in df.columns:
        df["location_hint"] = df["post"].apply(lambda p: (p or {}).get("location_hint", "none"))

    annotator = FastLLMAnnotator(
        model_name=args.model,
        prompt="",  # set your updated ANNOTATION_PROMPT here
        batch_size=args.batch,
        max_new_tokens=220,
        max_length=1536,
        retries=0,
        verbose=True,
        use_4bit=True,
        force_cuda=True,
        attn_implementation="sdpa",
    )

    df2 = annotator.annotate_dataframe(
        df,
        text_col="post_text",
        include_metadata=not args.no_meta,
        meta_cols=["visibility", "external_surface", "urls", "location_exposure", "location_hint"],
    )

    df2.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)