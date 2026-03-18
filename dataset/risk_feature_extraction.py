import json
from typing import Any, Dict, Optional, List, Iterable, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig



ANNOTATION_PROMPT = """You are an annotation assistant for privacy-risk research aligned with the CPRS-C formulation.
        Important: Do not diagnose mental health. Do not infer hidden traits. Label only observable textual signals.
        Task: Given a social media post, extract features consistent with the taxonomy:
        Direct Exposure x(E):
        • Explicit PII (email/phone/address/ID/coordinate)
        • Sensitive disclosure (health/finance/employment/legal_immigration/relationship)
        • Location exposure (real-time or identifiable place)
        • Visibility amplifier
        • External surface (URL/contact request)
        Contextual Vulnerability x(V ):
        • distress_cues
        • urgency_panic_cues
        • crisis_impulsivity_cues
        • coercion_solicitation_cues
        • conflict_harassment_cues
        • oversharing_regret_boundary_cues
        For each signal: Return present: true/false. If present, return intensity: low / medium / high based only on observable linguistic strength.
        Return ONLY valid JSON matching this schema:
        {
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
        },
        "visibility_amplifier": {
        "present": true/false,
        "value": "public/friends/private/unknown",
        "intensity": "low/medium/high/none"
        },
        "external_surface": {
        "present": true/false,
        "intensity": "low/medium/high/none"
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

import json
import time
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BNB = False

from transformers import logging
# logging.set_verbosity_info()
logging.set_verbosity_warning()

class FastLLMAnnotator:
    def __init__(
        self,
        model_name: str,
        prompt: str,
        batch_size: int = 16,
        max_new_tokens: int = 192,
        max_length: int = 1536,          # reduce to speed up tokenization + avoid huge padding
        retries: int = 0,
        verbose: bool = True,
        # speed knobs:
        use_4bit: bool = True,           # biggest win
        force_cuda: bool = True,         # avoid slow CPU/disk offload
        attn_implementation: str = "sdpa" # "flash_attention_2" if installed, else "sdpa"
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
        start = text.find("{")
        if start == -1:
            return None
        stack = []
        for i in range(start, len(text)):
            if text[i] == "{":
                stack.append("{")
            elif text[i] == "}":
                if stack:
                    stack.pop()
                    if not stack:
                        candidate = text[start : i + 1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            return None
        return None

    @staticmethod
    def validate_schema(obj: Dict[str, Any]) -> bool:
        required = [
            "explicit_pii",
            "sensitive_disclosure",
            "location_exposure",
            "visibility_amplifier",
            "external_surface",
            "contextual_vulnerability",
        ]
        return isinstance(obj, dict) and all(k in obj for k in required)

    # ---------- Load model ----------
    def load(self):
        t0 = time.time()
        if self.verbose:
            print(f"[FastLLMAnnotator] Loading tokenizer: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.tokenizer.padding_side = "left"  # decoder-only best practice
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Decide device_map
        if self.force_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("force_cuda=True but CUDA not available.")
            device_map = "cuda:0"
        else:
            device_map = "auto"

        quant_config = None
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

        if self.use_4bit:
            if not _HAS_BNB:
                raise RuntimeError(
                    "use_4bit=True but bitsandbytes/BitsAndBytesConfig not available. "
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
            attn_implementation=self.attn_implementation,  # "sdpa" usually available; flash_attention_2 if installed
            low_cpu_mem_usage=True,
        )

        # Deterministic generation (prevents any hidden sampling flags)
        # self.model.generation_config = GenerationConfig(do_sample=False)
        self.model.generation_config = GenerationConfig(
            do_sample=False,   # greedy
            temperature=None,
            top_p=None,
            top_k=None,
        )

        # Speed: kv-cache
        self.model.config.use_cache = True

        if self.verbose:
            print("[FastLLMAnnotator] Loaded in %.2fs" % (time.time() - t0))
            print("[FastLLMAnnotator] hf_device_map:", getattr(self.model, "hf_device_map", None))

        # Warmup to compile kernels / reduce first-step latency
        # self._warmup()

    def _warmup(self):
        if not self.verbose:
            return
        try:
            dummy_posts = ["Return {\"ok\": true} as JSON only."] * min(2, self.batch_size)
            _ = self.annotate_batch(dummy_posts)
            print("[FastLLMAnnotator] Warmup done.")
        except Exception as e:
            print("[FastLLMAnnotator] Warmup skipped:", str(e))

    def _ensure_loaded(self):
        if self.model is None or self.tokenizer is None:
            self.load()

    # ---------- Prompt building ----------
    def _build_prompts(self, posts: List[str]) -> List[str]:
        prompts = []
        for t in posts:
            messages = [
                {"role": "system", "content": "Return ONLY valid JSON. No markdown. No extra text."},
                {"role": "user", "content": self.prompt + "\n\nPOST:\n" + str(t)},
            ]
            prompts.append(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        return prompts

    def _build_prompts(self, posts):
        prompts = []
        for t in posts:
            messages = [
                {"role": "system", "content": "Return ONLY valid JSON. No markdown. No extra text."},
                {"role": "user", "content": self.prompt + "\n\nPOST:\n" + str(t)},
            ]
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            else:
                # fallback: simple concatenation (works, but less ideal)
                prompts.append(
                    "SYSTEM: Return ONLY valid JSON. No markdown. No extra text.\n"
                    "USER: " + self.prompt + "\n\nPOST:\n" + str(t) + "\nASSISTANT:"
                )
        return prompts

    # ---------- Batched generate ----------
    @torch.inference_mode()
    def annotate_batch(self, posts: List[str]) -> Tuple[List[Optional[Dict[str, Any]]], List[str]]:
        self._ensure_loaded()

        print("called annotate_batch...")
        prompts = self._build_prompts(posts)
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

        decoded = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        objs: List[Optional[Dict[str, Any]]] = []
        for d in decoded:
            obj = self.extract_json_object(d)
            if obj and self.validate_schema(obj):
                objs.append(obj)
            else:
                objs.append(None)
        return objs, decoded

    def annotate_dataframe(self, df: pd.DataFrame, text_col: str = "post_text") -> pd.DataFrame:
        self._ensure_loaded()
        df = df.copy()

        texts = df[text_col].astype(str).tolist()
        ann_list: List[Optional[Dict[str, Any]]] = [None] * len(df)
        raw_list: List[str] = [""] * len(df)

        t0 = time.time()
        print("batching...")
        for start in range(0, len(texts), self.batch_size):
            end = min(start + self.batch_size, len(texts))
            batch = texts[start:end]

            if self.verbose:
                rate = (start / max(1e-9, (time.time() - t0)))
                print(f"[FastLLMAnnotator] {start}/{len(texts)} | batch {start}:{end} | ~{rate:.2f} rows/s")

            objs, raws = self.annotate_batch(batch)

            # optional retry for failures (kept off by default for speed)
            if self.retries > 0 and any(o is None for o in objs):
                retry_texts = [t for t, o in zip(batch, objs) if o is None]
                retry_pos = [i for i, o in enumerate(objs) if o is None]
                if retry_texts:
                    ro, rr = self.annotate_batch(retry_texts)
                    for p, o2, r2 in zip(retry_pos, ro, rr):
                        if o2 is not None:
                            objs[p] = o2
                            raws[p] = r2

            ann_list[start:end] = objs
            raw_list[start:end] = raws

        df["llm_annotation"] = ann_list
        df["llm_annotation_json"] = [
            json.dumps(x, ensure_ascii=False) if x else None for x in ann_list
        ]
        df["llm_raw_output"] = raw_list
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