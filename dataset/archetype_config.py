# synth_privacy_drift.py

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For drift detectors
from river import drift  # ADWIN
from functools import partial
import torch
from bayesian_changepoint_detection import (
    online_changepoint_detection,
    constant_hazard,
    StudentT,
)
from signal_processing_algorithms.energy_statistics import energy_statistics



# ---------- 3. Pure-Python generator with configurable drift & adversaries ----------

ARCHETYPES = [
    "stable_normal",
    "oversharer",
    "compromised",
    "spammer",
    "cold_user",
    "high_privacy",
]

ADVERSARY_TYPES = [
    "none",
    "phishing",
    "scam",
    "prompt_injection",
]


@dataclass
class SynthConfig:
    n_users: int = 10
    days: int = 120
    base_date: datetime = datetime(2025, 1, 1)

    # Drift severity: 0 (barely any change) to 1 (very sharp spikes)
    drift_severity: float = 0.7

    # Mixture of user archetypes (will be normalized)
    archetype_mix: Dict[str, float] = None

    # Mixture of adversary types
    adversary_mix: Dict[str, float] = None

    seed: int = 7

    def __post_init__(self):
        if self.archetype_mix is None:
            self.archetype_mix = {
                "stable_normal": 0.25,
                "oversharer": 0.2,
                "compromised": 0.2,
                "spammer": 0.15,
                "cold_user": 0.1,
                "high_privacy": 0.1,
            }
        if self.adversary_mix is None:
            self.adversary_mix = {
                "none": 0.6,
                "phishing": 0.15,
                "scam": 0.15,
                "prompt_injection": 0.1,
            }
