from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_healthcare.constants import CHRONIC_CONDITIONS, GENDER_WEIGHTS, GENDERS


def generate_patients(n: int, *, seed: int | None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.array([f"PAT-{i:06d}" for i in range(1, n + 1)], dtype=object)
    age = np.clip(rng.normal(48.0, 19.0, size=n).astype(int), 0, 95)
    gender = rng.choice(np.array(GENDERS, dtype=object), size=n, p=np.array(GENDER_WEIGHTS))
    p_none = 0.32
    p_rest = (1.0 - p_none) / (len(CHRONIC_CONDITIONS) - 1)
    probs = np.array([p_none if c == "none" else p_rest for c in CHRONIC_CONDITIONS])
    chronic = rng.choice(np.array(CHRONIC_CONDITIONS, dtype=object), size=n, p=probs)

    return pd.DataFrame(
        {
            "patient_id": ids,
            "age": age.astype(np.int16),
            "gender": gender,
            "chronic_condition": chronic,
        }
    )
