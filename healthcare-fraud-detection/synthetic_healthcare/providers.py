from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_healthcare.constants import REGIONS, SPECIALIZATIONS


def generate_providers(n: int, *, seed: int | None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.array([f"PRV-{i:05d}" for i in range(1, n + 1)], dtype=object)
    specialization = rng.choice(np.array(SPECIALIZATIONS, dtype=object), size=n)
    region = rng.choice(np.array(REGIONS, dtype=object), size=n)

    return pd.DataFrame(
        {
            "provider_id": ids,
            "specialization": specialization,
            "region": region,
        }
    )
