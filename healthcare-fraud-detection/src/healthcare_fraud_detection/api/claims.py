"""Map API claim payloads to the feature columns expected by the trained preprocessor."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from healthcare_fraud_detection.api.schemas import ClaimInput


def claims_to_dataframe(claims: Sequence[ClaimInput]) -> pd.DataFrame:
    rows: list[dict] = []
    for c in claims:
        d = c.model_dump()
        ts = pd.to_datetime(d.pop("claim_date"))
        d["claim_month"] = int(ts.month)
        d["claim_dow"] = int(ts.dayofweek)
        rows.append(d)
    return pd.DataFrame(rows)
