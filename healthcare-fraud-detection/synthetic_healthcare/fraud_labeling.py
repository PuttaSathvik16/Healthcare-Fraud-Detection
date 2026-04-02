from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_healthcare.constants import (
    BURST_WINDOW_DAYS,
    HIGH_AMOUNT_QUANTILE_BASELINE,
    HIGH_AMOUNT_RATIO_OVER_P99,
    MAX_CLAIMS_PER_BURST_WINDOW,
)


def _high_amount_flags(
    claims: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.Series:
    p99 = baseline.groupby("procedure_code")["claim_amount"].quantile(HIGH_AMOUNT_QUANTILE_BASELINE)
    global_p99 = float(baseline["claim_amount"].quantile(0.995))
    thresholds = claims["procedure_code"].map(p99).fillna(global_p99) * HIGH_AMOUNT_RATIO_OVER_P99
    return claims["claim_amount"] > thresholds


def _duplicate_flags(claims: pd.DataFrame) -> pd.Series:
    key_cols = ["patient_id", "provider_id", "procedure_code", "claim_amount", "claim_date"]
    return claims.duplicated(subset=key_cols, keep=False)


def _burst_flags(claims: pd.DataFrame) -> pd.Series:
    c = claims[["patient_id", "claim_date"]].copy()
    c["claim_date"] = pd.to_datetime(c["claim_date"])
    c = c.sort_values(["patient_id", "claim_date"])
    result = pd.Series(False, index=claims.index)
    for _, grp in c.groupby("patient_id", sort=False):
        idx = grp.index.to_numpy()
        dts = grp["claim_date"].to_numpy(dtype="datetime64[ns]")
        m = len(grp)
        local_burst = np.zeros(m, dtype=bool)
        left = 0
        for right in range(m):
            while left <= right:
                delta_days = int((dts[right] - dts[left]) / np.timedelta64(1, "D"))
                if delta_days > BURST_WINDOW_DAYS:
                    left += 1
                else:
                    break
            if right - left + 1 > MAX_CLAIMS_PER_BURST_WINDOW:
                local_burst[left : right + 1] = True
        result.loc[idx] = local_burst
    return result


def label_fraud(claims: pd.DataFrame, baseline_claims: pd.DataFrame) -> pd.DataFrame:
    """
    fraud = 1 if any of:
      - claim_amount unusually high vs procedure-specific baseline (p99 on clean data)
      - duplicate rows on (patient_id, provider_id, procedure_code, claim_amount, claim_date)
      - more than MAX_CLAIMS_PER_BURST_WINDOW claims within BURST_WINDOW_DAYS for the patient
    """
    out = claims.copy()
    high = _high_amount_flags(out, baseline_claims)
    dup = _duplicate_flags(out)
    burst = _burst_flags(out)
    out["fraud"] = (high | dup | burst).astype(np.int8)
    return out
