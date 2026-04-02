from __future__ import annotations

import pandas as pd

from healthcare_fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def join_claims_with_dimensions(
    claims: pd.DataFrame,
    patients: pd.DataFrame,
    providers: pd.DataFrame,
) -> pd.DataFrame:
    """Inner join so every claim has exactly one patient and one provider row."""
    before = len(claims)
    merged = claims.merge(
        patients,
        on="patient_id",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_patient"),
    )
    lost_p = before - len(merged)
    if lost_p:
        logger.warning("Dropped %s claims with no matching patient", lost_p)

    step = len(merged)
    merged = merged.merge(
        providers,
        on="provider_id",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_provider"),
    )
    lost_v = step - len(merged)
    if lost_v:
        logger.warning("Dropped %s rows with no matching provider", lost_v)

    return merged.reset_index(drop=True)


def enforce_processed_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["claim_date"] = pd.to_datetime(out["claim_date"]).dt.normalize()
    out["claim_amount"] = out["claim_amount"].astype("float64")
    out["fraud"] = out["fraud"].astype("int64")
    out["age"] = pd.to_numeric(out["age"], errors="coerce").round().astype("int64")

    string_cols = [
        "claim_id",
        "patient_id",
        "provider_id",
        "procedure_code",
        "gender",
        "chronic_condition",
        "specialization",
        "region",
    ]
    for c in string_cols:
        if c in out.columns:
            out[c] = out[c].astype("string")
    return out
