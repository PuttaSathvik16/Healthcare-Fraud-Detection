import pandas as pd

from healthcare_fraud_detection.preprocessing.cleaners import coerce_numeric, drop_duplicate_claims
from healthcare_fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

# Extend with domain-specific columns as your schema stabilizes.
_DEFAULT_NUMERIC = ("billed_amount", "claim_count", "patient_age")


def preprocess_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical cleaning steps applied after extract and before feature engineering."""
    out = drop_duplicate_claims(df)
    numeric_cols = [c for c in _DEFAULT_NUMERIC if c in out.columns]
    if numeric_cols:
        out = coerce_numeric(out, numeric_cols)
    logger.debug("Preprocessed shape: %s", out.shape)
    return out
