import pandas as pd

from healthcare_fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def drop_duplicate_claims(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Remove duplicate claim rows; default subset is all columns."""
    before = len(df)
    out = df.drop_duplicates(subset=subset, keep="first")
    dropped = before - len(out)
    if dropped:
        logger.info("Dropped %s duplicate rows", dropped)
    return out.reset_index(drop=True)


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
