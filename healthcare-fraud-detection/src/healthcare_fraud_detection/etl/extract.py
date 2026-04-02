from pathlib import Path

import pandas as pd

from healthcare_fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def read_claims_table(
    path: Path | str,
    *,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load a tabular claims dataset from disk (CSV by default)."""
    p = Path(path)
    if not p.exists():
        logger.warning("Source file missing: %s", p)
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p, sep=sep, encoding=encoding)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported format: {suffix}")
