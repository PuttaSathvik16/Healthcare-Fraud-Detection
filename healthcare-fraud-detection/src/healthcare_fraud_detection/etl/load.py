from pathlib import Path

import pandas as pd

from healthcare_fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def write_processed_table(
    df: pd.DataFrame,
    path: Path | str,
    *,
    format: str = "parquet",
) -> Path:
    """Persist a processed DataFrame for downstream training and serving."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = format.lower()
    if fmt == "parquet":
        df.to_parquet(out, index=False)
    elif fmt == "csv":
        df.to_csv(out, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    logger.info("Wrote %s rows to %s", len(df), out)
    return out
