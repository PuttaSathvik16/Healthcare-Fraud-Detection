from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from healthcare_fraud_detection.etl.cleaning import (
    clean_claims_missing,
    clean_patients_missing,
    clean_providers_missing,
    dedupe_claims,
    dedupe_patients,
    dedupe_providers,
)
from healthcare_fraud_detection.etl.extract import read_claims_table
from healthcare_fraud_detection.etl.integrate import enforce_processed_schema, join_claims_with_dimensions
from healthcare_fraud_detection.etl.load import write_processed_table
from healthcare_fraud_detection.etl.sources import load_raw_fraud_tables
from healthcare_fraud_detection.preprocessing.pipeline import preprocess_claims
from healthcare_fraud_detection.utils.config import Settings, get_settings
from healthcare_fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

PROCESSED_CLAIMS_CSV = "claims_processed.csv"


def run_etl(
    source_path: Path | str,
    output_path: Path | str | None = None,
    *,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Extract raw claims, preprocess, and write to the processed data area."""
    cfg = settings or get_settings()
    raw = Path(source_path)
    out = Path(output_path) if output_path else cfg.resolved_processed_dir() / "claims_clean.parquet"
    df = read_claims_table(raw)
    df = preprocess_claims(df)
    write_processed_table(df, out, format="parquet")
    logger.info("ETL complete: %s -> %s", raw, out)
    return df


def run_fraud_detection_etl(
    raw_dir: Path | str | None = None,
    output_path: Path | str | None = None,
    *,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """
    Load patients, providers, and claims from ``data/raw``, clean, dedupe,
    join, enforce dtypes, and write ``data/processed/claims_processed.csv``.
    """
    cfg = settings or get_settings()
    out = Path(output_path) if output_path else cfg.resolved_processed_dir() / PROCESSED_CLAIMS_CSV

    patients, providers, claims = load_raw_fraud_tables(raw_dir, settings=cfg)
    patients = clean_patients_missing(patients)
    providers = clean_providers_missing(providers)
    claims = clean_claims_missing(claims)

    patients = dedupe_patients(patients)
    providers = dedupe_providers(providers)
    claims = dedupe_claims(claims)

    merged = join_claims_with_dimensions(claims, patients, providers)
    merged = enforce_processed_schema(merged)

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    logger.info("Healthcare fraud ETL complete: %s rows -> %s", len(merged), out)
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load patients/providers/claims from data/raw, merge, write claims_processed.csv."
    )
    parser.add_argument("--raw-dir", type=Path, default=None, help="Override raw data directory")
    parser.add_argument("--output", type=Path, default=None, help="Override output CSV path")
    args = parser.parse_args(argv)
    run_fraud_detection_etl(raw_dir=args.raw_dir, output_path=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
