from __future__ import annotations

from pathlib import Path

import pandas as pd

from healthcare_fraud_detection.etl.extract import read_claims_table
from healthcare_fraud_detection.utils.config import Settings, get_settings
from healthcare_fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_PATIENTS = "patients.csv"
DEFAULT_PROVIDERS = "providers.csv"
DEFAULT_CLAIMS = "claims.csv"


def load_patients(path: Path | str) -> pd.DataFrame:
    df = read_claims_table(path)
    logger.info("Loaded patients: %s rows from %s", len(df), path)
    return df


def load_providers(path: Path | str) -> pd.DataFrame:
    df = read_claims_table(path)
    logger.info("Loaded providers: %s rows from %s", len(df), path)
    return df


def load_claims(path: Path | str) -> pd.DataFrame:
    df = read_claims_table(path)
    logger.info("Loaded claims: %s rows from %s", len(df), path)
    return df


def load_raw_fraud_tables(
    raw_dir: Path | str | None = None,
    *,
    settings: Settings | None = None,
    patients_file: str = DEFAULT_PATIENTS,
    providers_file: str = DEFAULT_PROVIDERS,
    claims_file: str = DEFAULT_CLAIMS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = settings or get_settings()
    base = Path(raw_dir) if raw_dir is not None else cfg.resolved_raw_dir()
    patients = load_patients(base / patients_file)
    providers = load_providers(base / providers_file)
    claims = load_claims(base / claims_file)
    return patients, providers, claims
