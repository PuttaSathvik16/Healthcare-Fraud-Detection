from __future__ import annotations

import pandas as pd

from healthcare_fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def _strip_ids(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = out[c].astype("string").str.strip()
    return out


def clean_patients_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = _strip_ids(df, ["patient_id"])
    before = len(out)
    out = out.dropna(subset=["patient_id"])
    out = out[out["patient_id"] != ""]
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out = out.dropna(subset=["age"])
    for col in ("gender", "chronic_condition"):
        if col not in out.columns:
            out[col] = "unknown"
        else:
            out[col] = out[col].fillna("unknown").astype("string").str.strip()
    dropped = before - len(out)
    if dropped:
        logger.info("Patients: dropped %s rows with missing id or age", dropped)
    return out.reset_index(drop=True)


def clean_providers_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = _strip_ids(df, ["provider_id"])
    before = len(out)
    out = out.dropna(subset=["provider_id"])
    out = out[out["provider_id"] != ""]
    for col in ("specialization", "region"):
        if col not in out.columns:
            out[col] = "unknown"
        else:
            out[col] = out[col].fillna("unknown").astype("string").str.strip()
    dropped = before - len(out)
    if dropped:
        logger.info("Providers: dropped %s rows with missing provider_id", dropped)
    return out.reset_index(drop=True)


def clean_claims_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = _strip_ids(df, ["claim_id", "patient_id", "provider_id"])
    before = len(out)
    out = out.dropna(subset=["claim_id", "patient_id", "provider_id"])
    for c in ("claim_id", "patient_id", "provider_id"):
        out = out[out[c] != ""]
    out["claim_date"] = pd.to_datetime(out["claim_date"], errors="coerce")
    out = out.dropna(subset=["claim_date"])
    out["claim_amount"] = pd.to_numeric(out["claim_amount"], errors="coerce")
    median_amt = float(out["claim_amount"].median()) if out["claim_amount"].notna().any() else 0.0
    out["claim_amount"] = out["claim_amount"].fillna(median_amt)
    if "procedure_code" not in out.columns:
        out["procedure_code"] = "UNKNOWN"
    else:
        pc = out["procedure_code"].fillna("UNKNOWN").astype("string").str.strip()
        out["procedure_code"] = pc.mask(pc.eq(""), "UNKNOWN")
    if "fraud" not in out.columns:
        logger.warning("claims: no fraud column; filling 0")
        out["fraud"] = 0
    else:
        out["fraud"] = pd.to_numeric(out["fraud"], errors="coerce").fillna(0)
    dropped = before - len(out)
    if dropped:
        logger.info("Claims: dropped %s rows with missing keys or dates", dropped)
    return out.reset_index(drop=True)


def dedupe_patients(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    out = df.drop_duplicates(subset=["patient_id"], keep="first").reset_index(drop=True)
    if len(out) < before:
        logger.info("Patients: removed %s duplicate patient_id rows", before - len(out))
    return out


def dedupe_providers(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    out = df.drop_duplicates(subset=["provider_id"], keep="first").reset_index(drop=True)
    if len(out) < before:
        logger.info("Providers: removed %s duplicate provider_id rows", before - len(out))
    return out


def dedupe_claims(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    out = df.drop_duplicates(subset=["claim_id"], keep="first").reset_index(drop=True)
    if len(out) < before:
        logger.info("Claims: removed %s duplicate claim_id rows", before - len(out))
    return out
