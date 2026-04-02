from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from healthcare_fraud_detection.utils.config import Settings, get_settings
from healthcare_fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

FEATURES_FILENAME = "features.csv"

REQUIRED_FOR_FEATURES = frozenset(
    {"patient_id", "provider_id", "claim_amount", "claim_date", "procedure_code"}
)


def _validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_FOR_FEATURES - set(df.columns)
    if missing:
        raise ValueError(f"Processed claims frame missing columns: {sorted(missing)}")


def _add_provider_patient_aggregates(out: pd.DataFrame) -> pd.DataFrame:
    g_prov = out.groupby("provider_id", observed=True)["claim_amount"]
    out["avg_claim_amount_per_provider"] = g_prov.transform("mean")
    out["total_claims_per_provider"] = g_prov.transform("count")
    g_pat = out.groupby("patient_id", observed=True)["claim_amount"]
    out["total_claims_per_patient"] = g_pat.transform("count")
    return out


def _add_amount_anomalies(out: pd.DataFrame) -> pd.DataFrame:
    out["claim_amount_deviation_from_provider_avg"] = (
        out["claim_amount"] - out["avg_claim_amount_per_provider"]
    )
    mean_p = out.groupby("provider_id", observed=True)["claim_amount"].transform("mean")
    std_p = out.groupby("provider_id", observed=True)["claim_amount"].transform("std")
    std_safe = std_p.replace(0.0, np.nan)
    z = (out["claim_amount"] - mean_p) / std_safe
    out["claim_amount_zscore"] = z.fillna(0.0).astype("float64")
    return out


def _add_patient_claim_windows(out: pd.DataFrame) -> pd.DataFrame:
    """Count claims in rolling 7- and 30-day windows (inclusive) per patient."""
    work = out[["patient_id", "claim_date"]].copy()
    work["_row_order"] = np.arange(len(work), dtype=np.int64)
    work["claim_date"] = pd.to_datetime(work["claim_date"]).dt.normalize()
    work = work.sort_values(["patient_id", "claim_date", "_row_order"])

    c7 = np.zeros(len(work), dtype=np.int32)
    c30 = np.zeros(len(work), dtype=np.int32)
    for _, grp in work.groupby("patient_id", sort=False):
        idx = grp.index.to_numpy()
        dts = grp["claim_date"].to_numpy(dtype="datetime64[ns]")
        ord_rows = grp["_row_order"].to_numpy()
        n = len(grp)
        left7 = left30 = 0
        for i in range(n):
            t = dts[i]
            while left7 <= i and (t - dts[left7]) / np.timedelta64(1, "D") > 6:
                left7 += 1
            while left30 <= i and (t - dts[left30]) / np.timedelta64(1, "D") > 29:
                left30 += 1
            pos = int(ord_rows[i])
            c7[pos] = i - left7 + 1
            c30[pos] = i - left30 + 1

    out = out.copy()
    out["claims_last_7_days"] = c7.astype("int64")
    out["claims_last_30_days"] = c30.astype("int64")
    return out


def _add_duplicate_flag(out: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(out["claim_date"], errors="coerce").dt.normalize()
    tmp = out.assign(_claim_date_key=dates)
    sub = [
        "patient_id",
        "provider_id",
        "procedure_code",
        "claim_amount",
        "_claim_date_key",
    ]
    sub = [c for c in sub if c in tmp.columns]
    flag = tmp.duplicated(subset=sub, keep=False).to_numpy()
    out = out.copy()
    out["is_duplicate_claim"] = flag.astype(np.int8)
    return out


def build_fraud_feature_table(processed_claims: pd.DataFrame) -> pd.DataFrame:
    """
    Engineering layer on top of ``claims_processed`` / merged claims.

    Preserves row order. Expects ``fraud`` for supervision when present.
    """
    _validate_input(processed_claims)
    out = processed_claims.copy()
    out["claim_date"] = pd.to_datetime(out["claim_date"], errors="coerce").dt.normalize()

    out = _add_provider_patient_aggregates(out)
    out = _add_amount_anomalies(out)
    out = _add_patient_claim_windows(out)
    out = _add_duplicate_flag(out)

    logger.info(
        "Built fraud feature table: %s rows, %s columns",
        len(out),
        len(out.columns),
    )
    return out


ML_LINKAGE_DROP = ("claim_id", "patient_id", "provider_id")


def ml_ready_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Strip high-cardinality linkage columns; keep engineered features, amounts, and demographics."""
    drop = [c for c in ML_LINKAGE_DROP if c in features.columns]
    return features.drop(columns=drop, errors="ignore")


def save_fraud_features(
    df: pd.DataFrame,
    path: Path | str | None = None,
    *,
    settings: Settings | None = None,
    ml_ready: bool = True,
) -> Path:
    cfg = settings or get_settings()
    out_path = Path(path) if path else cfg.resolved_processed_dir() / FEATURES_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_write = ml_ready_frame(df) if ml_ready else df
    to_write.to_csv(out_path, index=False)
    logger.info("Wrote %s rows to %s", len(to_write), out_path)
    return out_path


def build_and_save_features(
    processed_claims: pd.DataFrame,
    output_path: Path | str | None = None,
    *,
    settings: Settings | None = None,
    ml_ready: bool = True,
) -> tuple[pd.DataFrame, Path]:
    feats = build_fraud_feature_table(processed_claims)
    path = save_fraud_features(feats, path=output_path, settings=settings, ml_ready=ml_ready)
    return feats, path


def load_processed_claims(
    path: Path | str | None = None,
    *,
    settings: Settings | None = None,
) -> pd.DataFrame:
    cfg = settings or get_settings()
    csv_path = Path(path) if path else cfg.resolved_processed_dir() / "claims_processed.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return pd.read_csv(csv_path)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build fraud ML features from processed claims.")
    parser.add_argument("--input", type=Path, default=None, help="claims_processed.csv path")
    parser.add_argument("--output", type=Path, default=None, help="features.csv path")
    parser.add_argument(
        "--keep-claim-id",
        action="store_true",
        help="If set, do not strip claim_id when saving",
    )
    args = parser.parse_args(argv)
    df = load_processed_claims(path=args.input)
    build_and_save_features(
        df,
        output_path=args.output,
        ml_ready=not args.keep_claim_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
