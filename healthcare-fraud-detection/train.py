#!/usr/bin/env python3
"""
Train fraud classifier from processed features (or merged claims) and save joblib bundle.

From the project root::

    python train.py
    python train.py --input data/processed/claims_processed.csv

Default input is ``data/processed/features.csv`` when present (drop-high-cardinality IDs
and expand dates are applied when those columns exist).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def _bootstrap_import_path() -> Path:
    root = Path(__file__).resolve().parent
    src = root / "src"
    for p in (str(src), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def _prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Avoid exploding ``claim_date`` / IDs into huge one-hot spaces for sklearn."""
    out = df.copy()
    if "claim_date" in out.columns:
        out["claim_date"] = pd.to_datetime(out["claim_date"], errors="coerce")
        out["claim_month"] = out["claim_date"].dt.month.fillna(0).astype("int64")
        out["claim_dow"] = out["claim_date"].dt.dayofweek.fillna(0).astype("int64")
        out = out.drop(columns=["claim_date"])
    for c in ("claim_id", "patient_id", "provider_id"):
        if c in out.columns:
            out = out.drop(columns=[c])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Training CSV (default: data/processed/features.csv, else claims_processed.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Joblib bundle path (default: models/artifacts/model.joblib from settings)",
    )
    parser.add_argument("--target", type=str, default="fraud", help="Label column name")
    args = parser.parse_args(argv)

    root = _bootstrap_import_path()
    from healthcare_fraud_detection.models.trainer import train_and_save
    from healthcare_fraud_detection.utils.config import get_settings

    cfg = get_settings()
    default_features = cfg.resolved_processed_dir() / "features.csv"
    default_claims = cfg.resolved_processed_dir() / "claims_processed.csv"
    in_path = args.input
    if in_path is None:
        in_path = default_features if default_features.exists() else default_claims

    if not in_path.is_file():
        raise SystemExit(f"Input not found: {in_path.resolve()}")

    prev = os.getcwd()
    try:
        os.chdir(root)
        raw = pd.read_csv(in_path)
        if args.target not in raw.columns:
            raise SystemExit(f"Column {args.target!r} missing from {in_path}")
        df = _prepare_training_frame(raw)
        out = train_and_save(df, output_path=args.output, target_col=args.target)
        print(f"Saved model -> {out.resolve()}")
    finally:
        os.chdir(prev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
