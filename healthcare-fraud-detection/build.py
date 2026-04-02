#!/usr/bin/env python3
"""
Build pipeline for healthcare fraud detection.

From the project root (venv activated recommended)::

    python build.py              # ETL only → data/processed/claims_processed.csv
    python build.py --generate   # Synthetic CSVs in data/raw/, then ETL
    python build.py --train      # ETL then train model → models/artifacts/model.joblib

Uses ``healthcare_fraud_detection`` (install with ``pip install -e .`` or rely on ``src/`` on path).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_import_path() -> Path:
    root = Path(__file__).resolve().parent
    src = root / "src"
    for p in (str(src), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def _run_generate(root: Path) -> None:
    import generate_synthetic_healthcare_data as gen

    prev = Path.cwd()
    try:
        import os

        os.chdir(root)
        gen.main()
    finally:
        os.chdir(prev)


def _run_etl(root: Path) -> Path:
    from healthcare_fraud_detection.etl.pipeline import run_fraud_detection_etl

    prev = Path.cwd()
    try:
        import os

        os.chdir(root)
        df = run_fraud_detection_etl()
    finally:
        os.chdir(prev)
    out = root / "data" / "processed" / "claims_processed.csv"
    return out


def _prepare_training_frame(df):
    """Drop leaky / high-cardinality IDs; encode dates as small numeric features."""
    import pandas as pd

    out = df.copy()
    out["claim_date"] = pd.to_datetime(out["claim_date"], errors="coerce")
    out["claim_month"] = out["claim_date"].dt.month.fillna(0).astype("int64")
    out["claim_dow"] = out["claim_date"].dt.dayofweek.fillna(0).astype("int64")
    out = out.drop(columns=["claim_date"])
    drop_ids = ("claim_id", "patient_id", "provider_id")
    out = out.drop(columns=[c for c in drop_ids if c in out.columns])
    return out


def _run_train(processed_csv: Path, root: Path) -> Path:
    import pandas as pd

    from healthcare_fraud_detection.models.trainer import train_and_save

    if not processed_csv.exists():
        raise FileNotFoundError(f"Processed data not found: {processed_csv}")
    df = _prepare_training_frame(pd.read_csv(processed_csv))
    prev = Path.cwd()
    try:
        import os

        os.chdir(root)
        return train_and_save(df, target_col="fraud")
    finally:
        os.chdir(prev)


def main(argv: list[str] | None = None) -> int:
    root = _bootstrap_import_path()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Write synthetic patients/providers/claims to data/raw/ before ETL",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="After ETL, train RandomForest and save joblib bundle",
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=None,
        help="Path to claims_processed.csv for --train (default: data/processed/claims_processed.csv)",
    )
    args = parser.parse_args(argv)

    if args.generate:
        _run_generate(root)

    processed = (
        args.processed
        if args.processed is not None
        else root / "data" / "processed" / "claims_processed.csv"
    )

    if not args.train:
        _run_etl(root)
        print(f"ETL complete -> {processed.resolve()}")
        return 0

    _run_etl(root)
    artifact = _run_train(processed, root)
    print(f"ETL complete -> {processed.resolve()}")
    print(f"Model saved -> {artifact.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
