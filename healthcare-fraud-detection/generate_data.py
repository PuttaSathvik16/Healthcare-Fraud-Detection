#!/usr/bin/env python3
"""
Generate synthetic healthcare claims for local development and demos.

Writes CSV or Parquet under data/raw/ by default. Columns match the package
preprocessing defaults and API ClaimRecord fields, including is_fraud for training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROVIDER_TYPES = ("FACILITY", "PHYSICIAN", "DME", "LAB", "HOME_HEALTH")
DIAGNOSIS_GROUPS = ("GRP_A", "GRP_B", "GRP_C", "GRP_D", "GRP_E")


def generate_claims(
    n_rows: int,
    *,
    fraud_rate: float,
    seed: int | None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    is_fraud = rng.random(n_rows) < fraud_rate

    billed_amount = rng.lognormal(mean=4.8, sigma=0.85, size=n_rows) * 80.0
    claim_count = rng.poisson(2.0, size=n_rows).astype(float) + 1.0
    patient_age = rng.integers(18, 92, size=n_rows).astype(float)

    fraud_idx = np.nonzero(is_fraud)[0]
    billed_amount[fraud_idx] *= rng.uniform(1.6, 3.2, size=len(fraud_idx))
    claim_count[fraud_idx] = np.minimum(
        claim_count[fraud_idx] + rng.poisson(6.0, size=len(fraud_idx)),
        40.0,
    )

    provider_type = rng.choice(np.array(PROVIDER_TYPES, dtype=object), size=n_rows)
    diagnosis_group = rng.choice(np.array(DIAGNOSIS_GROUPS, dtype=object), size=n_rows)

    claim_id = np.array([f"CLM-{i:08d}" for i in range(n_rows)], dtype=object)

    df = pd.DataFrame(
        {
            "claim_id": claim_id,
            "billed_amount": np.round(billed_amount, 2),
            "claim_count": claim_count,
            "patient_age": patient_age,
            "provider_type": provider_type,
            "diagnosis_group": diagnosis_group,
            "is_fraud": is_fraud.astype(np.int8),
        }
    )
    return df


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    default_out = root / "data" / "raw" / "synthetic_claims.csv"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=5_000, help="Number of claim rows")
    p.add_argument(
        "--fraud-rate",
        type=float,
        default=0.12,
        help="Approximate fraction of rows labeled is_fraud=1",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    p.add_argument(
        "--output",
        type=Path,
        default=default_out,
        help=f"Output file path (default: {default_out})",
    )
    p.add_argument(
        "--format",
        choices=("csv", "parquet"),
        default=None,
        help="Override format from extension; default: infer from --output suffix",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.fraud_rate <= 1.0):
        raise SystemExit("--fraud-rate must be between 0 and 1")
    if args.rows < 1:
        raise SystemExit("--rows must be at least 1")

    fmt = args.format
    if fmt is None:
        suf = args.output.suffix.lower()
        if suf == ".csv":
            fmt = "csv"
        elif suf in (".parquet", ".pq"):
            fmt = "parquet"
        else:
            fmt = "csv"
            args.output = args.output.with_suffix(".csv")

    df = generate_claims(args.rows, fraud_rate=args.fraud_rate, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        df.to_csv(args.output, index=False)
    else:
        df.to_parquet(args.output, index=False)

    fraud_n = int(df["is_fraud"].sum())
    print(f"Wrote {len(df)} rows ({fraud_n} fraud) to {args.output}")


if __name__ == "__main__":
    main()
