#!/usr/bin/env python3
"""
Generate three linked synthetic datasets (patients, providers, claims) with
rule-based fraud labels on claims.

Run from the project root:
    python generate_synthetic_healthcare_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from synthetic_healthcare.claims import build_claims_table
from synthetic_healthcare.constants import (
    DEFAULT_N_CLAIMS,
    DEFAULT_N_PATIENTS,
    DEFAULT_N_PROVIDERS,
    DEFAULT_SEED,
)
from synthetic_healthcare.fraud_labeling import label_fraud
from synthetic_healthcare.patients import generate_patients
from synthetic_healthcare.providers import generate_providers


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    default_out = root / "data" / "raw"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--patients", type=int, default=DEFAULT_N_PATIENTS)
    p.add_argument("--providers", type=int, default=DEFAULT_N_PROVIDERS)
    p.add_argument("--claims", type=int, default=DEFAULT_N_CLAIMS, help="Baseline claim rows before injections")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", type=Path, default=default_out)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.claims < 10_000:
        raise SystemExit("--claims must be at least 10,000")
    if args.patients < 10 or args.providers < 5:
        raise SystemExit("Need enough patients and providers for realistic sampling")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    patients = generate_patients(args.patients, seed=args.seed)
    providers = generate_providers(args.providers, seed=args.seed + 1)
    claims_raw, baseline = build_claims_table(
        patients,
        providers,
        n_claims=args.claims,
        seed=args.seed + 2,
    )
    claims = label_fraud(claims_raw, baseline)

    patients_path = args.output_dir / "patients.csv"
    providers_path = args.output_dir / "providers.csv"
    claims_path = args.output_dir / "claims.csv"

    patients.to_csv(patients_path, index=False)
    providers.to_csv(providers_path, index=False)
    claim_cols = [
        "claim_id",
        "patient_id",
        "provider_id",
        "procedure_code",
        "claim_amount",
        "claim_date",
        "fraud",
    ]
    claims[claim_cols].to_csv(claims_path, index=False)

    fraud_rate = float(claims["fraud"].mean())
    print(f"Wrote {len(patients)} rows -> {patients_path}")
    print(f"Wrote {len(providers)} rows -> {providers_path}")
    print(f"Wrote {len(claims)} rows -> {claims_path} (fraud rate {fraud_rate:.2%})")


if __name__ == "__main__":
    main()
