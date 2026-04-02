from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_healthcare.constants import (
    BURST_EXTRA_CLAIMS_PER_PATIENT,
    CLAIM_DATE_SPAN_DAYS,
    DEFAULT_N_CLAIMS,
    DUPLICATE_INJECTION_RATE,
    HIGH_AMOUNT_INJECTION_RATE,
    HIGH_AMOUNT_MULTIPLIER_MAX,
    HIGH_AMOUNT_MULTIPLIER_MIN,
    N_BURST_PATIENTS,
    PROCEDURE_PROFILES,
)


def _procedure_sampler(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    codes = np.array([p[0] for p in PROCEDURE_PROFILES], dtype=object)
    medians = {p[0]: float(p[1]) for p in PROCEDURE_PROFILES}
    weights = np.array([p[2] for p in PROCEDURE_PROFILES], dtype=float)
    weights /= weights.sum()
    return codes, medians, weights


def _sample_claim_amount(
    rng: np.random.Generator,
    procedure_code: str,
    code_to_median: dict[str, float],
) -> float:
    median = code_to_median[procedure_code]
    noise = rng.lognormal(mean=np.log(max(median, 1.0)), sigma=0.35)
    adj = rng.uniform(0.75, 1.35)
    return float(np.round(max(5.0, noise * adj), 2))


def generate_base_claims(
    patients: pd.DataFrame,
    providers: pd.DataFrame,
    n_claims: int,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    codes, code_to_median, weights = _procedure_sampler(rng)

    patient_ids = patients["patient_id"].to_numpy()
    provider_ids = providers["provider_id"].to_numpy()
    proc_sample = rng.choice(codes, size=n_claims, p=weights)

    claim_ids = np.array([f"CLM-{i:08d}" for i in range(1, n_claims + 1)], dtype=object)
    p_idx = rng.choice(len(patient_ids), size=n_claims, replace=True)
    v_idx = rng.choice(len(provider_ids), size=n_claims, replace=True)

    anchor = pd.Timestamp("2023-01-01")
    offsets = rng.integers(0, CLAIM_DATE_SPAN_DAYS + 1, size=n_claims)
    claim_dates = anchor + pd.to_timedelta(offsets, unit="D")

    amounts = np.array(
        [_sample_claim_amount(rng, pc, code_to_median) for pc in proc_sample],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "claim_id": claim_ids,
            "patient_id": patient_ids[p_idx],
            "provider_id": provider_ids[v_idx],
            "procedure_code": proc_sample,
            "claim_amount": amounts,
            "claim_date": claim_dates,
        }
    )


def inject_high_amount_claims(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = max(1, int(len(df) * HIGH_AMOUNT_INJECTION_RATE))
    idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
    out = df.copy()
    mult = rng.uniform(HIGH_AMOUNT_MULTIPLIER_MIN, HIGH_AMOUNT_MULTIPLIER_MAX, size=n)
    out.loc[idx, "claim_amount"] = np.round(out.loc[idx, "claim_amount"].to_numpy() * mult, 2)
    return out


def inject_duplicate_claim_rows(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = max(1, int(len(df) * DUPLICATE_INJECTION_RATE))
    pick = df.loc[rng.choice(df.index.to_numpy(), size=n, replace=False)].copy()
    start = int(df["claim_id"].str.replace("CLM-", "", regex=False).astype(int).max()) + 1
    new_ids = np.array([f"CLM-{i:08d}" for i in range(start, start + len(pick))], dtype=object)
    pick["claim_id"] = new_ids
    return pd.concat([df, pick], ignore_index=True)


def inject_burst_claims(
    df: pd.DataFrame,
    patients: pd.DataFrame,
    providers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    burst_patients = rng.choice(patients["patient_id"].to_numpy(), size=N_BURST_PATIENTS, replace=False)
    provider_pool = providers["provider_id"].to_numpy()
    codes, code_to_median, weights = _procedure_sampler(rng)
    start = int(df["claim_id"].str.replace("CLM-", "", regex=False).astype(int).max()) + 1
    rows: list[dict[str, object]] = []
    cid = start
    anchor = pd.Timestamp("2023-01-01")
    for pid in burst_patients:
        base_day = int(rng.integers(40, CLAIM_DATE_SPAN_DAYS - 20))
        for _ in range(BURST_EXTRA_CLAIMS_PER_PATIENT):
            day_off = int(rng.integers(0, 6))
            proc = str(rng.choice(codes, p=weights))
            med = code_to_median[proc]
            amt = float(np.round(max(15.0, rng.lognormal(np.log(med), 0.25)), 2))
            rows.append(
                {
                    "claim_id": f"CLM-{cid:08d}",
                    "patient_id": pid,
                    "provider_id": str(rng.choice(provider_pool)),
                    "procedure_code": proc,
                    "claim_amount": amt,
                    "claim_date": anchor + pd.Timedelta(days=base_day + day_off),
                }
            )
            cid += 1
    extra = pd.DataFrame(rows)
    return pd.concat([df, extra], ignore_index=True)


def build_claims_table(
    patients: pd.DataFrame,
    providers: pd.DataFrame,
    n_claims: int = DEFAULT_N_CLAIMS,
    *,
    seed: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (all_claims, baseline_claims).

    baseline_claims are used to compute amount thresholds so injected outliers
    are not absorbed into the reference distribution.
    """
    rng = np.random.default_rng(seed)
    baseline = generate_base_claims(patients, providers, n_claims, rng=rng)
    stretched = inject_high_amount_claims(baseline, rng)
    with_dupes = inject_duplicate_claim_rows(stretched, rng)
    full = inject_burst_claims(with_dupes, patients, providers, rng)
    full["claim_date"] = pd.to_datetime(full["claim_date"]).dt.normalize()
    baseline["claim_date"] = pd.to_datetime(baseline["claim_date"]).dt.normalize()
    return full, baseline
