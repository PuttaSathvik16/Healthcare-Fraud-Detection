"""Realistic value sets and default generation parameters."""

from __future__ import annotations

GENDERS = ("M", "F", "NB", "U")
GENDER_WEIGHTS = (0.46, 0.46, 0.03, 0.05)

CHRONIC_CONDITIONS = (
    "none",
    "diabetes",
    "hypertension",
    "COPD",
    "asthma",
    "CHF",
    "CKD",
    "depression",
    "CAD",
    "osteoarthritis",
)

SPECIALIZATIONS = (
    "family_medicine",
    "internal_medicine",
    "cardiology",
    "orthopedics",
    "neurology",
    "dermatology",
    "psychiatry",
    "pediatrics",
    "general_surgery",
    "emergency_medicine",
    "radiology",
    "anesthesiology",
)

REGIONS = ("NORTHEAST", "SOUTH", "MIDWEST", "WEST", "PACIFIC")

# (procedure_code, typical_median_usd, relative_frequency)
PROCEDURE_PROFILES: tuple[tuple[str, float, float], ...] = (
    ("99213", 120.0, 0.18),
    ("99214", 165.0, 0.14),
    ("99215", 220.0, 0.06),
    ("99203", 145.0, 0.08),
    ("80053", 45.0, 0.07),
    ("85025", 38.0, 0.06),
    ("73721", 520.0, 0.05),
    ("70450", 890.0, 0.04),
    ("72148", 780.0, 0.04),
    ("45378", 1250.0, 0.03),
    ("66984", 3200.0, 0.02),
    ("27447", 18500.0, 0.02),
    ("J0897", 2100.0, 0.03),
    ("J1745", 4800.0, 0.02),
    ("36415", 18.0, 0.10),
    ("93000", 185.0, 0.06),
)

DEFAULT_SEED = 42
DEFAULT_N_PATIENTS = 4_000
DEFAULT_N_PROVIDERS = 350
DEFAULT_N_CLAIMS = 12_000
HIGH_AMOUNT_INJECTION_RATE = 0.035
DUPLICATE_INJECTION_RATE = 0.02
BURST_EXTRA_CLAIMS_PER_PATIENT = 8
N_BURST_PATIENTS = 45
HIGH_AMOUNT_MULTIPLIER_MIN = 4.0
HIGH_AMOUNT_MULTIPLIER_MAX = 18.0
CLAIM_DATE_SPAN_DAYS = 520
BURST_WINDOW_DAYS = 7
MAX_CLAIMS_PER_BURST_WINDOW = 5
HIGH_AMOUNT_QUANTILE_BASELINE = 0.99
HIGH_AMOUNT_RATIO_OVER_P99 = 1.12
