from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool


class ClaimInput(BaseModel):
    """
    Claim row aligned with ``features.csv`` / training pipeline (after ETL + feature engineering).

    Send ``claim_date`` as an ISO date; the API maps it to ``claim_month`` and ``claim_dow``.
    """

    procedure_code: str
    claim_amount: float = Field(ge=0)
    claim_date: str
    age: int = Field(ge=0, le=120)
    gender: str
    chronic_condition: str = "none"
    specialization: str
    region: str
    avg_claim_amount_per_provider: float = 0.0
    total_claims_per_provider: int = 0
    total_claims_per_patient: int = 0
    claim_amount_deviation_from_provider_avg: float = 0.0
    claim_amount_zscore: float = 0.0
    claims_last_7_days: int = 0
    claims_last_30_days: int = 0
    is_duplicate_claim: int = Field(0, ge=0, le=1)


class PredictRequest(BaseModel):
    claims: list[ClaimInput] = Field(min_length=1)


class PredictResponse(BaseModel):
    fraud_score: list[float]


class ExplainRequest(BaseModel):
    claims: list[ClaimInput] = Field(min_length=1)
    openai_model: str | None = None


class ExplainResponse(BaseModel):
    fraud_score: list[float]
    explanation: list[str]
