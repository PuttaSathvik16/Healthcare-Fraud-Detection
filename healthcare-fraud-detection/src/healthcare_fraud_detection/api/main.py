import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException

from healthcare_fraud_detection.api.claims import claims_to_dataframe
from healthcare_fraud_detection.api.schemas import (
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from healthcare_fraud_detection.models.inference import (
    FraudPredictor,
    load_predictor_from_disk,
    top_global_feature_importances,
)
from healthcare_fraud_detection.utils.config import get_settings
from healthcare_fraud_detection.utils.explanations import generate_fraud_explanation
from healthcare_fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

_predictor: FraudPredictor | None = None
_bundle_raw: dict | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _predictor, _bundle_raw
    try:
        _predictor, _bundle_raw = load_predictor_from_disk()
        logger.info("Fraud model loaded for API inference")
    except FileNotFoundError:
        _predictor = None
        _bundle_raw = None
        logger.warning("No trained model found; /predict and /explain will return 503")
    yield


app = FastAPI(
    title="Healthcare Fraud Detection API",
    version="0.2.0",
    description="Fraud risk scores and optional OpenAI explanations for healthcare claims.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(model_loaded=_predictor is not None)


def _require_model() -> FraudPredictor:
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train and save models/fraud_model.pkl or models/artifacts/model.joblib.",
        )
    return _predictor


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    pred = _require_model()
    df = claims_to_dataframe(body.claims)
    scores = pred.predict_proba(df)
    return PredictResponse(fraud_score=[float(s) for s in scores])


@app.post("/explain", response_model=ExplainResponse)
def explain(body: ExplainRequest) -> ExplainResponse:
    pred = _require_model()
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not set; cannot generate explanations.",
        )

    df = claims_to_dataframe(body.claims)
    scores = pred.predict_proba(df)
    highlights = top_global_feature_importances(_bundle_raw, k=8)
    model_name = body.openai_model or "gpt-4o-mini"

    explanations: list[str] = []
    for i, claim in enumerate(body.claims):
        text = generate_fraud_explanation(
            float(scores[i]),
            claim.model_dump(),
            highlights,
            model=model_name,
        )
        explanations.append(text)

    return ExplainResponse(
        fraud_score=[float(s) for s in scores],
        explanation=explanations,
    )


def run_server() -> None:
    settings = get_settings()
    uvicorn.run(
        "healthcare_fraud_detection.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
