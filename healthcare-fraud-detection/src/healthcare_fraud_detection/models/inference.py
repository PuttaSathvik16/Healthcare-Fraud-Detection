from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from healthcare_fraud_detection.features.engineering import build_feature_matrix
from healthcare_fraud_detection.utils.config import Settings, get_settings
from healthcare_fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


class FraudPredictor:
    """Loads a trained sklearn model + preprocessor and scores new claim rows."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        self._model = bundle["model"]
        self._pre: ColumnTransformer = bundle["preprocessor"]
        self._target_col: str = bundle.get("target_col", "is_fraud")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        work = df.drop(columns=[self._target_col], errors="ignore")
        fm, _ = build_feature_matrix(
            work,
            target_col=self._target_col,
            fit_preprocessor=False,
            preprocessor=self._pre,
        )
        return np.asarray(self._model.predict_proba(fm.X)[:, 1])


def load_predictor(path: Path | str | None = None, *, settings: Settings | None = None) -> FraudPredictor:
    cfg = settings or get_settings()
    p = Path(path) if path else cfg.resolved_model_path()
    if not p.exists():
        logger.warning("Model artifact not found at %s", p)
        raise FileNotFoundError(p)
    bundle = joblib.load(p)
    return FraudPredictor(bundle)


def load_predictor_from_disk(
    *,
    settings: Settings | None = None,
) -> tuple[FraudPredictor, dict[str, Any] | None]:
    """
    Prefer ``models/fraud_model.pkl`` (RF bundle from training pipeline); else ``model.joblib``.

    Returns the predictor and the raw joblib dict (for explanation metadata), or ``None`` if
    the legacy artifact has no usable metadata.
    """
    cfg = settings or get_settings()
    fraud_path = cfg.resolved_fraud_bundle_path()
    if fraud_path.exists():
        raw: dict[str, Any] = joblib.load(fraud_path)
        bundle = {
            "model": raw["random_forest"],
            "preprocessor": raw["preprocessor"],
            "target_col": raw.get("target_col", "fraud"),
        }
        return FraudPredictor(bundle), raw

    legacy = cfg.resolved_model_path()
    if legacy.exists():
        raw = joblib.load(legacy)
        return FraudPredictor(raw), raw if isinstance(raw, dict) else None

    logger.warning("No model at %s or %s", fraud_path, legacy)
    raise FileNotFoundError(fraud_path)


def top_global_feature_importances(
    raw_bundle: dict[str, Any] | None,
    *,
    k: int = 8,
) -> list[tuple[str, float]]:
    """RF-based global importances for LLM context (not row-specific SHAP)."""
    if not raw_bundle:
        return []
    model = raw_bundle.get("random_forest") or raw_bundle.get("model")
    if model is None or not hasattr(model, "feature_importances_"):
        return []
    names: list[str]
    if "evaluation" in raw_bundle and "feature_names" in raw_bundle["evaluation"]:
        names = list(raw_bundle["evaluation"]["feature_names"])
    else:
        names = list(raw_bundle.get("feature_names", []))
    imp = np.asarray(model.feature_importances_)
    if len(names) != len(imp):
        return []
    idx = np.argsort(imp)[::-1][:k]
    return [(names[i], float(imp[i])) for i in idx]
