from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from healthcare_fraud_detection.features.engineering import build_feature_matrix
from healthcare_fraud_detection.utils.config import Settings, get_settings
from healthcare_fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def train_and_save(
    df: pd.DataFrame,
    output_path: Path | str | None = None,
    *,
    target_col: str = "is_fraud",
    settings: Settings | None = None,
    model_params: dict[str, Any] | None = None,
) -> Path:
    """
    Fit preprocessor + classifier on labeled data and persist as a single joblib artifact.
    """
    cfg = settings or get_settings()
    out = Path(output_path) if output_path else cfg.resolved_model_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    fm, pre = build_feature_matrix(df, target_col=target_col, fit_preprocessor=True)
    if fm.target is None:
        raise ValueError(f"Training data must include '{target_col}'")
    params = model_params or {"n_estimators": 100, "random_state": 42, "class_weight": "balanced"}
    clf = RandomForestClassifier(**params)
    clf.fit(fm.X, fm.target)

    bundle: dict[str, Any] = {
        "model": clf,
        "preprocessor": pre,
        "target_col": target_col,
        "feature_names": fm.feature_names,
    }
    joblib.dump(bundle, out)
    logger.info("Saved model bundle to %s", out)
    return out
