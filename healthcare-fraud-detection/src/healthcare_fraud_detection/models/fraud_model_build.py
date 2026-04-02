"""
Train Logistic Regression and Random Forest on the feature dataset, evaluate on a holdout,
and persist both estimators plus the fitted preprocessor to ``models/fraud_model.pkl``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from healthcare_fraud_detection.features.engineering import build_feature_matrix
from healthcare_fraud_detection.utils.config import Settings, get_settings
from healthcare_fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

FRAUD_MODEL_FILENAME = "fraud_model.pkl"


def load_feature_dataset(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Match ``train.py``: date features and drop linkage IDs."""
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


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_and_evaluate_fraud_models(
    df: pd.DataFrame,
    *,
    target_col: str = "fraud",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column {target_col!r}")

    work = prepare_training_frame(df)
    if target_col not in work.columns:
        raise ValueError(f"Target {target_col!r} dropped during preparation")

    y = work[target_col].astype(int).to_numpy()
    stratify = y if np.unique(y).size > 1 and np.min(np.bincount(y)) >= 2 else None

    train_df, test_df = train_test_split(
        work,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    fm_train, preprocessor = build_feature_matrix(
        train_df,
        target_col=target_col,
        fit_preprocessor=True,
    )
    fm_test, _ = build_feature_matrix(
        test_df,
        target_col=target_col,
        fit_preprocessor=False,
        preprocessor=preprocessor,
    )
    assert fm_train.target is not None and fm_test.target is not None

    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
        solver="lbfgs",
    )
    lr.fit(fm_train.X, fm_train.target)

    rf = RandomForestClassifier(
        n_estimators=150,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(fm_train.X, fm_train.target)

    y_test = fm_test.target
    pred_lr = lr.predict(fm_test.X)
    pred_rf = rf.predict(fm_test.X)

    metrics = {
        "logistic_regression": _classification_metrics(y_test, pred_lr),
        "random_forest": _classification_metrics(y_test, pred_rf),
    }

    bundle: dict[str, Any] = {
        "version": 1,
        "target_col": target_col,
        "preprocessor": preprocessor,
        "logistic_regression": lr,
        "random_forest": rf,
        "evaluation": {
            "test_size": test_size,
            "random_state": random_state,
            "metrics": metrics,
            "feature_names": fm_train.feature_names,
        },
    }
    return bundle, metrics


def save_fraud_model_bundle(
    bundle: dict[str, Any],
    path: Path | str | None = None,
    *,
    settings: Settings | None = None,
) -> Path:
    cfg = settings or get_settings()
    out = Path(path) if path else (cfg.project_root / "models" / FRAUD_MODEL_FILENAME)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    logger.info("Saved fraud model bundle to %s", out)
    return out


def build_fraud_detection_model(
    feature_path: Path | str | None = None,
    *,
    output_path: Path | str | None = None,
    target_col: str = "fraud",
    test_size: float = 0.2,
    random_state: int = 42,
    settings: Settings | None = None,
) -> tuple[dict[str, Any], Path]:
    cfg = settings or get_settings()
    csv_path = (
        Path(feature_path)
        if feature_path is not None
        else cfg.resolved_processed_dir() / "features.csv"
    )
    df = load_feature_dataset(csv_path)
    bundle, metrics = train_and_evaluate_fraud_models(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )
    for name, m in metrics.items():
        logger.info(
            "%s — accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
            name,
            m["accuracy"],
            m["precision"],
            m["recall"],
            m["f1"],
        )
    out = save_fraud_model_bundle(bundle, path=output_path, settings=cfg)
    return bundle, out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="features.csv path")
    parser.add_argument("--output", type=Path, default=None, help="Path for fraud_model.pkl")
    parser.add_argument("--target", type=str, default="fraud")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    build_fraud_detection_model(
        feature_path=args.input,
        output_path=args.output,
        target_col=args.target,
        test_size=args.test_size,
        random_state=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
