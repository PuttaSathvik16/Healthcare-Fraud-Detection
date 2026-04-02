from __future__ import annotations

import importlib
from typing import Any

from healthcare_fraud_detection.features.engineering import FeatureMatrix, build_feature_matrix

__all__ = [
    "FEATURES_FILENAME",
    "FeatureMatrix",
    "build_and_save_features",
    "build_fraud_feature_table",
    "build_feature_matrix",
    "load_processed_claims",
    "ml_ready_frame",
    "save_fraud_features",
]


def __getattr__(name: str) -> Any:
    if name in (
        "FEATURES_FILENAME",
        "build_and_save_features",
        "build_fraud_feature_table",
        "load_processed_claims",
        "ml_ready_frame",
        "save_fraud_features",
    ):
        mod = importlib.import_module("healthcare_fraud_detection.features.fraud_features")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
