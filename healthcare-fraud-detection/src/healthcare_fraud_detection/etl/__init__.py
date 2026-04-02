from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from healthcare_fraud_detection.etl.extract import read_claims_table
from healthcare_fraud_detection.etl.load import write_processed_table

__all__ = ["read_claims_table", "run_etl", "run_fraud_detection_etl", "write_processed_table"]

if TYPE_CHECKING:
    from healthcare_fraud_detection.etl.pipeline import run_etl as run_etl
    from healthcare_fraud_detection.etl.pipeline import run_fraud_detection_etl as run_fraud_detection_etl


def __getattr__(name: str) -> Any:
    if name in ("run_etl", "run_fraud_detection_etl"):
        mod = importlib.import_module("healthcare_fraud_detection.etl.pipeline")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
