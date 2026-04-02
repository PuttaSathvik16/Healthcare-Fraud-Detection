"""
Train fraud detection models (logistic regression + random forest), evaluate, and save
``models/fraud_model.pkl``. See ``fraud_model_build`` for implementation and CLI flags.
"""

from __future__ import annotations

import sys

from healthcare_fraud_detection.models.fraud_model_build import main


if __name__ == "__main__":
    sys.exit(main())
