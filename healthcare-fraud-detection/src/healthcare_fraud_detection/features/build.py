"""Build ML features from processed claims and write ``data/processed/features.csv``."""

from __future__ import annotations

import sys

from healthcare_fraud_detection.features.fraud_features import main


if __name__ == "__main__":
    sys.exit(main())
