#!/usr/bin/env python3
"""
Explain the fraud model bundle (``models/fraud_model.pkl``):

- Random Forest: feature importances (Gini-based)
- Logistic Regression: coefficients (higher → more fraud risk, given standardized inputs)

Optional: ``--permutation`` runs sklearn permutation importance on a sample (slower).

From the project root::

    python explain.py
    python explain.py --top 15 --output reports/feature_explanation.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _bootstrap_import_path() -> Path:
    root = Path(__file__).resolve().parent
    src = root / "src"
    for p in (str(src), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def _format_table(rows: list[tuple[str, float]], width: int = 48, signed: bool = False) -> str:
    lines = [f"{'feature':<{width}} weight"]
    lines.append("-" * (width + 12))
    fmt = "+.6f" if signed else ".6f"
    for name, val in rows:
        lines.append(f"{str(name)[:width]:<{width}} {val:{fmt}}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Joblib bundle (default: models/fraud_model.pkl under project root)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="features.csv for permutation importance (default: data/processed/features.csv)",
    )
    parser.add_argument("--top", type=int, default=20, help="How many features to show per section")
    parser.add_argument(
        "--permutation",
        action="store_true",
        help="Run permutation importance (needs --data; samples up to 3000 rows)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write report to this file (directories created as needed)",
    )
    parser.add_argument("--sample", type=int, default=3000, help="Max rows for permutation importance")
    args = parser.parse_args(argv)

    root = _bootstrap_import_path()
    import joblib
    from sklearn.inspection import permutation_importance

    from healthcare_fraud_detection.features.engineering import build_feature_matrix
    from healthcare_fraud_detection.models.fraud_model_build import (
        FRAUD_MODEL_FILENAME,
        prepare_training_frame,
    )
    from healthcare_fraud_detection.utils.config import get_settings

    cfg = get_settings()
    model_path = args.model if args.model else (cfg.project_root / "models" / FRAUD_MODEL_FILENAME)
    if not model_path.is_file():
        raise SystemExit(f"Model bundle not found: {model_path.resolve()}\nTrain first: python -m healthcare_fraud_detection.models.train")

    bundle = joblib.load(model_path)
    target_col = bundle.get("target_col", "fraud")
    names: list[str] = list(bundle["evaluation"]["feature_names"])
    rf = bundle["random_forest"]
    lr = bundle["logistic_regression"]
    pre = bundle["preprocessor"]

    chunks: list[str] = []
    chunks.append(f"Bundle: {model_path.resolve()}")
    chunks.append(f"Target column: {target_col}")
    chunks.append("")

    imp = np.asarray(rf.feature_importances_)
    order = np.argsort(imp)[::-1][: args.top]
    rf_rows = [(names[i], float(imp[i])) for i in order]
    chunks.append("Random Forest — feature importances (higher = more splits on this feature)")
    chunks.append(_format_table(rf_rows, signed=False))
    chunks.append("")

    coef = np.ravel(np.asarray(lr.coef_))
    if len(coef) != len(names):
        chunks.append(f"Warning: LR coef length {len(coef)} != feature name count {len(names)}; skipping coefficients.")
    else:
        pos = np.argsort(coef)[::-1][: args.top]
        neg = np.argsort(coef)[: args.top]
        chunks.append("Logistic Regression — top coefficients toward fraud (positive)")
        chunks.append(_format_table([(names[i], float(coef[i])) for i in pos], signed=True))
        chunks.append("")
        chunks.append("Logistic Regression — top coefficients away from fraud (negative)")
        chunks.append(_format_table([(names[i], float(coef[i])) for i in neg], signed=True))
        chunks.append("")

    if args.permutation:
        data_path = (
            args.data if args.data else (cfg.resolved_processed_dir() / "features.csv")
        )
        if not data_path.is_file():
            raise SystemExit(f"--permutation requires data at {data_path}")
        import pandas as pd

        df = prepare_training_frame(pd.read_csv(data_path))
        if target_col not in df.columns:
            raise SystemExit(f"Target {target_col!r} missing from {data_path}")
        if len(df) > args.sample:
            df = df.sample(n=args.sample, random_state=42)
        fm, _ = build_feature_matrix(
            df,
            target_col=target_col,
            fit_preprocessor=False,
            preprocessor=pre,
        )
        assert fm.target is not None
        X = fm.X
        y = fm.target
        fnames = list(fm.feature_names)
        for label, est in (("Random Forest", rf), ("Logistic Regression", lr)):
            pi = permutation_importance(est, X, y, n_repeats=5, random_state=42, n_jobs=-1)
            mean = pi.importances_mean
            idx = np.argsort(mean)[::-1][: args.top]
            rows = [(fnames[i], float(mean[i])) for i in idx]
            chunks.append(f"{label} — permutation importance (drop in score when shuffled)")
            chunks.append(_format_table(rows, signed=False))
            chunks.append("")

    report = "\n".join(chunks).rstrip() + "\n"
    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote {args.output.resolve()}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
