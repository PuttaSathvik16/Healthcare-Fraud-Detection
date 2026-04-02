from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from healthcare_fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureMatrix:
    """Training or inference matrix with aligned feature names."""

    X: np.ndarray
    feature_names: list[str]
    target: np.ndarray | None = None


def _default_numeric(df: pd.DataFrame, *, exclude: str) -> list[str]:
    return [
        c
        for c in df.columns
        if c != exclude and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]


def _default_categorical(df: pd.DataFrame, *, exclude: str) -> list[str]:
    return [
        c
        for c in df.columns
        if c != exclude
        and (
            pd.api.types.is_object_dtype(df[c])
            or pd.api.types.is_string_dtype(df[c])
            or pd.api.types.is_categorical_dtype(df[c])
        )
    ]


def build_feature_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    target_col: str = "is_fraud",
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    fit_preprocessor: bool = True,
    preprocessor: ColumnTransformer | None = None,
) -> tuple[FeatureMatrix, ColumnTransformer | None]:
    """
    Build X (and y if target present). When fit_preprocessor is True, fits a new
    ColumnTransformer; otherwise expects a fitted preprocessor for inference.
    """
    num = numeric_cols if numeric_cols is not None else _default_numeric(df, exclude=target_col)
    cat = categorical_cols if categorical_cols is not None else _default_categorical(df, exclude=target_col)
    y: np.ndarray | None = None
    work = df.copy()
    if target_col in work.columns:
        y = work[target_col].astype(int).to_numpy()
        work = work.drop(columns=[target_col])

    num = [c for c in num if c in work.columns]
    cat = [c for c in cat if c in work.columns]

    pre = preprocessor or build_feature_preprocessor(num, cat)
    if fit_preprocessor:
        X = pre.fit_transform(work)
        feature_names = list(pre.get_feature_names_out())
        logger.info("Fitted feature matrix: %s features", len(feature_names))
        return FeatureMatrix(X=np.asarray(X), feature_names=feature_names, target=y), pre

    if preprocessor is None:
        raise ValueError("preprocessor required when fit_preprocessor is False")
    X = pre.transform(work)
    feature_names = list(pre.get_feature_names_out())
    return FeatureMatrix(X=np.asarray(X), feature_names=feature_names, target=y), None
