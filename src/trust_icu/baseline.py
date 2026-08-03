"""Locked Phase 0 baseline utilities for synthetic and approved clinical matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class BaselineMetrics:
    n: int
    events: int
    prevalence: float
    pr_auc: float
    pr_auc_prevalence_ratio: float
    roc_auc: float
    brier: float
    calibration_slope: float
    calibration_intercept: float


def assert_matrix_is_leakage_safe(
    frame: pd.DataFrame,
    *,
    target_column: str,
    prohibited_tokens: tuple[str, ...] = (
        "target",
        "outcome",
        "death",
        "discharge",
        "followup",
        "future",
        "post_index",
    ),
    prohibited_exact: tuple[str, ...] = (
        "patient_id",
        "hospital_admission_id",
        "stay_id",
        "hospital_id",
        "site_id",
        "dataset_id",
    ),
) -> list[str]:
    """Return predictor columns after blocking identifiers and future/outcome metadata."""

    if target_column not in frame:
        raise ValueError(f"Target column not found: {target_column}")
    candidates = [column for column in frame.columns if column != target_column]
    blocked = [
        column
        for column in candidates
        if column.lower() in prohibited_exact
        or any(token in column.lower() for token in prohibited_tokens)
    ]
    if blocked:
        raise ValueError(f"Prohibited predictor columns detected: {sorted(blocked)}")
    return candidates


def _calibration_model(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logits, y_true)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def evaluate_probabilities(y_true: pd.Series, probabilities: np.ndarray) -> BaselineMetrics:
    """Evaluate discrimination and calibration using prespecified Phase 0 metrics."""

    y = pd.to_numeric(y_true, errors="raise").astype(int)
    if y.nunique() != 2:
        raise ValueError("Evaluation target must contain two classes.")
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(y),):
        raise ValueError("Probabilities must be a one-dimensional array matching y_true.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and between zero and one.")
    prevalence = float(y.mean())
    pr_auc = float(average_precision_score(y, probabilities))
    slope, intercept = _calibration_model(y, probabilities)
    return BaselineMetrics(
        n=len(y),
        events=int(y.sum()),
        prevalence=prevalence,
        pr_auc=pr_auc,
        pr_auc_prevalence_ratio=pr_auc / prevalence,
        roc_auc=float(roc_auc_score(y, probabilities)),
        brier=float(brier_score_loss(y, probabilities)),
        calibration_slope=slope,
        calibration_intercept=intercept,
    )


def _split_predictor_types(frame: pd.DataFrame, predictors: list[str]) -> tuple[list[str], list[str]]:
    numeric = [column for column in predictors if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in predictors if column not in numeric]
    return numeric, categorical


def fit_logistic_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str,
    data_classification: Literal["synthetic", "credentialed_locked"],
    random_state: int = 20260804,
) -> tuple[Pipeline, np.ndarray, BaselineMetrics]:
    """Fit the prespecified regularized logistic-regression feasibility baseline."""

    if data_classification not in {"synthetic", "credentialed_locked"}:
        raise RuntimeError("Training is allowed only for synthetic or credentialed_locked data.")
    predictors = assert_matrix_is_leakage_safe(train, target_column=target_column)
    if set(test.columns) != set(train.columns):
        raise ValueError("Train and test matrices must contain identical columns.")
    y_train = pd.to_numeric(train[target_column], errors="raise").astype(int)
    y_test = pd.to_numeric(test[target_column], errors="raise").astype(int)
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Both train and test targets must contain two classes.")

    numeric, categorical = _split_predictor_types(train, predictors)
    transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        [
            ("preprocess", transformer),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )
    pipeline.fit(train[predictors], y_train)
    probabilities = pipeline.predict_proba(test[predictors])[:, 1]
    return pipeline, probabilities, evaluate_probabilities(y_test, probabilities)


def fit_catboost_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str,
    data_classification: Literal["synthetic", "credentialed_locked"],
    random_state: int = 20260804,
) -> tuple[Any, np.ndarray, BaselineMetrics]:
    """Fit the fixed CatBoost feasibility baseline without using external labels for tuning."""

    if data_classification not in {"synthetic", "credentialed_locked"}:
        raise RuntimeError("Training is allowed only for synthetic or credentialed_locked data.")
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:  # pragma: no cover - exercised only in the ML environment
        raise RuntimeError('CatBoost is not installed. Install trust-icu with the "ml" extra.') from exc

    predictors = assert_matrix_is_leakage_safe(train, target_column=target_column)
    if set(test.columns) != set(train.columns):
        raise ValueError("Train and test matrices must contain identical columns.")
    y_train = pd.to_numeric(train[target_column], errors="raise").astype(int)
    y_test = pd.to_numeric(test[target_column], errors="raise").astype(int)
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Both train and test targets must contain two classes.")

    x_train = train[predictors].copy()
    x_test = test[predictors].copy()
    _, categorical = _split_predictor_types(x_train, predictors)
    for column in categorical:
        x_train[column] = x_train[column].astype("string").fillna("__MISSING__")
        x_test[column] = x_test[column].astype("string").fillna("__MISSING__")

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        random_seed=random_state,
        random_strength=1.0,
        l2_leaf_reg=5.0,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )
    model.fit(x_train, y_train, cat_features=categorical)
    probabilities = model.predict_proba(x_test)[:, 1]
    return model, probabilities, evaluate_probabilities(y_test, probabilities)
