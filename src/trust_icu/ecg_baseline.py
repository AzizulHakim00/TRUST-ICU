"""Locked low-capacity baseline, calibration, and transportability metrics for TRUST-ECG."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from trust_icu.ecg_data import EXPECTED_LEADS

FEATURE_STATISTICS = (
    "mean",
    "std",
    "minimum",
    "maximum",
    "median",
    "q05",
    "q25",
    "q75",
    "q95",
    "rms",
    "mean_absolute_first_difference",
    "std_first_difference",
)


@dataclass(frozen=True)
class LogisticReferenceModel:
    scaler: StandardScaler
    estimators: tuple[LogisticRegression, ...]
    label_codes: tuple[str, ...]
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class PlattCalibrator:
    estimators: tuple[LogisticRegression, ...]
    label_codes: tuple[str, ...]


@dataclass(frozen=True)
class BinaryProbabilityMetrics:
    n: int
    positives: int
    negatives: int
    prevalence: float
    pr_auc: float
    pr_auc_to_prevalence_ratio: float
    roc_auc: float
    brier: float
    brier_skill_vs_prevalence: float
    calibration_slope: float
    calibration_intercept: float


@dataclass(frozen=True)
class PairCertification:
    status: Literal[
        "certified",
        "calibration_recovery_candidate",
        "discrimination_failure",
        "insufficient_support",
    ]
    metrics: BinaryProbabilityMetrics | None
    reasons: tuple[str, ...]


def feature_names() -> tuple[str, ...]:
    names = tuple(f"{lead}__{stat}" for lead in EXPECTED_LEADS for stat in FEATURE_STATISTICS)
    if len(names) != 144:
        raise RuntimeError("Locked handcrafted ECG feature contract must contain exactly 144 features.")
    return names


def extract_handcrafted_features(waveform_mv: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Extract the locked 12 per-lead summary statistics from physical-mV ECG data."""

    waveform = np.asarray(waveform_mv, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if waveform.ndim != 2 or waveform.shape[0] != len(EXPECTED_LEADS):
        raise ValueError("Waveform must have shape (12, time).")
    if mask.ndim != 1 or mask.shape[0] != waveform.shape[1]:
        raise ValueError("Validity mask must match waveform time dimension.")
    if int(mask.sum()) < 2:
        raise ValueError("At least two valid samples are required for handcrafted ECG features.")
    valid = waveform[:, mask]
    if not np.isfinite(valid).all():
        raise ValueError("Handcrafted ECG features require finite physical-mV samples.")

    rows: list[float] = []
    for lead_values in valid:
        differences = np.diff(lead_values)
        rows.extend(
            [
                float(np.mean(lead_values)),
                float(np.std(lead_values, ddof=0)),
                float(np.min(lead_values)),
                float(np.max(lead_values)),
                float(np.median(lead_values)),
                float(np.quantile(lead_values, 0.05)),
                float(np.quantile(lead_values, 0.25)),
                float(np.quantile(lead_values, 0.75)),
                float(np.quantile(lead_values, 0.95)),
                float(np.sqrt(np.mean(np.square(lead_values)))),
                float(np.mean(np.abs(differences))),
                float(np.std(differences, ddof=0)),
            ]
        )
    features = np.asarray(rows, dtype=np.float64)
    if features.shape != (144,) or not np.isfinite(features).all():
        raise RuntimeError("Handcrafted ECG feature extraction violated the locked 144-feature contract.")
    return features


def external_partition(
    *,
    source: str,
    record_id: str,
    seed: int = 20260808,
    certification_fraction: float = 0.60,
) -> Literal["certification", "recovery_pool"]:
    """Assign an external record without using diagnosis labels or model output."""

    if not source or not record_id:
        raise ValueError("Source and record_id are required for deterministic external partitioning.")
    if not 0.0 < certification_fraction < 1.0:
        raise ValueError("certification_fraction must be strictly between 0 and 1.")
    material = f"{seed}|{source}|{record_id}".encode()
    integer = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    uniform = integer / 2**64
    return "certification" if uniform < certification_fraction else "recovery_pool"


def fit_logistic_reference(
    X: np.ndarray,
    y: np.ndarray,
    *,
    label_codes: tuple[str, ...],
) -> LogisticReferenceModel:
    """Fit the fixed independent binary Logistic Regression reference on development folds only."""

    features = np.asarray(X, dtype=np.float64)
    targets = np.asarray(y, dtype=np.int64)
    if features.ndim != 2 or features.shape[1] != 144:
        raise ValueError("Logistic reference requires an (n, 144) feature matrix.")
    if targets.ndim != 2 or targets.shape[0] != features.shape[0]:
        raise ValueError("Multi-label target matrix must align with feature rows.")
    if targets.shape[1] != len(label_codes):
        raise ValueError("label_codes length must match target columns.")
    if not np.isfinite(features).all() or not np.isin(targets, [0, 1]).all():
        raise ValueError("Logistic reference inputs must be finite features and binary targets.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    estimators: list[LogisticRegression] = []
    for index in range(targets.shape[1]):
        column = targets[:, index]
        if np.unique(column).size != 2:
            raise ValueError(f"Label {label_codes[index]} lacks both classes in model-fitting data.")
        estimator = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=20260808,
        )
        estimator.fit(scaled, column)
        estimators.append(estimator)
    return LogisticReferenceModel(
        scaler=scaler,
        estimators=tuple(estimators),
        label_codes=label_codes,
        feature_names=feature_names(),
    )


def logistic_decision_scores(model: LogisticReferenceModel, X: np.ndarray) -> np.ndarray:
    features = np.asarray(X, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 144:
        raise ValueError("Logistic inference requires an (n, 144) feature matrix.")
    scaled = model.scaler.transform(features)
    return np.column_stack([estimator.decision_function(scaled) for estimator in model.estimators])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_platt_calibrators(
    raw_scores: np.ndarray,
    y: np.ndarray,
    *,
    label_codes: tuple[str, ...],
) -> PlattCalibrator:
    """Fit independent unregularized scalar Platt maps on the locked calibration fold only."""

    scores = np.asarray(raw_scores, dtype=np.float64)
    targets = np.asarray(y, dtype=np.int64)
    if scores.ndim != 2 or targets.shape != scores.shape or scores.shape[1] != len(label_codes):
        raise ValueError("Calibration scores, targets, and label codes must align.")
    if not np.isfinite(scores).all() or not np.isin(targets, [0, 1]).all():
        raise ValueError("Calibration inputs must be finite scores and binary targets.")
    estimators: list[LogisticRegression] = []
    for index, code in enumerate(label_codes):
        column = targets[:, index]
        if np.unique(column).size != 2:
            raise ValueError(f"Label {code} lacks both classes in calibration data.")
        estimator = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=2000,
            random_state=20260808,
        )
        estimator.fit(scores[:, [index]], column)
        estimators.append(estimator)
    return PlattCalibrator(estimators=tuple(estimators), label_codes=label_codes)


def apply_platt_calibrators(calibrator: PlattCalibrator, raw_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(calibrator.estimators):
        raise ValueError("Raw score matrix does not match the Platt calibrator.")
    return np.column_stack(
        [
            estimator.predict_proba(scores[:, [index]])[:, 1]
            for index, estimator in enumerate(calibrator.estimators)
        ]
    )


def _calibration_slope_intercept(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))[:, None]
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    model.fit(logits, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def evaluate_binary_probabilities(y: np.ndarray, probabilities: np.ndarray) -> BinaryProbabilityMetrics:
    targets = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 1 or probs.shape != targets.shape:
        raise ValueError("Binary targets and probabilities must be aligned one-dimensional arrays.")
    if not np.isin(targets, [0, 1]).all() or not np.isfinite(probs).all():
        raise ValueError("Metrics require binary targets and finite probabilities.")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1].")
    positives = int(targets.sum())
    negatives = int(targets.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Binary probability metrics require both outcome classes.")
    prevalence = positives / targets.size
    pr_auc = float(average_precision_score(targets, probs))
    roc_auc = float(roc_auc_score(targets, probs))
    brier = float(brier_score_loss(targets, probs))
    prevalence_brier = prevalence * (1.0 - prevalence)
    brier_skill = 1.0 - brier / prevalence_brier
    slope, intercept = _calibration_slope_intercept(targets, probs)
    return BinaryProbabilityMetrics(
        n=int(targets.size),
        positives=positives,
        negatives=negatives,
        prevalence=float(prevalence),
        pr_auc=pr_auc,
        pr_auc_to_prevalence_ratio=float(pr_auc / prevalence),
        roc_auc=roc_auc,
        brier=brier,
        brier_skill_vs_prevalence=float(brier_skill),
        calibration_slope=slope,
        calibration_intercept=intercept,
    )


def certify_label_domain_pair(
    y: np.ndarray,
    probabilities: np.ndarray,
    *,
    minimum_positives: int = 50,
    minimum_negatives: int = 50,
    minimum_pr_auc_to_prevalence_ratio: float = 2.0,
    maximum_absolute_slope_deviation: float = 0.35,
    maximum_absolute_intercept: float = 0.75,
) -> PairCertification:
    """Apply the prospective research envelope to one label x external-domain pair."""

    targets = np.asarray(y, dtype=np.int64)
    positives = int(targets.sum()) if targets.ndim == 1 else 0
    negatives = int(targets.size - positives) if targets.ndim == 1 else 0
    if targets.ndim != 1 or positives < minimum_positives or negatives < minimum_negatives:
        return PairCertification(
            status="insufficient_support",
            metrics=None,
            reasons=("certification_partition_support_below_threshold",),
        )

    metrics = evaluate_binary_probabilities(targets, probabilities)
    if metrics.pr_auc_to_prevalence_ratio < minimum_pr_auc_to_prevalence_ratio:
        return PairCertification(
            status="discrimination_failure",
            metrics=metrics,
            reasons=("pr_auc_to_prevalence_ratio_below_viability_threshold",),
        )

    calibration_failures: list[str] = []
    if abs(metrics.calibration_slope - 1.0) > maximum_absolute_slope_deviation:
        calibration_failures.append("calibration_slope_outside_envelope")
    if abs(metrics.calibration_intercept) > maximum_absolute_intercept:
        calibration_failures.append("calibration_intercept_outside_envelope")
    if metrics.brier_skill_vs_prevalence <= 0.0:
        calibration_failures.append("nonpositive_brier_skill_vs_prevalence")
    if calibration_failures:
        return PairCertification(
            status="calibration_recovery_candidate",
            metrics=metrics,
            reasons=tuple(calibration_failures),
        )
    return PairCertification(status="certified", metrics=metrics, reasons=())


def raw_scores_to_probabilities(raw_scores: np.ndarray) -> np.ndarray:
    """Convert fixed model logits/decision scores to uncalibrated probabilities."""

    scores = np.asarray(raw_scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Raw scores must be finite.")
    return _sigmoid(scores)
