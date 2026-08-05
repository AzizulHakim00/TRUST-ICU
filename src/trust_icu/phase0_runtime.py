"""Hardened runtime bindings for the locked Phase 0 orchestration.

This module keeps the public orchestration contract stable while replacing three internal
operations with scalable and alignment-safe implementations before execution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import trust_icu.phase0_runner as _runner


def _cluster_bootstrap(
    y_true: pd.Series,
    probabilities: np.ndarray,
    hospitals: pd.Series,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[_runner.BootstrapInterval, ...]:
    """Estimate external uncertainty by resampling hospitals without refitting calibration."""

    if iterations <= 0:
        return ()
    frame = pd.DataFrame(
        {
            "y": pd.to_numeric(y_true, errors="raise").astype(int).to_numpy(),
            "p": np.asarray(probabilities, dtype=float),
            "hospital": hospitals.astype("string").to_numpy(),
        }
    )
    if frame["hospital"].isna().any():
        raise ValueError("Cluster bootstrap requires non-missing hospital identifiers.")
    grouped = {str(name): group.index.to_numpy() for name, group in frame.groupby("hospital")}
    names = np.asarray(list(grouped), dtype=object)
    if len(names) < 2:
        raise ValueError("Cluster bootstrap requires at least two external hospitals.")

    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"pr_auc": [], "roc_auc": [], "brier": []}
    for _ in range(iterations):
        sampled = rng.choice(names, size=len(names), replace=True)
        indices = np.concatenate([grouped[str(name)] for name in sampled])
        y_sample = frame.loc[indices, "y"]
        if y_sample.nunique() != 2:
            continue
        p_sample = frame.loc[indices, "p"].to_numpy()
        values["pr_auc"].append(float(average_precision_score(y_sample, p_sample)))
        values["roc_auc"].append(float(roc_auc_score(y_sample, p_sample)))
        values["brier"].append(float(brier_score_loss(y_sample, p_sample)))

    alpha = (1.0 - confidence_level) / 2.0
    estimates = {
        "pr_auc": float(average_precision_score(frame["y"], frame["p"])),
        "roc_auc": float(roc_auc_score(frame["y"], frame["p"])),
        "brier": float(brier_score_loss(frame["y"], frame["p"])),
    }
    intervals: list[_runner.BootstrapInterval] = []
    for metric, samples in values.items():
        if not samples:
            raise RuntimeError(f"No valid hospital-bootstrap samples for {metric}.")
        intervals.append(
            _runner.BootstrapInterval(
                metric=metric,
                estimate=estimates[metric],
                lower=float(np.quantile(samples, alpha)),
                upper=float(np.quantile(samples, 1.0 - alpha)),
                successful_iterations=len(samples),
                requested_iterations=iterations,
            )
        )
    return tuple(intervals)


def _hospital_robustness(
    y_true: pd.Series,
    probabilities: np.ndarray,
    hospitals: pd.Series,
    *,
    minimum_rows: int,
    minimum_events: int,
) -> _runner.HospitalRobustness:
    """Summarize site performance without fitting calibration models per hospital."""

    frame = pd.DataFrame(
        {
            "y": pd.to_numeric(y_true, errors="raise").astype(int).to_numpy(),
            "p": np.asarray(probabilities, dtype=float),
            "hospital": hospitals.astype("string").to_numpy(),
        }
    )
    if frame["hospital"].isna().any():
        raise ValueError("Hospital robustness requires non-missing hospital identifiers.")
    metric_pairs: list[tuple[float, float]] = []
    for _, group in frame.groupby("hospital"):
        events = int(group["y"].sum())
        negatives = len(group) - events
        if len(group) < minimum_rows or events < minimum_events or negatives < minimum_events:
            continue
        metric_pairs.append(
            (
                float(average_precision_score(group["y"], group["p"])),
                float(brier_score_loss(group["y"], group["p"])),
            )
        )
    if metric_pairs:
        pr_values = np.asarray([item[0] for item in metric_pairs])
        brier_values = np.asarray([item[1] for item in metric_pairs])
        median_pr = float(np.median(pr_values))
        tenth_pr = float(np.quantile(pr_values, 0.10))
        worst_pr = float(np.min(pr_values))
        median_brier = float(np.median(brier_values))
        worst_brier = float(np.max(brier_values))
    else:
        median_pr = tenth_pr = worst_pr = median_brier = worst_brier = None
    return _runner.HospitalRobustness(
        total_hospitals=int(frame["hospital"].nunique()),
        eligible_metric_hospitals=len(metric_pairs),
        minimum_rows=minimum_rows,
        minimum_events=minimum_events,
        median_pr_auc=median_pr,
        tenth_percentile_pr_auc=tenth_pr,
        worst_pr_auc=worst_pr,
        median_brier=median_brier,
        worst_brier=worst_brier,
    )


def _task_matrices(
    cohort: pd.DataFrame,
    events: pd.DataFrame,
    features: pd.DataFrame,
    task: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Join labels and features one-to-one by stay ID so row order cannot misalign."""

    labeled = _runner.assign_task_labels(cohort, events, task)
    eligibility = f"{task}__primary_analysis_eligible"
    target = f"{task}__target"
    labeled = labeled.loc[labeled[eligibility].astype(bool)].copy()
    metadata_columns = ["stay_id", "patient_id", "icu_admit_time", target]
    if "hospital_id" in labeled:
        metadata_columns.append("hospital_id")
    combined = labeled[metadata_columns].merge(
        features,
        on="stay_id",
        how="inner",
        sort=False,
        validate="one_to_one",
    )
    if len(combined) != len(labeled):
        raise RuntimeError("Feature and label rows are not one-to-one after task filtering.")
    metadata = combined[metadata_columns].copy()
    feature_columns = [column for column in features.columns if column != "stay_id"]
    matrix = combined[feature_columns + [target]].copy()
    return matrix, metadata, target


def install_runtime_hardening() -> None:
    """Bind hardened internal operations into the orchestration module."""

    _runner._cluster_bootstrap = _cluster_bootstrap
    _runner._hospital_robustness = _hospital_robustness
    _runner._task_matrices = _task_matrices


install_runtime_hardening()

BootstrapInterval = _runner.BootstrapInterval
HospitalRobustness = _runner.HospitalRobustness
MissingnessShift = _runner.MissingnessShift
ModelEvaluation = _runner.ModelEvaluation
Phase0BaselineReport = _runner.Phase0BaselineReport
TaskPhase0Report = _runner.TaskPhase0Report
TemporalSplitReport = _runner.TemporalSplitReport
VerifiedCanonicalRun = _runner.VerifiedCanonicalRun
build_feature_matrix_from_extract = _runner.build_feature_matrix_from_extract
build_phase0_dry_run_plan = _runner.build_phase0_dry_run_plan
execute_phase0_baselines = _runner.execute_phase0_baselines
run_task_phase0 = _runner.run_task_phase0
temporal_patient_purged_split = _runner.temporal_patient_purged_split
verify_credentialed_run = _runner.verify_credentialed_run

__all__ = [
    "BootstrapInterval",
    "HospitalRobustness",
    "MissingnessShift",
    "ModelEvaluation",
    "Phase0BaselineReport",
    "TaskPhase0Report",
    "TemporalSplitReport",
    "VerifiedCanonicalRun",
    "build_feature_matrix_from_extract",
    "build_phase0_dry_run_plan",
    "execute_phase0_baselines",
    "install_runtime_hardening",
    "run_task_phase0",
    "temporal_patient_purged_split",
    "verify_credentialed_run",
]
