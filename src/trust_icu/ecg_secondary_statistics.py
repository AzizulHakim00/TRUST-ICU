"""Secondary uncertainty and estimability summaries for TRUST-ECG.

These helpers operate on frozen predictions or aggregate Phase-1 counts. They do not modify the
primary certification matrix and they treat non-estimable repeats as reported exclusions.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from trust_icu.ecg_baseline import evaluate_binary_probabilities
from trust_icu.ecg_secondary_sensitivity import SecondaryEnvelope, classify_pair_secondary
from trust_icu.ecg_statistical_core import (
    quantile_summary,
    stratified_binary_bootstrap_indices,
    wilson_interval,
)

_GATE_METRICS = (
    "pr_auc_to_prevalence_ratio",
    "calibration_slope",
    "calibration_intercept",
    "brier_skill_vs_prevalence",
)


def wilson_rate_interval(successes: int, total: int) -> tuple[float, float]:
    """Return a 95% Wilson interval after validating integer count semantics."""

    if not isinstance(successes, int) or not isinstance(total, int):
        raise ValueError("Wilson interval counts must be integers.")
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval counts must satisfy 0 <= successes <= total and total > 0.")
    return wilson_interval(successes, total)


def bootstrap_gate_uncertainty(
    y: np.ndarray,
    probabilities: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap uncertainty for the metrics underlying the official certification gate.

    Bootstrap samples are stratified by the binary outcome so calibration and discrimination
    metrics remain estimable. The returned gate-satisfaction rate is secondary uncertainty
    evidence; it never replaces the frozen point-estimate certification status.
    """

    targets = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 1 or probs.shape != targets.shape:
        raise ValueError("Bootstrap targets and probabilities must be aligned one-dimensional arrays.")
    if not np.isin(targets, [0, 1]).all() or not np.isfinite(probs).all():
        raise ValueError("Bootstrap inputs require binary targets and finite probabilities.")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("Bootstrap probabilities must lie in [0, 1].")
    if np.unique(targets).size != 2:
        raise ValueError("Bootstrap gate uncertainty requires both classes.")
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("Bootstrap repeats must be a positive integer.")
    if not isinstance(seed, int):
        raise ValueError("Bootstrap seed must be an integer.")

    values: dict[str, list[float]] = {name: [] for name in _GATE_METRICS}
    gate_satisfaction = 0
    envelope = SecondaryEnvelope.official()
    rng = np.random.default_rng(seed)

    for _ in range(repeats):
        indices = stratified_binary_bootstrap_indices(targets, rng)
        metrics = evaluate_binary_probabilities(targets[indices], probs[indices])
        payload = asdict(metrics)
        for name in _GATE_METRICS:
            values[name].append(float(payload[name]))
        pair = {"status": "secondary_bootstrap", "metrics": payload, "reasons": []}
        if classify_pair_secondary(pair, envelope) == "certified":
            gate_satisfaction += 1

    gate_low, gate_high = wilson_rate_interval(gate_satisfaction, repeats)
    return {
        "bootstrap_repeats_requested": repeats,
        "valid_bootstrap_repeats": repeats,
        "seed": seed,
        "gate_satisfaction_count": gate_satisfaction,
        "gate_satisfaction_rate": gate_satisfaction / repeats,
        "gate_satisfaction_wilson_low": gate_low,
        "gate_satisfaction_wilson_high": gate_high,
        "metric_intervals": {
            name: quantile_summary(values[name])
            for name in _GATE_METRICS
        },
    }


def summarize_phase1_estimability(
    *,
    repeats_requested: int,
    estimable_repeats: int,
    recovered_repeats: int,
    nonestimable_reasons: dict[str, int],
) -> dict[str, Any]:
    """Summarize Phase-1 repeat estimability without imputing excluded repeats."""

    counts = (repeats_requested, estimable_repeats, recovered_repeats)
    if any(not isinstance(value, int) for value in counts):
        raise ValueError("Phase-1 counts must be integers.")
    if repeats_requested <= 0:
        raise ValueError("Phase-1 counts require a positive requested-repeat count.")
    if not (0 <= recovered_repeats <= estimable_repeats <= repeats_requested):
        raise ValueError("Phase-1 counts are inconsistent.")
    if not isinstance(nonestimable_reasons, dict):
        raise ValueError("Phase-1 non-estimable reasons must be a count dictionary.")
    if any(
        not isinstance(reason, str)
        or not reason
        or not isinstance(count, int)
        or count < 0
        for reason, count in nonestimable_reasons.items()
    ):
        raise ValueError("Phase-1 non-estimable reason counts are invalid.")

    nonestimable = repeats_requested - estimable_repeats
    if sum(nonestimable_reasons.values()) != nonestimable:
        raise ValueError("Phase-1 counts do not match the non-estimable reason totals.")

    if estimable_repeats == 0:
        rate = None
        low = None
        high = None
    else:
        rate = recovered_repeats / estimable_repeats
        low, high = wilson_rate_interval(recovered_repeats, estimable_repeats)

    return {
        "repeats_requested": repeats_requested,
        "estimable_repeats": estimable_repeats,
        "nonestimable_repeats": nonestimable,
        "nonestimable_reasons": dict(sorted(nonestimable_reasons.items())),
        "recovered_repeats": recovered_repeats,
        "recovery_rate_among_estimable": rate,
        "recovery_rate_wilson_low": low,
        "recovery_rate_wilson_high": high,
    }
