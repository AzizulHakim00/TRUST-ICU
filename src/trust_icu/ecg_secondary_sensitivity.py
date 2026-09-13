"""Secondary certification-envelope sensitivity analyses for TRUST-ECG v0.4.

This module operates only on aggregate Phase-0 metric payloads. It never
changes the frozen primary gate or model; alternative envelopes are explicitly
secondary sensitivity analyses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import product
from typing import Any

_STATUS_ORDER = (
    "certified",
    "calibration_recovery_candidate",
    "discrimination_failure",
    "insufficient_support",
)


@dataclass(frozen=True)
class EnvelopeDefinition:
    """One secondary representation of the TRUST-ECG certification gate."""

    minimum_positives: int = 50
    minimum_negatives: int = 50
    minimum_pr_auc_to_prevalence_ratio: float = 2.0
    maximum_absolute_slope_deviation: float = 0.35
    maximum_absolute_intercept: float = 0.75
    require_positive_brier_skill: bool = True
    use_discrimination: bool = True
    use_slope: bool = True
    use_intercept: bool = True
    use_brier_skill: bool = True

    def __post_init__(self) -> None:
        if self.minimum_positives < 0 or self.minimum_negatives < 0:
            raise ValueError("Support thresholds must be non-negative.")
        if self.minimum_pr_auc_to_prevalence_ratio <= 0:
            raise ValueError("PR-AUC/prevalence threshold must be positive.")
        if self.maximum_absolute_slope_deviation < 0:
            raise ValueError("Slope-deviation threshold cannot be negative.")
        if self.maximum_absolute_intercept < 0:
            raise ValueError("Intercept threshold cannot be negative.")


def _finite_float(payload: Mapping[str, Any], key: str) -> float:
    value = float(payload[key])
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"Metric {key} must be finite.")
    return value


def classify_metric_payload(
    metrics: Mapping[str, Any] | None,
    envelope: EnvelopeDefinition | None = None,
) -> str:
    """Classify one aggregate pair under a secondary envelope definition."""

    gate = envelope or EnvelopeDefinition()
    if metrics is None:
        return "insufficient_support"

    positives = int(metrics["positives"])
    negatives = int(metrics["negatives"])
    if positives < gate.minimum_positives or negatives < gate.minimum_negatives:
        return "insufficient_support"

    ratio = _finite_float(metrics, "pr_auc_to_prevalence_ratio")
    if gate.use_discrimination and ratio < gate.minimum_pr_auc_to_prevalence_ratio:
        return "discrimination_failure"

    calibration_failed = False
    if gate.use_slope:
        slope = _finite_float(metrics, "calibration_slope")
        calibration_failed |= abs(slope - 1.0) > gate.maximum_absolute_slope_deviation
    if gate.use_intercept:
        intercept = _finite_float(metrics, "calibration_intercept")
        calibration_failed |= abs(intercept) > gate.maximum_absolute_intercept
    if gate.use_brier_skill and gate.require_positive_brier_skill:
        skill = _finite_float(metrics, "brier_skill_vs_prevalence")
        calibration_failed |= skill <= 0.0

    return "calibration_recovery_candidate" if calibration_failed else "certified"


def _iter_pairs(phase0_report: Mapping[str, Any]):
    external = phase0_report.get("external_certification")
    if not isinstance(external, Mapping) or not external:
        raise ValueError("Phase-0 report lacks external_certification metrics.")
    for source in sorted(external):
        source_payload = external[source]
        if not isinstance(source_payload, Mapping):
            raise ValueError(f"Invalid external payload for {source}.")
        for label_code in sorted(source_payload):
            pair = source_payload[label_code]
            if not isinstance(pair, Mapping):
                raise ValueError(f"Invalid pair payload for {source}/{label_code}.")
            yield str(source), str(label_code), pair


def _status_counts(
    phase0_report: Mapping[str, Any],
    envelope: EnvelopeDefinition,
) -> dict[str, int]:
    counts = {status: 0 for status in _STATUS_ORDER}
    for _, _, pair in _iter_pairs(phase0_report):
        metrics = pair.get("metrics")
        if metrics is not None and not isinstance(metrics, Mapping):
            raise ValueError("Pair metrics must be an object or null.")
        status = classify_metric_payload(metrics, envelope)
        counts[status] += 1
    return counts


def envelope_sensitivity_rows(
    phase0_report: Mapping[str, Any],
    *,
    pr_ratio_thresholds: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0),
    slope_deviations: tuple[float, ...] = (0.25, 0.35, 0.50),
    intercept_limits: tuple[float, ...] = (0.50, 0.75, 1.00),
    brier_requirements: tuple[bool, ...] = (True, False),
) -> list[dict[str, Any]]:
    """Return the predeclared 72-setting certification-envelope grid."""

    rows: list[dict[str, Any]] = []
    for ratio, slope, intercept, require_brier in product(
        pr_ratio_thresholds,
        slope_deviations,
        intercept_limits,
        brier_requirements,
    ):
        envelope = EnvelopeDefinition(
            minimum_pr_auc_to_prevalence_ratio=float(ratio),
            maximum_absolute_slope_deviation=float(slope),
            maximum_absolute_intercept=float(intercept),
            require_positive_brier_skill=bool(require_brier),
        )
        counts = _status_counts(phase0_report, envelope)
        rows.append(
            {
                "minimum_pr_auc_to_prevalence_ratio": float(ratio),
                "maximum_absolute_slope_deviation": float(slope),
                "maximum_absolute_intercept": float(intercept),
                "require_positive_brier_skill": bool(require_brier),
                "is_official_envelope": envelope == EnvelopeDefinition(),
                **counts,
            }
        )
    return rows


def framework_gate_ablation_rows(
    phase0_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Ablate evaluation-gate components without altering the frozen ResNet."""

    official = EnvelopeDefinition()
    variants = {
        "official_full_gate": official,
        "discrimination_only": replace(
            official,
            use_slope=False,
            use_intercept=False,
            use_brier_skill=False,
        ),
        "calibration_only": replace(official, use_discrimination=False),
        "minus_slope_criterion": replace(official, use_slope=False),
        "minus_intercept_criterion": replace(official, use_intercept=False),
        "minus_brier_skill_criterion": replace(official, use_brier_skill=False),
        "support_25_per_class": replace(
            official,
            minimum_positives=25,
            minimum_negatives=25,
        ),
        "support_100_per_class": replace(
            official,
            minimum_positives=100,
            minimum_negatives=100,
        ),
    }
    rows: list[dict[str, Any]] = []
    for name, envelope in variants.items():
        rows.append({"ablation": name, **_status_counts(phase0_report, envelope)})
    return rows
