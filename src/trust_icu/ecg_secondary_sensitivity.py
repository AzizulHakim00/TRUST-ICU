"""Secondary certification-envelope sensitivity for TRUST-ECG.

This module is deliberately separate from the frozen Phase-0 implementation. It reclassifies
already-computed aggregate label-domain metrics under explicitly secondary envelopes and never
changes the canonical primary certification report.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Literal

PairStatus = Literal[
    "certified",
    "calibration_recovery_candidate",
    "discrimination_failure",
    "insufficient_support",
]

_STATUSES: tuple[PairStatus, ...] = (
    "certified",
    "calibration_recovery_candidate",
    "discrimination_failure",
    "insufficient_support",
)


@dataclass(frozen=True)
class SecondaryEnvelope:
    """One explicitly secondary TRUST-ECG research envelope."""

    minimum_positives: int = 50
    minimum_negatives: int = 50
    pr_ratio_min: float = 2.0
    slope_deviation_max: float = 0.35
    intercept_abs_max: float = 0.75
    require_positive_brier_skill: bool = True
    require_discrimination: bool = True
    require_slope: bool = True
    require_intercept: bool = True

    @classmethod
    def official(cls) -> SecondaryEnvelope:
        """Return the frozen Phase-0 gate exactly, for reference-only reproduction."""

        return cls()


def _metrics(pair: dict[str, Any]) -> dict[str, Any] | None:
    metrics = pair.get("metrics")
    if metrics is None:
        return None
    if not isinstance(metrics, dict):
        raise ValueError("Phase-0 pair metrics must be a dictionary or null.")
    return metrics


def _number(metrics: dict[str, Any], key: str) -> float:
    try:
        value = float(metrics[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Phase-0 pair metrics are missing a numeric {key!r} value.") from exc
    return value


def classify_pair_secondary(pair: dict[str, Any], envelope: SecondaryEnvelope) -> PairStatus:
    """Reclassify one frozen Phase-0 pair under a secondary envelope.

    The function is pure and preserves the primary ordering of decisions: support first,
    discrimination second, then calibration. Pairs whose primary report intentionally omitted
    metrics for insufficient support remain insufficient rather than having support fabricated.
    """

    if not isinstance(pair, dict):
        raise ValueError("Phase-0 pair must be a dictionary.")
    metrics = _metrics(pair)
    if metrics is None:
        return "insufficient_support"

    positives = int(_number(metrics, "positives"))
    negatives = int(_number(metrics, "negatives"))
    if positives < envelope.minimum_positives or negatives < envelope.minimum_negatives:
        return "insufficient_support"

    if envelope.require_discrimination:
        ratio = _number(metrics, "pr_auc_to_prevalence_ratio")
        if ratio < envelope.pr_ratio_min:
            return "discrimination_failure"

    calibration_failed = False
    if envelope.require_slope:
        slope = _number(metrics, "calibration_slope")
        calibration_failed = calibration_failed or (
            abs(slope - 1.0) > envelope.slope_deviation_max
        )
    if envelope.require_intercept:
        intercept = _number(metrics, "calibration_intercept")
        calibration_failed = calibration_failed or (
            abs(intercept) > envelope.intercept_abs_max
        )
    if envelope.require_positive_brier_skill:
        brier_skill = _number(metrics, "brier_skill_vs_prevalence")
        calibration_failed = calibration_failed or brier_skill <= 0.0

    return "calibration_recovery_candidate" if calibration_failed else "certified"


def _count_matrix(
    matrix: dict[str, dict[str, dict[str, Any]]], envelope: SecondaryEnvelope
) -> dict[str, int]:
    counts = {status: 0 for status in _STATUSES}
    for source_pairs in matrix.values():
        if not isinstance(source_pairs, dict):
            raise ValueError("External certification matrix must map sources to label dictionaries.")
        for pair in source_pairs.values():
            status = classify_pair_secondary(pair, envelope)
            counts[status] += 1
    return counts


def evaluate_envelope_grid(
    matrix: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Evaluate the fixed secondary sensitivity grid over a frozen certification matrix."""

    official = SecondaryEnvelope.official()
    rows: list[dict[str, Any]] = []
    for pr_ratio, slope_deviation, intercept_abs, require_brier in product(
        (1.5, 2.0, 2.5, 3.0),
        (0.25, 0.35, 0.50),
        (0.50, 0.75, 1.00),
        (True, False),
    ):
        envelope = SecondaryEnvelope(
            pr_ratio_min=pr_ratio,
            slope_deviation_max=slope_deviation,
            intercept_abs_max=intercept_abs,
            require_positive_brier_skill=require_brier,
        )
        counts = _count_matrix(matrix, envelope)
        rows.append(
            {
                "pr_ratio_min": pr_ratio,
                "slope_deviation_max": slope_deviation,
                "intercept_abs_max": intercept_abs,
                "require_positive_brier_skill": require_brier,
                "is_official": envelope == official,
                **counts,
                "evaluable_pairs": (
                    counts["certified"]
                    + counts["calibration_recovery_candidate"]
                    + counts["discrimination_failure"]
                ),
            }
        )
    return rows


def evaluate_framework_ablations(
    matrix: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Evaluate gate-component ablations without changing model predictions."""

    official = SecondaryEnvelope.official()
    variants: tuple[tuple[str, SecondaryEnvelope], ...] = (
        ("full_official_gate", official),
        (
            "discrimination_only",
            replace(
                official,
                require_slope=False,
                require_intercept=False,
                require_positive_brier_skill=False,
            ),
        ),
        (
            "calibration_only",
            replace(official, require_discrimination=False),
        ),
        ("without_slope", replace(official, require_slope=False)),
        ("without_intercept", replace(official, require_intercept=False)),
        (
            "without_brier_skill",
            replace(official, require_positive_brier_skill=False),
        ),
    )
    rows: list[dict[str, Any]] = []
    for name, envelope in variants:
        counts = _count_matrix(matrix, envelope)
        rows.append(
            {
                "ablation": name,
                **counts,
                "evaluable_pairs": (
                    counts["certified"]
                    + counts["calibration_recovery_candidate"]
                    + counts["discrimination_failure"]
                ),
            }
        )
    return rows
