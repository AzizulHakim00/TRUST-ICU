"""Aggregate-only feasibility gates for TRUST-ICU Phase 0.

This module intentionally accepts no patient-level arrays. It converts audited aggregate
statistics into a deterministic go/no-go decision after external validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from trust_icu.config import StudyConfig


@dataclass(frozen=True)
class DatasetAudit:
    task: str
    development_n: int
    development_events: int
    external_n: int
    external_events: int
    external_hospitals: int
    external_pr_auc: float
    external_brier: float
    external_calibration_slope: float
    external_calibration_intercept: float
    no_post_index_leakage: bool
    outcome_definition_equivalent: bool

    @property
    def development_event_rate(self) -> float:
        return self.development_events / self.development_n

    @property
    def external_event_rate(self) -> float:
        return self.external_events / self.external_n

    @property
    def external_pr_auc_prevalence_ratio(self) -> float:
        prevalence = self.external_event_rate
        return self.external_pr_auc / prevalence if prevalence > 0 else 0.0

    def validate_ranges(self) -> None:
        for label, total, events in (
            ("development", self.development_n, self.development_events),
            ("external", self.external_n, self.external_events),
        ):
            if total <= 0:
                raise ValueError(f"{label}_n must be greater than zero.")
            if not 0 <= events <= total:
                raise ValueError(f"{label}_events must be between zero and {label}_n.")
        if self.external_hospitals <= 0:
            raise ValueError("external_hospitals must be greater than zero.")
        if not 0 <= self.external_pr_auc <= 1:
            raise ValueError("external_pr_auc must be between zero and one.")
        if not 0 <= self.external_brier <= 1:
            raise ValueError("external_brier must be between zero and one.")


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    observed: Any
    requirement: str


@dataclass(frozen=True)
class GateDecision:
    task: str
    continue_to_architecture_development: bool
    passed_checks: int
    total_checks: int
    recommended_action: str
    checks: tuple[GateCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_feasibility(audit: DatasetAudit, config: StudyConfig) -> GateDecision:
    """Apply preregistered gates without optimizing or changing thresholds."""

    audit.validate_ranges()
    gates = config.gates
    checks = (
        GateCheck(
            "development_positive_events",
            audit.development_events >= gates.minimum_positive_events_per_primary_task,
            audit.development_events,
            f">={gates.minimum_positive_events_per_primary_task}",
        ),
        GateCheck(
            "external_positive_events",
            audit.external_events >= gates.minimum_external_positive_events_per_primary_task,
            audit.external_events,
            f">={gates.minimum_external_positive_events_per_primary_task}",
        ),
        GateCheck(
            "development_event_rate",
            gates.minimum_event_rate
            <= audit.development_event_rate
            <= gates.maximum_event_rate,
            audit.development_event_rate,
            f"[{gates.minimum_event_rate}, {gates.maximum_event_rate}]",
        ),
        GateCheck(
            "external_event_rate",
            gates.minimum_event_rate <= audit.external_event_rate <= gates.maximum_event_rate,
            audit.external_event_rate,
            f"[{gates.minimum_event_rate}, {gates.maximum_event_rate}]",
        ),
        GateCheck(
            "external_pr_auc_prevalence_ratio",
            audit.external_pr_auc_prevalence_ratio
            >= gates.minimum_external_pr_auc_prevalence_ratio,
            audit.external_pr_auc_prevalence_ratio,
            f">={gates.minimum_external_pr_auc_prevalence_ratio}",
        ),
        GateCheck(
            "external_calibration_slope",
            abs(audit.external_calibration_slope - 1.0)
            <= gates.maximum_external_calibration_slope_deviation,
            audit.external_calibration_slope,
            f"abs(slope-1)<={gates.maximum_external_calibration_slope_deviation}",
        ),
        GateCheck(
            "external_calibration_intercept",
            abs(audit.external_calibration_intercept)
            <= gates.maximum_absolute_external_calibration_intercept,
            audit.external_calibration_intercept,
            f"abs(intercept)<={gates.maximum_absolute_external_calibration_intercept}",
        ),
        GateCheck(
            "external_hospital_count",
            audit.external_hospitals >= gates.minimum_number_of_external_hospitals,
            audit.external_hospitals,
            f">={gates.minimum_number_of_external_hospitals}",
        ),
        GateCheck(
            "no_post_index_leakage",
            audit.no_post_index_leakage if gates.require_no_post_index_leakage else True,
            audit.no_post_index_leakage,
            "true",
        ),
        GateCheck(
            "outcome_definition_equivalence",
            audit.outcome_definition_equivalent
            if gates.require_outcome_definition_equivalence
            else True,
            audit.outcome_definition_equivalent,
            "true",
        ),
    )
    passed = sum(check.passed for check in checks)
    if gates.require_all_checks_to_continue:
        continue_study = passed == len(checks)
    else:
        continue_study = passed >= len(checks) - 1

    action = (
        "Proceed to the locked architecture-development protocol."
        if continue_study
        else "Stop architecture development for this task; repair definitions before modelling or remove the task."
    )
    return GateDecision(
        task=audit.task,
        continue_to_architecture_development=continue_study,
        passed_checks=passed,
        total_checks=len(checks),
        recommended_action=action,
        checks=checks,
    )
