"""Outcome-contract loading, hashing and lock enforcement.

The module operates only on public metadata. It deliberately does not contain patient-level
extraction logic or final eICU vocabularies, which must be derived inside a credentialed
environment and clinically reviewed before modelling.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


VALID_STATUSES = {
    "upstream_concept_verified",
    "vocabulary_discovery_required",
    "pending_credentialed_validation",
    "locally_validated",
    "locked",
}
LOCKED_SOURCE_STATUSES = {"locally_validated", "locked"}
REQUIRED_OUTCOMES = {
    "invasive_mechanical_ventilation",
    "vasopressor_initiation",
    "renal_replacement_therapy",
}


@dataclass(frozen=True)
class OutcomeLockReport:
    task: str
    ready_for_model_training: bool
    mimic_source_locked: bool
    eicu_source_locked: bool
    clinical_equivalence_approved: bool
    synthetic_timeline_tests_passed: bool
    missing_requirements: tuple[str, ...]
    contract_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(raw: dict[str, Any]) -> str:
    return json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def contract_sha256(raw: dict[str, Any]) -> str:
    """Return a stable hash for the complete outcome-contract document."""

    return hashlib.sha256(_canonical_json(raw).encode("utf-8")).hexdigest()


def _validate_window(name: str, values: dict[str, Any]) -> None:
    start = int(values["start_hours"])
    end = int(values["end_hours"])
    if start < 0 or end <= start:
        raise ValueError(f"{name} must satisfy 0 <= start_hours < end_hours.")
    if values.get("interval") != "left_closed_right_open":
        raise ValueError(f"{name} must use a left-closed, right-open interval.")


def _validate_source_status(task: str, database: str, source: dict[str, Any]) -> None:
    status = source.get("status")
    if status not in VALID_STATUSES:
        raise ValueError(
            f"{task}.{database}.status must be one of {sorted(VALID_STATUSES)}; got {status!r}."
        )


def validate_outcome_contracts(raw: dict[str, Any]) -> None:
    """Validate public contract structure without implying clinical approval."""

    if not isinstance(raw, dict):
        raise ValueError("Outcome-contract root must be a mapping.")
    _validate_window("observation_window", raw["observation_window"])
    _validate_window("prediction_window", raw["prediction_window"])

    observation_end = int(raw["observation_window"]["end_hours"])
    prediction_start = int(raw["prediction_window"]["start_hours"])
    if observation_end != prediction_start:
        raise ValueError("Prediction must start exactly when the observation window ends.")

    outcomes = raw.get("outcomes")
    if not isinstance(outcomes, dict):
        raise ValueError("outcomes must be a mapping.")
    missing = REQUIRED_OUTCOMES - outcomes.keys()
    if missing:
        raise ValueError(f"Missing required outcome contracts: {sorted(missing)}")

    for task, contract in outcomes.items():
        if not isinstance(contract, dict):
            raise ValueError(f"Outcome contract {task!r} must be a mapping.")
        task_status = contract.get("task_status")
        if task_status not in VALID_STATUSES:
            raise ValueError(f"Invalid task_status for {task}: {task_status!r}")
        for database in ("mimic", "eicu"):
            source = contract.get(database)
            if not isinstance(source, dict):
                raise ValueError(f"{task} must define a {database} source mapping.")
            _validate_source_status(task, database, source)


def load_outcome_contracts(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate the versioned outcome contracts."""

    contract_path = Path(path)
    if not contract_path.is_file():
        raise FileNotFoundError(f"Outcome contracts not found: {contract_path}")
    with contract_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    validate_outcome_contracts(raw)
    return raw


def evaluate_outcome_locks(raw: dict[str, Any]) -> tuple[OutcomeLockReport, ...]:
    """Evaluate whether each task is allowed to proceed to model training.

    A source being present in an upstream repository is not enough. Both databases must be
    locally validated on the exact installed versions, clinical equivalence must be approved,
    and synthetic boundary tests must pass.
    """

    validate_outcome_contracts(raw)
    document_hash = contract_sha256(raw)
    reports: list[OutcomeLockReport] = []

    for task, contract in raw["outcomes"].items():
        mimic_locked = contract["mimic"]["status"] in LOCKED_SOURCE_STATUSES
        eicu_locked = contract["eicu"]["status"] in LOCKED_SOURCE_STATUSES
        equivalence = contract.get("clinical_equivalence_review") == "approved"
        timelines = contract.get("synthetic_timeline_tests") == "passed"

        missing: list[str] = []
        if not mimic_locked:
            missing.append("mimic_local_validation")
        if not eicu_locked:
            missing.append("eicu_local_validation")
        if not equivalence:
            missing.append("clinical_equivalence_review")
        if not timelines:
            missing.append("synthetic_timeline_tests")

        reports.append(
            OutcomeLockReport(
                task=task,
                ready_for_model_training=not missing,
                mimic_source_locked=mimic_locked,
                eicu_source_locked=eicu_locked,
                clinical_equivalence_approved=equivalence,
                synthetic_timeline_tests_passed=timelines,
                missing_requirements=tuple(missing),
                contract_sha256=document_hash,
            )
        )
    return tuple(reports)


def assert_task_training_allowed(raw: dict[str, Any], task: str) -> OutcomeLockReport:
    """Fail closed when a task has not completed outcome locking."""

    reports = {report.task: report for report in evaluate_outcome_locks(raw)}
    if task not in reports:
        raise KeyError(f"Unknown outcome task: {task}")
    report = reports[task]
    if not report.ready_for_model_training:
        missing = ", ".join(report.missing_requirements)
        raise RuntimeError(f"Model training is prohibited for {task}; missing: {missing}.")
    return report


def classify_event_offset_minutes(
    offset_minutes: int,
    *,
    observation_end_minutes: int = 360,
    prediction_end_minutes: int = 1080,
) -> str:
    """Classify an event using the locked [0, 6h) and [6h, 18h) boundaries."""

    if observation_end_minutes <= 0:
        raise ValueError("observation_end_minutes must be positive.")
    if prediction_end_minutes <= observation_end_minutes:
        raise ValueError("prediction_end_minutes must exceed observation_end_minutes.")
    if offset_minutes < 0:
        return "pre_icu"
    if offset_minutes < observation_end_minutes:
        return "observation"
    if offset_minutes < prediction_end_minutes:
        return "prediction"
    return "post_prediction"
