"""Fail-closed activation logic for the conditional TRUST-ICU Phase 1 study.

This module does not implement Phase 1 modelling. It only verifies the prospective protocol and
an immutable Phase 0 aggregate report, then identifies which tasks are permitted to advance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from trust_icu.reporting import load_and_verify_phase0_report

_EXPECTED_TASKS = {
    "invasive_mechanical_ventilation",
    "vasopressor_initiation",
    "renal_replacement_therapy",
}


@dataclass(frozen=True)
class Phase1ActivationReport:
    protocol_version: str
    protocol_sha256: str
    status: str
    eligible_tasks: tuple[str, ...]
    blocked_tasks: tuple[str, ...]
    blockers: tuple[str, ...]
    phase0_report_sha256: str | None
    architecture_or_method_development_allowed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_phase1_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    raw = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Phase 1 protocol must be a YAML mapping.")
    return raw


def validate_phase1_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = load_phase1_protocol(protocol_path)

    if str(protocol.get("status")) != "inactive_until_phase0_task_passes":
        raise ValueError("Phase 1 protocol must remain inactive until a Phase 0 task passes.")

    activation = protocol.get("activation")
    if not isinstance(activation, dict):
        raise ValueError("Phase 1 activation block is required.")
    if activation.get("minimum_eligible_tasks") != 1:
        raise ValueError("Phase 1 must require at least one eligible Phase 0 task.")
    if activation.get("require_verified_phase0_hash") is not True:
        raise ValueError("Verified Phase 0 report hashes are mandatory.")
    if activation.get("require_locked_outcome_contracts") is not True:
        raise ValueError("Locked outcome contracts are mandatory for Phase 1.")
    if activation.get("prohibit_activation_from_synthetic_report") is not True:
        raise ValueError("Synthetic Phase 0 reports must not activate Phase 1.")

    eligible = set(protocol.get("eligible_tasks", []))
    if eligible != _EXPECTED_TASKS:
        raise ValueError("Phase 1 eligible task list must exactly match the three primary tasks.")

    frozen = set(protocol.get("frozen_from_phase0", []))
    required_frozen = {
        "outcome_definition",
        "observation_window",
        "prediction_window",
        "feature_contract",
        "development_selected_model",
        "calibration_map",
        "primary_external_cohort",
        "primary_feasibility_decision",
    }
    if not required_frozen.issubset(frozen):
        raise ValueError("Phase 1 must freeze every required Phase 0 design element.")

    budgets = (
        protocol.get("local_label_budgets", {}).get("samples_per_site")
        if isinstance(protocol.get("local_label_budgets"), dict)
        else None
    )
    if budgets != [0, 50, 100, 250, 500, 1000]:
        raise ValueError("Phase 1 local label budgets must remain prespecified.")

    localization = protocol.get("localization_baselines")
    if not isinstance(localization, dict):
        raise ValueError("Phase 1 localization baseline block is required.")
    for prohibited in ("model_retraining", "fine_tuning", "external_feature_selection"):
        block = localization.get(prohibited)
        if not isinstance(block, dict) or block.get("allowed") is not False:
            raise ValueError(f"{prohibited} must remain prohibited in Phase 1.")

    privacy = protocol.get("privacy")
    if not isinstance(privacy, dict):
        raise ValueError("Phase 1 privacy block is required.")
    for required_true in (
        "public_outputs_aggregate_only",
        "prohibit_patient_identifiers",
        "prohibit_hospital_identifiers",
        "prohibit_row_level_predictions",
        "secure_runtime_required_for_patient_level_analysis",
    ):
        if privacy.get(required_true) is not True:
            raise ValueError(f"Phase 1 privacy rule must be true: {required_true}")

    return {
        "valid": True,
        "version": str(protocol.get("version")),
        "status": str(protocol.get("status")),
        "sha256": _sha256_file(protocol_path),
        "label_budgets": budgets,
        "eligible_tasks": sorted(eligible),
    }


def evaluate_phase1_activation(
    *,
    protocol_path: str | Path,
    phase0_report_path: str | Path | None = None,
) -> Phase1ActivationReport:
    """Return a deterministic activation decision without implementing Phase 1 methods."""

    protocol_path = Path(protocol_path).resolve()
    protocol_validation = validate_phase1_protocol(protocol_path)

    if phase0_report_path is None:
        return Phase1ActivationReport(
            protocol_version=protocol_validation["version"],
            protocol_sha256=protocol_validation["sha256"],
            status="awaiting_verified_phase0_report",
            eligible_tasks=(),
            blocked_tasks=tuple(sorted(_EXPECTED_TASKS)),
            blockers=("verified_phase0_report_missing",),
            phase0_report_sha256=None,
            architecture_or_method_development_allowed=False,
        )

    report_path = Path(phase0_report_path).resolve()
    report = load_and_verify_phase0_report(report_path)

    data_classification = report.get("data_classification")
    if data_classification == "synthetic":
        raise RuntimeError("Synthetic Phase 0 reports cannot activate Phase 1.")

    tasks = report.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 3:
        raise ValueError("Verified Phase 0 report must contain exactly three task reports.")

    eligible_tasks: list[str] = []
    blocked_tasks: list[str] = []
    for task_report in tasks:
        if not isinstance(task_report, dict):
            raise ValueError("Each Phase 0 task report must be an object.")
        task = str(task_report.get("task", ""))
        if task not in _EXPECTED_TASKS:
            raise ValueError(f"Unexpected Phase 0 task: {task!r}")
        decision = task_report.get("feasibility_decision")
        if not isinstance(decision, dict):
            raise ValueError(f"Missing feasibility decision for task {task}.")
        if decision.get("continue_to_architecture_development") is True:
            eligible_tasks.append(task)
        else:
            blocked_tasks.append(task)

    if len(set(eligible_tasks) | set(blocked_tasks)) != 3:
        raise ValueError("Phase 0 task decisions are incomplete or duplicated.")

    blockers: list[str] = []
    if not eligible_tasks:
        blockers.append("no_phase0_task_passed_prespecified_feasibility_gate")

    return Phase1ActivationReport(
        protocol_version=protocol_validation["version"],
        protocol_sha256=protocol_validation["sha256"],
        status="active_for_eligible_tasks" if eligible_tasks else "phase1_not_activated",
        eligible_tasks=tuple(sorted(eligible_tasks)),
        blocked_tasks=tuple(sorted(blocked_tasks)),
        blockers=tuple(blockers),
        phase0_report_sha256=_sha256_file(report_path),
        architecture_or_method_development_allowed=bool(eligible_tasks),
    )


def write_activation_report(report: Phase1ActivationReport, output: str | Path) -> None:
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
