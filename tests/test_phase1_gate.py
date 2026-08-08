from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from trust_icu.phase1_gate import evaluate_phase1_activation, validate_phase1_protocol

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "schemas/phase1_conditional_protocol.yaml"
TASKS = (
    "invasive_mechanical_ventilation",
    "vasopressor_initiation",
    "renal_replacement_therapy",
)


def _write_phase0_report(path: Path, passing: set[str], *, synthetic: bool = False) -> Path:
    payload = {
        "study": "TRUST-ICU",
        "phase": "phase0",
        "data_classification": "synthetic" if synthetic else "credentialed_aggregate",
        "tasks": [
            {
                "task": task,
                "feasibility_decision": {
                    "continue_to_architecture_development": task in passing,
                },
            }
            for task in TASKS
        ],
        "report_sha256": "",
    }
    material = copy.deepcopy(payload)
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["report_sha256"] = hashlib.sha256(encoded).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase1_protocol_is_valid_and_inactive() -> None:
    report = validate_phase1_protocol(PROTOCOL)
    assert report["valid"] is True
    assert report["status"] == "inactive_until_phase0_task_passes"
    assert report["label_budgets"] == [0, 50, 100, 250, 500, 1000]


def test_phase1_without_phase0_report_is_blocked() -> None:
    activation = evaluate_phase1_activation(protocol_path=PROTOCOL)
    assert activation.architecture_or_method_development_allowed is False
    assert activation.status == "awaiting_verified_phase0_report"
    assert activation.eligible_tasks == ()


def test_phase1_activates_only_passing_tasks(tmp_path: Path) -> None:
    report_path = _write_phase0_report(
        tmp_path / "phase0.json",
        {"vasopressor_initiation"},
    )
    activation = evaluate_phase1_activation(
        protocol_path=PROTOCOL,
        phase0_report_path=report_path,
    )
    assert activation.architecture_or_method_development_allowed is True
    assert activation.status == "active_for_eligible_tasks"
    assert activation.eligible_tasks == ("vasopressor_initiation",)
    assert set(activation.blocked_tasks) == {
        "invasive_mechanical_ventilation",
        "renal_replacement_therapy",
    }


def test_phase1_remains_inactive_when_all_tasks_fail(tmp_path: Path) -> None:
    report_path = _write_phase0_report(tmp_path / "phase0.json", set())
    activation = evaluate_phase1_activation(
        protocol_path=PROTOCOL,
        phase0_report_path=report_path,
    )
    assert activation.architecture_or_method_development_allowed is False
    assert activation.status == "phase1_not_activated"
    assert activation.blockers == ("no_phase0_task_passed_prespecified_feasibility_gate",)


def test_synthetic_phase0_report_cannot_activate_phase1(tmp_path: Path) -> None:
    report_path = _write_phase0_report(
        tmp_path / "phase0.json",
        {"invasive_mechanical_ventilation"},
        synthetic=True,
    )
    with pytest.raises(RuntimeError, match="Synthetic Phase 0 reports cannot activate Phase 1"):
        evaluate_phase1_activation(
            protocol_path=PROTOCOL,
            phase0_report_path=report_path,
        )


def test_tampered_phase0_report_is_rejected(tmp_path: Path) -> None:
    report_path = _write_phase0_report(
        tmp_path / "phase0.json",
        {"renal_replacement_therapy"},
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["tasks"][0]["feasibility_decision"]["continue_to_architecture_development"] = True
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 verification failed"):
        evaluate_phase1_activation(
            protocol_path=PROTOCOL,
            phase0_report_path=report_path,
        )
