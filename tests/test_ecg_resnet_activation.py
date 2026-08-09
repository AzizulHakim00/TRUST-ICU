from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_open_ecg_resnet_activation.py"
SPEC = importlib.util.spec_from_file_location("validate_open_ecg_resnet_activation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

H = "a" * 64
BASE = {
    "activation_version": "0.1.0",
    "study": "TRUST-ECG",
    "protocol_version": "0.4.0",
    "reference_model": "logistic_reference",
    "logistic_workflow_run_id": 31330770461,
    "logistic_workflow_path": ".github/workflows/open-ecg-real-waveform-phase0-v04.yml",
    "logistic_branch": "open-ecg-transportability",
    "logistic_head_sha": "d4e66412a783ba383d56d36741b2e66b73f087f0",
    "logistic_conclusion": "success",
    "logistic_primary_gate_eligible": False,
    "external_recovery_pool_used": False,
    "waveform_audit_ready": True,
    "model_index_ready": True,
    "challenge_ptbxl_model_input": False,
    "label_codes": ["164889003", "164890007"],
    "logistic_report_sha256": H,
    "header_audit_sha256": H,
    "ptbxl_label_concordance_audit_sha256": H,
    "label_manifest_sha256": H,
    "waveform_audit_sha256": H,
    "normalization_stats_sha256": H,
    "ptbxl_assignment_sha256": H,
    "model_index_sha256": H,
    "model_index_audit_sha256": H,
}


def _payload(**changes):
    payload = dict(BASE)
    payload.update(changes)
    return payload


def test_valid_activation_is_accepted() -> None:
    summary = MODULE.validate_activation_payload(_payload())
    assert summary["logistic_workflow_run_id"] == 31330770461
    assert summary["protocol_version"] == "0.4.0"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("logistic_conclusion", "failure"),
        ("logistic_primary_gate_eligible", True),
        ("external_recovery_pool_used", True),
        ("waveform_audit_ready", False),
        ("model_index_ready", False),
        ("challenge_ptbxl_model_input", True),
        ("logistic_workflow_path", ".github/workflows/other.yml"),
    ],
)
def test_activation_fails_closed_on_boundary_drift(field: str, value) -> None:
    with pytest.raises(ValueError):
        MODULE.validate_activation_payload(_payload(**{field: value}))


def test_activation_rejects_invalid_hash() -> None:
    with pytest.raises(ValueError, match="logistic_report_sha256"):
        MODULE.validate_activation_payload(_payload(logistic_report_sha256="not-a-hash"))


def test_activation_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="unique"):
        MODULE.validate_activation_payload(_payload(label_codes=["A", "A"]))
