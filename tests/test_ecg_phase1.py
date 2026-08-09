from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from trust_icu.ecg_phase1 import (
    _canonical_hash,
    _fit_intercept_only,
    _fit_local_platt,
    _sigmoid,
    _stable_draw_indices,
    build_phase1_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "schemas/open_ecg_protocol.yaml"
LABELS = ["RBBB", "AF", "LBBB", "IAVB", "PAC", "NSR", "STACH"]
SOURCES = ["georgia", "cpsc_2018", "cpsc_2018_extra"]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _report_payload(candidate_pairs: set[tuple[str, str]]) -> dict:
    external = {}
    for source in SOURCES:
        external[source] = {}
        for code in LABELS:
            status = (
                "calibration_recovery_candidate"
                if (source, code) in candidate_pairs
                else "certified"
            )
            external[source][code] = {
                "status": status,
                "metrics": None,
                "reasons": [] if status == "certified" else ["calibration_intercept_outside_envelope"],
            }
    payload = {
        "report_version": "0.1.0",
        "study": "TRUST-ECG",
        "model_name": "resnet1d_fixed",
        "model_role": "predeclared_primary_waveform_baseline_not_novel_architecture",
        "primary_gate_eligible": True,
        "protocol_version": "0.4.0",
        "protocol_sha256": _file_sha256(PROTOCOL),
        "model_index_audit_sha256": "1" * 64,
        "model_index_sha256": "2" * 64,
        "label_manifest_sha256": "3" * 64,
        "model_sha256": "4" * 64,
        "label_codes": LABELS,
        "role_rows": {},
        "internal_test": {},
        "external_certification": external,
        "external_recovery_pool_used": False,
        "calibration_fit_role": "ptb_xl_fold_9_only",
        "optimization_role_used": True,
        "report_sha256": "",
    }
    payload["report_sha256"] = _canonical_hash(payload, "report_sha256")
    return payload


def _write_report(tmp_path: Path, candidate_pairs: set[tuple[str, str]]) -> Path:
    path = tmp_path / "phase0.json"
    path.write_text(
        json.dumps(_report_payload(candidate_pairs), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_phase1_activates_only_with_candidates_in_two_domains(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        {
            ("georgia", "RBBB"),
            ("cpsc_2018", "AF"),
        },
    )
    plan = build_phase1_plan(report, PROTOCOL)
    assert plan.activated is True
    assert plan.candidate_domains == ("georgia", "cpsc_2018")
    assert {(item.source, item.label_code) for item in plan.candidate_pairs} == {
        ("georgia", "RBBB"),
        ("cpsc_2018", "AF"),
    }
    assert plan.target_label_budgets == (0, 50, 100, 250, 500, 1000)
    assert plan.repeats == 100
    assert plan.sampling_stratified is False


def test_phase1_remains_blocked_with_only_one_candidate_domain(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        {
            ("georgia", "RBBB"),
            ("georgia", "AF"),
        },
    )
    plan = build_phase1_plan(report, PROTOCOL)
    assert plan.activated is False
    assert plan.status == "blocked_insufficient_recovery_candidate_domains"


def test_phase1_rejects_tampered_primary_report(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        {
            ("georgia", "RBBB"),
            ("cpsc_2018", "AF"),
        },
    )
    payload = json.loads(report.read_text())
    payload["external_recovery_pool_used"] = True
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="premature recovery-pool use"):
        build_phase1_plan(report, PROTOCOL)


def test_phase1_sampling_is_deterministic_without_replacement() -> None:
    first = _stable_draw_indices(
        100,
        25,
        seed=20260808,
        source="georgia",
        label_code="AF",
        repeat=7,
    )
    second = _stable_draw_indices(
        100,
        25,
        seed=20260808,
        source="georgia",
        label_code="AF",
        repeat=7,
    )
    other = _stable_draw_indices(
        100,
        25,
        seed=20260808,
        source="georgia",
        label_code="AF",
        repeat=8,
    )
    assert np.array_equal(first, second)
    assert np.unique(first).size == 25
    assert not np.array_equal(first, other)


def test_intercept_only_recalibration_matches_adaptation_prevalence() -> None:
    y = np.asarray([0] * 60 + [1] * 40, dtype=np.int64)
    base_logits = np.linspace(-1.0, 2.0, y.size)
    shift = _fit_intercept_only(base_logits, y)
    recalibrated = _sigmoid(base_logits + shift)
    assert np.mean(recalibrated) == pytest.approx(np.mean(y), abs=1e-10)


def test_platt_recalibration_returns_finite_parameters() -> None:
    y = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    logits = np.asarray([-2.0, -1.0, -0.25, 0.25, 1.0, 2.0], dtype=np.float64)
    slope, intercept = _fit_local_platt(logits, y, seed=20260808)
    assert np.isfinite(slope)
    assert np.isfinite(intercept)
