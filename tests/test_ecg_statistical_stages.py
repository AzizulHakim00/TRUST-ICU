from __future__ import annotations

import json

import pytest

from trust_icu.ecg_statistical_stages import (
    LOCKED_LABEL_CODES,
    build_model_stage_payload,
    build_phase1_stage_payload,
    load_stage_payload,
    verify_common_identity,
    write_stage_payload,
)


def _model_payload() -> dict:
    return build_model_stage_payload(
        protocol_version="0.4.0",
        protocol_sha256="p" * 64,
        phase0_report_sha256="r" * 64,
        phase0_model_sha256="m" * 64,
        model_index_sha256="i" * 64,
        label_manifest_sha256="l" * 64,
        normalization_stats_sha256="n" * 64,
        bootstrap_repeats=2000,
        model_results={"internal": {"macro": {}}},
    )


def _phase1_payload() -> dict:
    return build_phase1_stage_payload(
        protocol_version="0.4.0",
        protocol_sha256="p" * 64,
        phase0_report_sha256="r" * 64,
        phase0_model_sha256="m" * 64,
        model_index_sha256="i" * 64,
        label_manifest_sha256="l" * 64,
        normalization_stats_sha256="n" * 64,
        phase1_report_sha256="q" * 64,
        matched_results={"pair": {}},
    )


def test_stage_payload_round_trip_and_locked_label_order(tmp_path):
    path = tmp_path / "stage.json"
    digest = write_stage_payload(path, _model_payload())
    loaded = load_stage_payload(path, "model_comparison_statistics")
    assert loaded["stage_sha256"] == digest
    assert tuple(loaded["label_codes"]) == LOCKED_LABEL_CODES


def test_stage_payload_fails_closed_after_tampering(tmp_path):
    path = tmp_path / "stage.json"
    write_stage_payload(path, _model_payload())
    payload = json.loads(path.read_text())
    payload["bootstrap_repeats"] = 1999
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="SHA-256 verification failed"):
        load_stage_payload(path, "model_comparison_statistics")


def test_cross_stage_identity_mismatch_fails_closed():
    model = _model_payload()
    phase1 = _phase1_payload()
    phase1["model_index_sha256"] = "x" * 64
    with pytest.raises(RuntimeError, match="model_index_sha256"):
        verify_common_identity(model, phase1)


def test_cross_stage_identity_matches():
    verify_common_identity(_model_payload(), _phase1_payload())
