from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from trust_icu.ecg_manifest import (
    build_label_manifest,
    load_and_verify_header_audit,
    load_and_verify_label_manifest,
    load_and_verify_ptbxl_label_concordance_audit,
    write_label_manifest,
)
from trust_icu.ecg_ptbxl_labels import PTBXL_SCP_TO_CHALLENGE

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "schemas/open_ecg_protocol.yaml"
MAPPING = ROOT / "schemas/challenge2020_scored_classes.csv"


def _hash(payload: dict, key: str) -> str:
    material = copy.deepcopy(payload)
    material[key] = ""
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_valid_header_audit(path: Path) -> Path:
    labels = []
    for code, spec in PTBXL_SCP_TO_CHALLENGE.items():
        labels.append(
            {
                "canonical_code": code,
                "abbreviation": spec["abbreviation"],
                "member_codes": [code],
                "development_positives": spec["challenge_positive_count"],
                "external_positives": {
                    "georgia": 200,
                    "cpsc_2018": 200,
                    "cpsc_2018_extra": 200,
                },
                "external_domains_meeting_threshold": 3,
                "eligible": True,
            }
        )
    payload = {
        "protocol_version": "0.4.0",
        "ready_for_waveform_stage": True,
        "ptbxl_crosswalk": {"required": False, "valid": True},
        "eligible_labels": list(PTBXL_SCP_TO_CHALLENGE),
        "labels": labels,
        "manifest_sha256": "",
    }
    payload["manifest_sha256"] = _hash(payload, "manifest_sha256")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_valid_concordance(path: Path) -> Path:
    rows = []
    for code, spec in PTBXL_SCP_TO_CHALLENGE.items():
        rows.append(
            {
                "canonical_code": code,
                "abbreviation": spec["abbreviation"],
                "scp_codes": list(spec["scp_codes"]),
                "challenge_positive_count": spec["challenge_positive_count"],
                "original_ptbxl_union_key_present_count": spec["challenge_positive_count"],
                "exact_union_key_present_match": True,
            }
        )
    payload = {
        "audit_version": "0.2.0",
        "ptbxl_rows": 21837,
        "unique_ecg_ids": 21837,
        "unique_patients": 18885,
        "folds_present": list(range(1, 11)),
        "patients_spanning_multiple_folds": 0,
        "label_rows": rows,
        "selected_count_semantics": "union_of_scp_key_presence_per_record",
        "all_labels_exactly_concordant": True,
        "ready_for_original_ptbxl_development": True,
        "blockers": [],
        "audit_sha256": "",
    }
    payload["audit_sha256"] = _hash(payload, "audit_sha256")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _build(tmp_path: Path) -> tuple[Path, Path, dict]:
    header = _write_valid_header_audit(tmp_path / "header.json")
    concordance = _write_valid_concordance(tmp_path / "concordance.json")
    manifest = build_label_manifest(
        header_audit_path=header,
        ptbxl_label_concordance_path=concordance,
        protocol_path=PROTOCOL,
        scored_mapping_path=MAPPING,
    )
    return header, concordance, manifest


def test_label_manifest_is_deterministic_and_verifiable(tmp_path: Path) -> None:
    header = _write_valid_header_audit(tmp_path / "header.json")
    concordance = _write_valid_concordance(tmp_path / "concordance.json")
    first = build_label_manifest(
        header_audit_path=header,
        ptbxl_label_concordance_path=concordance,
        protocol_path=PROTOCOL,
        scored_mapping_path=MAPPING,
    )
    second = build_label_manifest(
        header_audit_path=header,
        ptbxl_label_concordance_path=concordance,
        protocol_path=PROTOCOL,
        scored_mapping_path=MAPPING,
    )
    assert first == second
    assert first["label_count"] == 7
    assert first["challenge_ptbxl_model_input"] is False
    assert first["manifest_sha256"] == _hash(first, "manifest_sha256")
    output = tmp_path / "labels.json"
    report = write_label_manifest(first, output)
    assert set(report.canonical_codes) == set(PTBXL_SCP_TO_CHALLENGE)
    assert report.ptbxl_label_concordance_audit_sha256
    loaded = load_and_verify_label_manifest(output)
    assert loaded["manifest_sha256"] == report.manifest_sha256


def test_tampered_header_audit_is_rejected(tmp_path: Path) -> None:
    audit = _write_valid_header_audit(tmp_path / "header.json")
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["eligible_labels"].pop()
    audit.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="header audit SHA-256 verification failed"):
        load_and_verify_header_audit(audit)


def test_reverse_crosswalk_requirement_is_rejected(tmp_path: Path) -> None:
    audit = _write_valid_header_audit(tmp_path / "header.json")
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["ptbxl_crosswalk"] = {"required": True, "valid": True}
    payload["manifest_sha256"] = _hash(payload, "manifest_sha256")
    audit.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="refuses a Challenge/PTB-XL reverse-crosswalk"):
        load_and_verify_header_audit(audit)


def test_tampered_ptbxl_concordance_is_rejected(tmp_path: Path) -> None:
    audit = _write_valid_concordance(tmp_path / "concordance.json")
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["label_rows"][0]["original_ptbxl_union_key_present_count"] += 1
    audit.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="concordance audit SHA-256 verification failed"):
        load_and_verify_ptbxl_label_concordance_audit(audit)


def test_unready_ptbxl_concordance_cannot_lock_labels(tmp_path: Path) -> None:
    header = _write_valid_header_audit(tmp_path / "header.json")
    concordance = _write_valid_concordance(tmp_path / "concordance.json")
    payload = json.loads(concordance.read_text(encoding="utf-8"))
    payload["all_labels_exactly_concordant"] = False
    payload["ready_for_original_ptbxl_development"] = False
    payload["blockers"] = ["mismatch"]
    payload["audit_sha256"] = _hash(payload, "audit_sha256")
    concordance.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="seven PTB-XL development label unions"):
        build_label_manifest(
            header_audit_path=header,
            ptbxl_label_concordance_path=concordance,
            protocol_path=PROTOCOL,
            scored_mapping_path=MAPPING,
        )


def test_evidence_sources_must_agree_on_all_seven_labels(tmp_path: Path) -> None:
    header = _write_valid_header_audit(tmp_path / "header.json")
    concordance = _write_valid_concordance(tmp_path / "concordance.json")
    payload = json.loads(header.read_text(encoding="utf-8"))
    removed = payload["eligible_labels"].pop()
    payload["labels"] = [item for item in payload["labels"] if item["canonical_code"] != removed]
    payload["manifest_sha256"] = _hash(payload, "manifest_sha256")
    header.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="agree on exactly seven labels"):
        build_label_manifest(
            header_audit_path=header,
            ptbxl_label_concordance_path=concordance,
            protocol_path=PROTOCOL,
            scored_mapping_path=MAPPING,
        )


def test_tampered_label_manifest_is_rejected(tmp_path: Path) -> None:
    _, _, manifest = _build(tmp_path)
    output = tmp_path / "labels.json"
    write_label_manifest(manifest, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["labels"][0]["abbreviation"] = "changed"
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 verification failed"):
        load_and_verify_label_manifest(output)
