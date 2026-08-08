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
    write_label_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "schemas/open_ecg_protocol.yaml"
MAPPING = ROOT / "schemas/challenge2020_scored_classes.csv"


def _hash(payload: dict, key: str) -> str:
    material = copy.deepcopy(payload)
    material[key] = ""
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_valid_audit(path: Path) -> Path:
    payload = {
        "protocol_version": "0.3.0",
        "ready_for_waveform_stage": True,
        "ptbxl_crosswalk": {"valid": True},
        "eligible_labels": ["164889003", "426783006"],
        "labels": [
            {
                "canonical_code": "164889003",
                "abbreviation": "AF",
                "member_codes": ["164889003"],
                "development_positives": 1514,
                "external_positives": {"georgia": 570, "cpsc_2018": 1221, "cpsc_2018_extra": 153},
                "external_domains_meeting_threshold": 3,
                "eligible": True,
            },
            {
                "canonical_code": "426783006",
                "abbreviation": "NSR",
                "member_codes": ["426783006"],
                "development_positives": 18092,
                "external_positives": {"georgia": 1752, "cpsc_2018": 918, "cpsc_2018_extra": 4},
                "external_domains_meeting_threshold": 2,
                "eligible": True,
            },
        ],
        "manifest_sha256": "",
    }
    payload["manifest_sha256"] = _hash(payload, "manifest_sha256")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_label_manifest_is_deterministic_and_verifiable(tmp_path: Path) -> None:
    audit = _write_valid_audit(tmp_path / "audit.json")
    first = build_label_manifest(
        header_audit_path=audit,
        protocol_path=PROTOCOL,
        scored_mapping_path=MAPPING,
    )
    second = build_label_manifest(
        header_audit_path=audit,
        protocol_path=PROTOCOL,
        scored_mapping_path=MAPPING,
    )
    assert first == second
    assert first["label_count"] == 2
    assert first["manifest_sha256"] == _hash(first, "manifest_sha256")
    output = tmp_path / "labels.json"
    report = write_label_manifest(first, output)
    assert report.canonical_codes == ("164889003", "426783006")
    loaded = load_and_verify_label_manifest(output)
    assert loaded["manifest_sha256"] == report.manifest_sha256


def test_tampered_header_audit_is_rejected(tmp_path: Path) -> None:
    audit = _write_valid_audit(tmp_path / "audit.json")
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["eligible_labels"].append("270492004")
    audit.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 verification failed"):
        load_and_verify_header_audit(audit)


def test_unready_header_audit_cannot_lock_labels(tmp_path: Path) -> None:
    audit = _write_valid_audit(tmp_path / "audit.json")
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["ready_for_waveform_stage"] = False
    payload["manifest_sha256"] = _hash(payload, "manifest_sha256")
    audit.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="blocked until the header audit"):
        build_label_manifest(
            header_audit_path=audit,
            protocol_path=PROTOCOL,
            scored_mapping_path=MAPPING,
        )


def test_tampered_label_manifest_is_rejected(tmp_path: Path) -> None:
    audit = _write_valid_audit(tmp_path / "audit.json")
    manifest = build_label_manifest(
        header_audit_path=audit,
        protocol_path=PROTOCOL,
        scored_mapping_path=MAPPING,
    )
    output = tmp_path / "labels.json"
    write_label_manifest(manifest, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["labels"][0]["abbreviation"] = "changed"
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 verification failed"):
        load_and_verify_label_manifest(output)
