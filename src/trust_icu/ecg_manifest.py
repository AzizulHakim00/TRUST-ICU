"""Tamper-evident label manifest locking for TRUST-ECG."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EcgLabelManifest:
    output_path: str
    label_count: int
    canonical_codes: tuple[str, ...]
    source_header_audit_sha256: str
    manifest_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: dict[str, Any], hash_key: str) -> str:
    material = copy.deepcopy(payload)
    material[hash_key] = ""
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_and_verify_header_audit(path: str | Path) -> dict[str, Any]:
    """Load the aggregate ECG header audit and verify its embedded SHA-256."""

    audit_path = Path(path).expanduser().resolve()
    if not audit_path.is_file():
        raise FileNotFoundError(f"ECG header audit not found: {audit_path}")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ECG header audit root must be a JSON object.")
    observed = str(payload.get("manifest_sha256", ""))
    expected = _canonical_hash(payload, "manifest_sha256")
    if not observed or observed != expected:
        raise ValueError("ECG header audit SHA-256 verification failed.")
    if payload.get("ready_for_waveform_stage") is not True:
        raise RuntimeError("ECG label locking is blocked until the header audit is ready for waveform stage.")
    if not isinstance(payload.get("ptbxl_crosswalk"), dict) or payload["ptbxl_crosswalk"].get("valid") is not True:
        raise RuntimeError("ECG label locking requires a verified Challenge/PTB-XL crosswalk.")
    return payload


def build_label_manifest(
    *,
    header_audit_path: str | Path,
    protocol_path: str | Path,
    scored_mapping_path: str | Path,
) -> dict[str, Any]:
    """Build a deterministic label manifest from a verified pre-model header audit."""

    audit = load_and_verify_header_audit(header_audit_path)
    protocol_file = Path(protocol_path).expanduser().resolve()
    mapping_file = Path(scored_mapping_path).expanduser().resolve()
    protocol = yaml.safe_load(protocol_file.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("Open ECG protocol must be a mapping.")
    if str(audit.get("protocol_version")) != str(protocol.get("version")):
        raise ValueError("Header audit protocol version does not match the current ECG protocol.")

    label_rules = protocol.get("prediction_task", {}).get("labels", {})
    inclusion = label_rules.get("inclusion_rules") if isinstance(label_rules, dict) else None
    if not isinstance(inclusion, dict):
        raise ValueError("ECG protocol label inclusion rules are missing.")
    audit_labels = audit.get("labels")
    if not isinstance(audit_labels, list):
        raise ValueError("ECG header audit labels must be a list.")
    eligible = [item for item in audit_labels if isinstance(item, dict) and item.get("eligible") is True]
    expected_codes = tuple(str(code) for code in audit.get("eligible_labels", []))
    observed_codes = tuple(str(item.get("canonical_code", "")) for item in eligible)
    if set(expected_codes) != set(observed_codes) or len(expected_codes) != len(observed_codes):
        raise ValueError("Header audit eligible-label summary is internally inconsistent.")
    if not eligible:
        raise RuntimeError("No eligible ECG labels are available to lock.")

    labels = []
    for item in sorted(eligible, key=lambda value: int(str(value["canonical_code"]))):
        labels.append(
            {
                "canonical_code": str(item["canonical_code"]),
                "abbreviation": str(item["abbreviation"]),
                "member_codes": [str(code) for code in item["member_codes"]],
                "development_positives": int(item["development_positives"]),
                "external_positives": {
                    str(name): int(count)
                    for name, count in sorted(dict(item["external_positives"]).items())
                },
                "external_domains_meeting_threshold": int(item["external_domains_meeting_threshold"]),
            }
        )

    manifest: dict[str, Any] = {
        "manifest_version": "0.1.0",
        "study": "TRUST-ECG",
        "status": "locked_before_waveform_model_training",
        "protocol_version": str(protocol["version"]),
        "source_header_audit_sha256": str(audit["manifest_sha256"]),
        "protocol_sha256": _sha256_file(protocol_file),
        "scored_mapping_sha256": _sha256_file(mapping_file),
        "inclusion_rules": {
            "minimum_development_positive_records": int(inclusion["minimum_development_positive_records"]),
            "minimum_external_positive_records_per_domain": int(
                inclusion["minimum_external_positive_records_per_domain"]
            ),
            "minimum_external_domains_meeting_positive_threshold": int(
                inclusion["minimum_external_domains_meeting_positive_threshold"]
            ),
        },
        "labels": labels,
        "label_count": len(labels),
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest, "manifest_sha256")
    return manifest


def write_label_manifest(manifest: dict[str, Any], output: str | Path) -> EcgLabelManifest:
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected = _canonical_hash(manifest, "manifest_sha256")
    if str(manifest.get("manifest_sha256")) != expected:
        raise ValueError("Refusing to write an ECG label manifest with an invalid hash.")
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return EcgLabelManifest(
        output_path=str(output_path),
        label_count=int(manifest["label_count"]),
        canonical_codes=tuple(str(item["canonical_code"]) for item in manifest["labels"]),
        source_header_audit_sha256=str(manifest["source_header_audit_sha256"]),
        manifest_sha256=str(manifest["manifest_sha256"]),
    )


def load_and_verify_label_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"ECG label manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ECG label manifest root must be a JSON object.")
    observed = str(payload.get("manifest_sha256", ""))
    expected = _canonical_hash(payload, "manifest_sha256")
    if not observed or observed != expected:
        raise ValueError("ECG label manifest SHA-256 verification failed.")
    if payload.get("status") != "locked_before_waveform_model_training":
        raise ValueError("ECG label manifest has an unexpected status.")
    return payload
