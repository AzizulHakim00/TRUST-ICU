"""Tamper-evident two-source label manifest locking for TRUST-ECG."""

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
    ptbxl_label_concordance_audit_sha256: str
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
    """Load Challenge aggregate label-support audit without requiring obsolete PTB-XL crosswalk."""

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
        raise RuntimeError("ECG label locking is blocked until the Challenge label-support audit is ready.")
    crosswalk = payload.get("ptbxl_crosswalk")
    if isinstance(crosswalk, dict) and crosswalk.get("required") is True:
        raise RuntimeError("Protocol v0.4 label locking refuses a Challenge/PTB-XL reverse-crosswalk requirement.")
    return payload


def load_and_verify_ptbxl_label_concordance_audit(path: str | Path) -> dict[str, Any]:
    """Load the real original-PTB-XL aggregate label concordance and verify its embedded hash."""

    audit_path = Path(path).expanduser().resolve()
    if not audit_path.is_file():
        raise FileNotFoundError(f"PTB-XL label-concordance audit not found: {audit_path}")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PTB-XL label-concordance audit root must be a JSON object.")
    observed = str(payload.get("audit_sha256", ""))
    expected = _canonical_hash(payload, "audit_sha256")
    if not observed or observed != expected:
        raise ValueError("PTB-XL label-concordance audit SHA-256 verification failed.")
    if str(payload.get("audit_version")) != "0.2.0":
        raise ValueError("PTB-XL label-concordance audit must use version 0.2.0.")
    if payload.get("selected_count_semantics") != "union_of_scp_key_presence_per_record":
        raise ValueError("PTB-XL label-concordance semantics do not match protocol v0.4.")
    if payload.get("all_labels_exactly_concordant") is not True:
        raise RuntimeError("All seven PTB-XL development label unions must be exactly concordant.")
    if payload.get("ready_for_original_ptbxl_development") is not True:
        raise RuntimeError("Original PTB-XL development remains blocked by its concordance audit.")
    return payload


def build_label_manifest(
    *,
    header_audit_path: str | Path,
    ptbxl_label_concordance_path: str | Path,
    protocol_path: str | Path,
    scored_mapping_path: str | Path,
) -> dict[str, Any]:
    """Lock one canonical label set from independent development and external aggregate evidence."""

    header_audit = load_and_verify_header_audit(header_audit_path)
    concordance = load_and_verify_ptbxl_label_concordance_audit(ptbxl_label_concordance_path)
    protocol_file = Path(protocol_path).expanduser().resolve()
    mapping_file = Path(scored_mapping_path).expanduser().resolve()
    protocol = yaml.safe_load(protocol_file.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("Open ECG protocol must be a mapping.")
    if str(protocol.get("version")) != "0.4.0":
        raise ValueError("Two-source ECG label locking requires protocol v0.4.0.")
    if str(header_audit.get("protocol_version")) != str(protocol.get("version")):
        raise ValueError("Challenge label-support audit protocol version does not match current ECG protocol.")

    label_rules = protocol.get("prediction_task", {}).get("labels", {})
    inclusion = label_rules.get("inclusion_rules") if isinstance(label_rules, dict) else None
    canonical_protocol = label_rules.get("canonical_labels") if isinstance(label_rules, dict) else None
    if not isinstance(inclusion, dict) or not isinstance(canonical_protocol, dict):
        raise ValueError("ECG protocol label inclusion/canonical mapping rules are missing.")

    audit_labels = header_audit.get("labels")
    if not isinstance(audit_labels, list):
        raise ValueError("Challenge label-support audit labels must be a list.")
    eligible = [item for item in audit_labels if isinstance(item, dict) and item.get("eligible") is True]
    audit_by_code = {str(item.get("canonical_code", "")): item for item in eligible}
    eligible_codes = {str(code) for code in header_audit.get("eligible_labels", [])}
    if eligible_codes != set(audit_by_code) or len(eligible_codes) != len(eligible):
        raise ValueError("Challenge eligible-label summary is internally inconsistent.")

    concordance_rows = concordance.get("label_rows")
    if not isinstance(concordance_rows, list):
        raise ValueError("PTB-XL label-concordance rows must be a list.")
    concordance_by_code = {
        str(item.get("canonical_code", "")): item
        for item in concordance_rows
        if isinstance(item, dict)
    }
    expected_codes = set(str(code) for code in canonical_protocol)
    if eligible_codes != expected_codes or set(concordance_by_code) != expected_codes:
        raise RuntimeError("Development concordance, external label support, and protocol must agree on exactly seven labels.")

    labels = []
    for code in sorted(expected_codes, key=int):
        external_item = audit_by_code[code]
        development_item = concordance_by_code[code]
        protocol_item = canonical_protocol[code]
        scp_codes = [str(value) for value in development_item.get("scp_codes", [])]
        if scp_codes != [str(value) for value in protocol_item.get("ptbxl_scp_codes", [])]:
            raise ValueError(f"PTB-XL SCP mapping differs from frozen protocol for {code}.")
        development_count = int(development_item["original_ptbxl_union_key_present_count"])
        if development_count != int(external_item["development_positives"]):
            raise ValueError(f"Development aggregate count differs between evidence sources for {code}.")
        if development_item.get("exact_union_key_present_match") is not True:
            raise RuntimeError(f"PTB-XL label union is not exactly concordant for {code}.")
        labels.append(
            {
                "canonical_code": code,
                "abbreviation": str(external_item["abbreviation"]),
                "ptbxl_scp_codes": scp_codes,
                "challenge_member_codes": [str(value) for value in external_item["member_codes"]],
                "development_positives": development_count,
                "external_positives": {
                    str(name): int(count)
                    for name, count in sorted(dict(external_item["external_positives"]).items())
                },
                "external_domains_meeting_threshold": int(
                    external_item["external_domains_meeting_threshold"]
                ),
            }
        )

    manifest: dict[str, Any] = {
        "manifest_version": "0.2.0",
        "study": "TRUST-ECG",
        "status": "locked_before_waveform_model_training",
        "protocol_version": str(protocol["version"]),
        "development_source": "original_ptbxl_v1_0_1",
        "external_source": "challenge2020_v1_0_2_georgia_cpsc",
        "challenge_ptbxl_model_input": False,
        "label_count_semantics": "union_of_scp_key_presence_per_record",
        "source_header_audit_sha256": str(header_audit["manifest_sha256"]),
        "ptbxl_label_concordance_audit_sha256": str(concordance["audit_sha256"]),
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
        ptbxl_label_concordance_audit_sha256=str(
            manifest["ptbxl_label_concordance_audit_sha256"]
        ),
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
    if payload.get("challenge_ptbxl_model_input") is not False:
        raise ValueError("Locked ECG label manifest must prohibit Challenge PTB-XL model input.")
    return payload
