"""Fail-closed stage envelopes for the split TRUST-ECG statistical addendum."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from trust_icu.ecg_statistical_core import canonical_hash, json_ready

LOCKED_LABEL_CODES = (
    "59118001",
    "164889003",
    "164909002",
    "270492004",
    "284470004",
    "426783006",
    "427084000",
)

COMMON_IDENTITY_FIELDS = (
    "protocol_version",
    "protocol_sha256",
    "phase0_report_sha256",
    "phase0_model_sha256",
    "model_index_sha256",
    "label_manifest_sha256",
    "normalization_stats_sha256",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_hash(payload: dict[str, Any]) -> str:
    return canonical_hash(payload, "stage_sha256")


def write_stage_payload(path: str | Path, payload: dict[str, Any]) -> str:
    target = Path(path)
    material = dict(payload)
    material["stage_sha256"] = ""
    material["stage_sha256"] = _stage_hash(material)
    target.write_text(json.dumps(json_ready(material), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(material["stage_sha256"])


def load_stage_payload(path: str | Path, expected_stage: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("study") != "TRUST-ECG":
        raise RuntimeError("Unexpected study in statistical stage payload.")
    if payload.get("stage") != expected_stage:
        raise RuntimeError(f"Unexpected statistical stage: {payload.get('stage')!r}.")
    observed = str(payload.get("stage_sha256", ""))
    if not observed or _stage_hash(payload) != observed:
        raise RuntimeError(f"Statistical stage SHA-256 verification failed: {expected_stage}.")
    labels = tuple(str(code) for code in payload.get("label_codes", ()))
    if labels != LOCKED_LABEL_CODES:
        raise RuntimeError("Locked TRUST-ECG label order changed.")
    return payload


def verify_common_identity(left: dict[str, Any], right: dict[str, Any]) -> None:
    mismatches = [field for field in COMMON_IDENTITY_FIELDS if str(left.get(field)) != str(right.get(field))]
    if mismatches:
        raise RuntimeError("Cross-stage identity mismatch: " + ", ".join(mismatches))


def build_model_stage_payload(
    *,
    protocol_version: str,
    protocol_sha256: str,
    phase0_report_sha256: str,
    phase0_model_sha256: str,
    model_index_sha256: str,
    label_manifest_sha256: str,
    normalization_stats_sha256: str,
    bootstrap_repeats: int,
    model_results: dict[str, Any],
) -> dict[str, Any]:
    return {
        "study": "TRUST-ECG",
        "stage": "model_comparison_statistics",
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        "phase0_report_sha256": phase0_report_sha256,
        "phase0_model_sha256": phase0_model_sha256,
        "model_index_sha256": model_index_sha256,
        "label_manifest_sha256": label_manifest_sha256,
        "normalization_stats_sha256": normalization_stats_sha256,
        "label_codes": list(LOCKED_LABEL_CODES),
        "bootstrap_repeats": int(bootstrap_repeats),
        "paired_resnet_vs_logistic": model_results,
        "privacy": {"record_level_output_persisted": False},
        "stage_sha256": "",
    }


def build_phase1_stage_payload(
    *,
    protocol_version: str,
    protocol_sha256: str,
    phase0_report_sha256: str,
    phase0_model_sha256: str,
    model_index_sha256: str,
    label_manifest_sha256: str,
    normalization_stats_sha256: str,
    phase1_report_sha256: str,
    matched_results: dict[str, Any],
) -> dict[str, Any]:
    return {
        "study": "TRUST-ECG",
        "stage": "phase1_matched_statistics",
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        "phase0_report_sha256": phase0_report_sha256,
        "phase0_model_sha256": phase0_model_sha256,
        "model_index_sha256": model_index_sha256,
        "label_manifest_sha256": label_manifest_sha256,
        "normalization_stats_sha256": normalization_stats_sha256,
        "label_codes": list(LOCKED_LABEL_CODES),
        "phase1_report_sha256": phase1_report_sha256,
        "matched_phase1_method_comparisons": matched_results,
        "privacy": {"record_level_output_persisted": False},
        "stage_sha256": "",
    }
