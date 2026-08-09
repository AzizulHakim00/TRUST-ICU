#!/usr/bin/env python3
"""Validate a TRUST-ECG v0.4 fixed-ResNet activation manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

EXPECTED_STUDY = "TRUST-ECG"
EXPECTED_PROTOCOL_VERSION = "0.4.0"
EXPECTED_REFERENCE_MODEL = "logistic_reference"
EXPECTED_WORKFLOW_PATH = ".github/workflows/open-ecg-real-waveform-phase0-v04.yml"
EXPECTED_BRANCH = "open-ecg-transportability"

_HASH_FIELDS = (
    "logistic_report_sha256",
    "header_audit_sha256",
    "ptbxl_label_concordance_audit_sha256",
    "label_manifest_sha256",
    "waveform_audit_sha256",
    "normalization_stats_sha256",
    "ptbxl_assignment_sha256",
    "model_index_sha256",
    "model_index_audit_sha256",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_activation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless the reference run and v0.4 evidence are fully bound."""

    _expect(payload.get("activation_version") == "0.1.0", "Unexpected activation version.")
    _expect(payload.get("study") == EXPECTED_STUDY, "Activation study must be TRUST-ECG.")
    _expect(
        payload.get("protocol_version") == EXPECTED_PROTOCOL_VERSION,
        "Activation protocol version must be 0.4.0.",
    )
    _expect(
        payload.get("reference_model") == EXPECTED_REFERENCE_MODEL,
        "Activation must be based on the Logistic reference run.",
    )
    _expect(
        payload.get("logistic_workflow_path") == EXPECTED_WORKFLOW_PATH,
        "Activation points to an unexpected Logistic workflow.",
    )
    _expect(
        payload.get("logistic_branch") == EXPECTED_BRANCH,
        "Activation points to an unexpected branch.",
    )
    run_id = payload.get("logistic_workflow_run_id")
    _expect(isinstance(run_id, int) and run_id > 0, "Logistic workflow run ID must be positive.")
    _expect(payload.get("logistic_conclusion") == "success", "Logistic workflow did not succeed.")

    head_sha = str(payload.get("logistic_head_sha", ""))
    _expect(bool(_GIT_SHA.fullmatch(head_sha)), "Logistic head SHA must be a 40-character git SHA.")

    _expect(
        payload.get("logistic_primary_gate_eligible") is False,
        "The Logistic reference must remain ineligible for the primary gate.",
    )
    _expect(
        payload.get("external_recovery_pool_used") is False,
        "Recovery-pool data must remain untouched before fixed-ResNet Phase 0.",
    )
    _expect(payload.get("waveform_audit_ready") is True, "Waveform audit was not ready.")
    _expect(payload.get("model_index_ready") is True, "Model index was not ready.")
    _expect(
        payload.get("challenge_ptbxl_model_input") is False,
        "Challenge-renamed PTB-XL cannot be used as development model input.",
    )

    label_codes = payload.get("label_codes")
    _expect(
        isinstance(label_codes, list)
        and label_codes
        and all(isinstance(code, str) and code for code in label_codes),
        "Activation must contain a non-empty list of locked label codes.",
    )
    _expect(len(set(label_codes)) == len(label_codes), "Locked label codes must be unique.")

    for field in _HASH_FIELDS:
        value = str(payload.get(field, ""))
        _expect(bool(_SHA256.fullmatch(value)), f"{field} must be a lowercase SHA-256 digest.")

    return {
        "activation_version": payload["activation_version"],
        "study": payload["study"],
        "protocol_version": payload["protocol_version"],
        "logistic_workflow_run_id": run_id,
        "logistic_head_sha": head_sha,
        "label_codes": label_codes,
        "logistic_report_sha256": payload["logistic_report_sha256"],
        "label_manifest_sha256": payload["label_manifest_sha256"],
        "waveform_audit_sha256": payload["waveform_audit_sha256"],
        "normalization_stats_sha256": payload["normalization_stats_sha256"],
        "model_index_sha256": payload["model_index_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", help="Path to the completed Logistic activation manifest.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "resnet_activation_manifest_required",
                    "activation_version": "0.1.0",
                    "protocol_version": EXPECTED_PROTOCOL_VERSION,
                    "required_hash_fields": list(_HASH_FIELDS),
                    "required_reference_conclusion": "success",
                    "logistic_primary_gate_eligible_must_be": False,
                    "external_recovery_pool_used_must_be": False,
                    "challenge_ptbxl_model_input_must_be": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.activation:
        raise SystemExit("--activation is required unless --dry-run is used.")
    path = Path(args.activation).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Activation manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Activation manifest must contain one JSON object.")
    summary = validate_activation_payload(payload)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
