"""Publication-safe reporting helpers for TRUST-ECG secondary analyses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_FORBIDDEN_PUBLIC_KEYS = {
    "record_id",
    "record_ids",
    "patient_id",
    "patient_ids",
    "raw_waveform",
    "raw_waveforms",
    "waveform",
    "waveforms",
    "record_level_probability",
    "record_level_probabilities",
    "record_level_prediction",
    "record_level_predictions",
    "record_level_attribution",
    "record_level_attributions",
    "patient_level_prediction",
    "patient_level_predictions",
}


def _require_sha256(value: Any, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest.")
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_report_hash(payload: dict[str, Any], key: str = "report_sha256") -> str:
    material = dict(payload)
    material[key] = ""
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_public_secondary_payload(payload: Any, *, path: str = "root") -> None:
    """Fail closed on record/patient-level material in publication payloads."""

    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"Public secondary payload contains forbidden {normalized!r} at {path}.")
            validate_public_secondary_payload(value, path=f"{path}.{key}")
        return
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            validate_public_secondary_payload(value, path=f"{path}[{index}]")
        return
    if payload is None or isinstance(payload, (str, int, float, bool)):
        return
    raise ValueError(
        f"Public secondary payload contains a non-JSON aggregate value at {path}: "
        f"{type(payload).__name__}."
    )


def validate_secondary_output_root(
    output_root: str | Path,
    *,
    primary_roots: list[str | Path] | tuple[str | Path, ...],
) -> Path:
    """Ensure secondary output cannot resolve inside any canonical primary result root."""

    output = Path(output_root).expanduser().resolve()
    for primary_root in primary_roots:
        primary = Path(primary_root).expanduser().resolve()
        if output == primary or primary in output.parents:
            raise ValueError("Secondary output root cannot be inside a primary results directory.")
    return output


def build_secondary_provenance(
    phase0_report: dict[str, Any],
    phase1_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind secondary outputs to the exact frozen primary report/model state."""

    if phase0_report.get("study") != "TRUST-ECG":
        raise ValueError("Secondary provenance requires a TRUST-ECG Phase-0 report.")
    phase0_protocol = _require_sha256(phase0_report.get("protocol_sha256"), "protocol_sha256")
    phase0_report_hash = _require_sha256(
        phase0_report.get("report_sha256"), "phase0_report_sha256"
    )
    model_hash = _require_sha256(phase0_report.get("model_sha256"), "phase0_model_sha256")
    model_index_hash = _require_sha256(
        phase0_report.get("model_index_sha256"), "model_index_sha256"
    )
    label_manifest_hash = _require_sha256(
        phase0_report.get("label_manifest_sha256"), "label_manifest_sha256"
    )

    provenance: dict[str, Any] = {
        "study": "TRUST-ECG",
        "analysis_role": "secondary_validation_only",
        "protocol_version": str(phase0_report.get("protocol_version")),
        "protocol_sha256": phase0_protocol,
        "primary_phase0_report_sha256": phase0_report_hash,
        "primary_model_sha256": model_hash,
        "model_index_sha256": model_index_hash,
        "label_manifest_sha256": label_manifest_hash,
        "primary_phase1_report_sha256": None,
    }

    if phase1_report is not None:
        if phase1_report.get("study") != "TRUST-ECG":
            raise ValueError("Secondary provenance requires a TRUST-ECG Phase-1 report.")
        if str(phase1_report.get("protocol_sha256")) != phase0_protocol:
            raise ValueError("Phase-0 and Phase-1 protocol hashes differ.")
        if str(phase1_report.get("phase0_report_sha256")) != phase0_report_hash:
            raise ValueError("Phase-1 does not reference the supplied Phase-0 report.")
        if str(phase1_report.get("phase0_model_sha256")) != model_hash:
            raise ValueError("Phase-1 does not reference the supplied primary model.")
        provenance["primary_phase1_report_sha256"] = _require_sha256(
            phase1_report.get("report_sha256"), "phase1_report_sha256"
        )
    return provenance


def load_and_verify_primary_reports(
    *,
    phase0_report_path: str | Path,
    phase1_report_path: str | Path | None,
    protocol_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """Load primary reports and verify canonical hashes plus cross-report provenance."""

    protocol = Path(protocol_path).expanduser().resolve()
    phase0_path = Path(phase0_report_path).expanduser().resolve()
    if not protocol.is_file() or not phase0_path.is_file():
        raise FileNotFoundError("Frozen TRUST-ECG protocol/Phase-0 report is missing.")

    phase0 = json.loads(phase0_path.read_text(encoding="utf-8"))
    if not isinstance(phase0, dict):
        raise ValueError("Phase-0 report must contain one JSON object.")
    if phase0.get("model_name") != "resnet1d_fixed" or phase0.get("primary_gate_eligible") is not True:
        raise ValueError("Secondary analysis requires the primary fixed-ResNet Phase-0 report.")
    if str(phase0.get("protocol_sha256")) != _sha256_file(protocol):
        raise ValueError("Phase-0 report does not match the supplied frozen protocol.")
    if str(phase0.get("report_sha256")) != _canonical_report_hash(phase0):
        raise ValueError("Phase-0 report SHA-256 verification failed.")

    phase1: dict[str, Any] | None = None
    if phase1_report_path is not None:
        phase1_path = Path(phase1_report_path).expanduser().resolve()
        if not phase1_path.is_file():
            raise FileNotFoundError("Frozen TRUST-ECG Phase-1 report is missing.")
        loaded = json.loads(phase1_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("Phase-1 report must contain one JSON object.")
        phase1 = loaded
        if phase1.get("stage") != "conditional_phase1_label_efficient_probability_recovery":
            raise ValueError("Unexpected TRUST-ECG Phase-1 report stage.")
        if str(phase1.get("report_sha256")) != _canonical_report_hash(phase1):
            raise ValueError("Phase-1 report SHA-256 verification failed.")

    provenance = build_secondary_provenance(phase0, phase1)
    return phase0, phase1, provenance


def _safe_output_name(name: str) -> str:
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in name):
        raise ValueError("Secondary output names may contain only lowercase letters, digits, '-' and '_'.")
    return name


def write_secondary_bundle(
    *,
    output_root: str | Path,
    payloads: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, str]:
    """Write deterministic aggregate JSON outputs and a SHA-256 manifest."""

    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    validate_public_secondary_payload(provenance)

    written: list[Path] = []
    provenance_path = output / "secondary_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    written.append(provenance_path)

    for name, payload in sorted(payloads.items()):
        safe_name = _safe_output_name(str(name))
        validate_public_secondary_payload(payload)
        path = output / f"{safe_name}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        written.append(path)

    manifest = {path.name: _sha256_file(path) for path in sorted(written)}
    manifest_path = output / "secondary_sha256_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
