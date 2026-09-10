"""Publication-safe aggregate reporting for TRUST-ECG secondary analyses."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trust_icu.ecg_secondary_sensitivity import (
    EnvelopeDefinition,
    classify_metric_payload,
    envelope_sensitivity_rows,
    framework_gate_ablation_rows,
)
from trust_icu.ecg_secondary_statistics import phase1_estimability_rows

_FORBIDDEN_KEYS = {
    "record_id",
    "record_ids",
    "patient_id",
    "patient_ids",
    "predictions",
    "logits",
    "waveform",
    "waveforms",
    "adaptation_indices",
}
_STATUS_ORDER = (
    "certified",
    "calibration_recovery_candidate",
    "discrimination_failure",
    "insufficient_support",
)


def _canonical_report_hash(payload: Mapping[str, Any]) -> tuple[str, str]:
    observed = str(payload.get("report_sha256", ""))
    material = dict(payload)
    material["report_sha256"] = ""
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return observed, hashlib.sha256(encoded).hexdigest()


def verify_embedded_report_hash(payload: Mapping[str, Any]) -> bool:
    """Verify a report's canonical embedded SHA-256 without mutating it."""

    observed, calculated = _canonical_report_hash(payload)
    return bool(observed) and observed == calculated


def _assert_aggregate_only(value: Any, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_KEYS:
                raise ValueError(f"Secondary aggregate payload contains forbidden key at {path}.{key}")
            _assert_aggregate_only(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_aggregate_only(child, path=f"{path}[{index}]")


def _official_status_counts(phase0_report: Mapping[str, Any]) -> dict[str, int]:
    external = phase0_report.get("external_certification")
    if not isinstance(external, Mapping) or not external:
        raise ValueError("Phase-0 report lacks external certification results.")
    counts = {status: 0 for status in _STATUS_ORDER}
    official = EnvelopeDefinition()
    for source, pairs in external.items():
        if not isinstance(pairs, Mapping):
            raise ValueError(f"Invalid Phase-0 source payload for {source}.")
        for label_code, pair in pairs.items():
            if not isinstance(pair, Mapping):
                raise ValueError(f"Invalid Phase-0 pair payload for {source}/{label_code}.")
            status = str(pair.get("status", ""))
            if status not in counts:
                raise ValueError(f"Unknown primary Phase-0 status: {status!r}")
            metrics = pair.get("metrics")
            if metrics is not None and not isinstance(metrics, Mapping):
                raise ValueError("Phase-0 pair metrics must be an object or null.")
            reproduced = classify_metric_payload(metrics, official)
            if reproduced != status:
                raise ValueError(
                    "Official secondary gate does not reproduce the frozen primary status "
                    f"for {source}/{label_code}: {reproduced!r} != {status!r}."
                )
            counts[status] += 1
    return counts


def build_secondary_aggregate_payload(
    phase0_report: Mapping[str, Any],
    phase1_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an aggregate-only secondary evidence payload bound to frozen reports."""

    if not verify_embedded_report_hash(phase0_report):
        raise ValueError("Phase-0 embedded report SHA-256 verification failed.")
    if not verify_embedded_report_hash(phase1_report):
        raise ValueError("Phase-1 embedded report SHA-256 verification failed.")

    payload: dict[str, Any] = {
        "study": "TRUST-ECG",
        "stage": "secondary_validation_aggregate_only",
        "primary_analysis_unchanged": True,
        "secondary_evidence_only": True,
        "primary_provenance": {
            "phase0_report_sha256": str(phase0_report["report_sha256"]),
            "phase1_report_sha256": str(phase1_report["report_sha256"]),
            "protocol_sha256": phase0_report.get("protocol_sha256"),
            "model_sha256": phase0_report.get("model_sha256"),
        },
        "official_status_counts": _official_status_counts(phase0_report),
        "envelope_sensitivity": envelope_sensitivity_rows(phase0_report),
        "framework_ablation": framework_gate_ablation_rows(phase0_report),
        "phase1_estimability": phase1_estimability_rows(phase1_report),
    }
    _assert_aggregate_only(payload)
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty aggregate CSV: {path.name}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for raw in rows:
        for key in raw:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for raw in rows:
            row = dict(raw)
            for key, value in row.items():
                if isinstance(value, (dict, list, tuple)):
                    row[key] = json.dumps(
                        value,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
            writer.writerow(row)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_secondary_aggregate_package(
    phase0_report: Mapping[str, Any],
    phase1_report: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    """Write manifest-tracked aggregate secondary files without primary mutation."""

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = build_secondary_aggregate_payload(phase0_report, phase1_report)

    paths = {
        "secondary_summary.json": root / "secondary_summary.json",
        "envelope_sensitivity.csv": root / "envelope_sensitivity.csv",
        "framework_ablation.csv": root / "framework_ablation.csv",
        "phase1_estimability.csv": root / "phase1_estimability.csv",
    }
    _write_json(paths["secondary_summary.json"], payload)
    _write_csv(paths["envelope_sensitivity.csv"], payload["envelope_sensitivity"])
    _write_csv(paths["framework_ablation.csv"], payload["framework_ablation"])
    _write_csv(paths["phase1_estimability.csv"], payload["phase1_estimability"])

    manifest = {name: _sha256(path) for name, path in sorted(paths.items())}
    manifest_path = root / "secondary_sha256_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "output_root": str(root),
        "manifest_entries": len(manifest),
        "manifest_sha256": _sha256(manifest_path),
        "primary_analysis_unchanged": True,
    }
