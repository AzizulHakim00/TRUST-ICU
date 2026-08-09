"""Secure local waveform-stage orchestration for the open TRUST-ECG study."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from trust_icu.ecg_crosswalk import resolve_ptbxl_checksum_crosswalk
from trust_icu.ecg_data import EXPECTED_SOURCES, parse_challenge_header, validate_ptbxl_folds
from trust_icu.ecg_manifest import load_and_verify_header_audit, load_and_verify_label_manifest
from trust_icu.ecg_signal import (
    NormalizationStats,
    StreamingLeadStats,
    parse_signal_header,
    standardize_signal,
    write_normalization_stats,
)

_FIT_FOLDS = (1, 2, 3, 4, 5, 6, 7)


@dataclass(frozen=True)
class PtbxlAssignment:
    challenge_record_id: str
    ecg_id: int
    strat_fold: int


@dataclass(frozen=True)
class EcgWaveformAudit:
    audit_version: str
    source_header_audit_sha256: str
    label_manifest_sha256: str
    label_count: int
    source_record_counts: dict[str, int]
    source_sampling_rates_hz: dict[str, list[float]]
    source_duration_seconds: dict[str, dict[str, float]]
    source_resampled_records: dict[str, int]
    source_cropped_records: dict[str, int]
    source_padded_records: dict[str, int]
    source_invalid_records: dict[str, int]
    source_header_corpus_sha256: dict[str, str]
    source_waveform_corpus_sha256: dict[str, str]
    ptbxl_assignment_sha256: str
    normalization_stats_sha256: str
    normalization_fit_folds: tuple[int, ...]
    ready_for_model_stage: bool
    blockers: tuple[str, ...]
    audit_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _numeric_record_id(record_id: str) -> int:
    match = re.search(r"(\d+)$", record_id)
    if not match:
        raise ValueError(f"Record ID has no terminal numeric component: {record_id!r}")
    return int(match.group(1))


def _canonical_hash(payload: dict[str, Any], hash_key: str) -> str:
    material = dict(payload)
    material[hash_key] = ""
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _corpus_hasher() -> Any:
    return hashlib.sha256()


def _update_corpus_hash(digest: Any, relative_path: str, raw: bytes) -> None:
    file_hash = hashlib.sha256(raw).hexdigest()
    digest.update(relative_path.encode("utf-8"))
    digest.update(b"\0")
    digest.update(file_hash.encode("ascii"))
    digest.update(b"\n")


def build_verified_ptbxl_assignments(
    *,
    challenge_ptbxl_root: str | Path,
    ptbxl_metadata_csv: str | Path,
    ptbxl_original_root: str | Path,
) -> tuple[PtbxlAssignment, ...]:
    """Build fold assignments from the same checksum-identity crosswalk used by the header gate."""

    root = Path(challenge_ptbxl_root).expanduser().resolve()
    challenge_records = [
        parse_challenge_header(path, source="ptb-xl") for path in sorted(root.rglob("*.hea"))
    ]
    resolved, crosswalk = resolve_ptbxl_checksum_crosswalk(
        challenge_records=challenge_records,
        metadata_csv=ptbxl_metadata_csv,
        original_ptbxl_root=ptbxl_original_root,
    )
    if crosswalk.get("valid") is not True:
        raise RuntimeError(f"PTB-XL assignment blocked by failed checksum crosswalk: {crosswalk}")
    folds = validate_ptbxl_folds(ptbxl_metadata_csv)
    if folds.get("valid") is not True:
        raise RuntimeError(f"PTB-XL assignment blocked by invalid patient-wise folds: {folds}")

    assignments = tuple(
        PtbxlAssignment(
            challenge_record_id=row.challenge_record_id,
            ecg_id=row.ecg_id,
            strat_fold=row.strat_fold,
        )
        for row in sorted(resolved, key=lambda row: _numeric_record_id(row.challenge_record_id))
    )
    if len(assignments) != len(challenge_records):
        raise RuntimeError("Checksum-resolved PTB-XL assignment does not cover every Challenge record.")
    if len({item.challenge_record_id for item in assignments}) != len(assignments):
        raise RuntimeError("Duplicate Challenge PTB-XL record IDs in verified assignment.")
    if len({item.ecg_id for item in assignments}) != len(assignments):
        raise RuntimeError("Duplicate original PTB-XL ecg_id values in verified assignment.")
    return assignments


def assignment_sha256(assignments: tuple[PtbxlAssignment, ...]) -> str:
    digest = hashlib.sha256()
    for item in sorted(assignments, key=lambda row: _numeric_record_id(row.challenge_record_id)):
        digest.update(f"{item.challenge_record_id},{item.ecg_id},{item.strat_fold}\n".encode())
    return digest.hexdigest()


def write_ptbxl_assignment(assignments: tuple[PtbxlAssignment, ...], output: str | Path) -> str:
    """Write the local record-to-fold map. It intentionally omits patient identifiers."""

    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["challenge_record_id", "ecg_id", "strat_fold"],
        )
        writer.writeheader()
        for item in assignments:
            writer.writerow(asdict(item))
    return assignment_sha256(assignments)


def _load_mat_from_raw(raw: bytes, path: Path) -> np.ndarray:
    payload = loadmat(io.BytesIO(raw))
    if "val" not in payload:
        raise ValueError(f"MAT file does not contain val: {path}")
    values = np.asarray(payload["val"], dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid digital ECG matrix: {path}")
    return values


def _source_header_paths(data_root: Path, source: str) -> list[Path]:
    source_root = data_root / source
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing Challenge source directory: {source_root}")
    return sorted(source_root.rglob("*.hea"), key=lambda path: (_numeric_record_id(path.stem), str(path)))


def prepare_waveform_stage(
    *,
    challenge_training_root: str | Path,
    ptbxl_metadata_csv: str | Path,
    ptbxl_original_root: str | Path,
    header_audit_path: str | Path,
    label_manifest_path: str | Path,
    output_root: str | Path,
) -> EcgWaveformAudit:
    """Audit every primary waveform and fit normalization on PTB-XL folds 1-7 only."""

    header_audit = load_and_verify_header_audit(header_audit_path)
    label_manifest = load_and_verify_label_manifest(label_manifest_path)
    if str(label_manifest["source_header_audit_sha256"]) != str(header_audit["manifest_sha256"]):
        raise ValueError("Label manifest is not tied to the supplied header audit.")

    data_root = Path(challenge_training_root).expanduser().resolve()
    out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    assignments = build_verified_ptbxl_assignments(
        challenge_ptbxl_root=data_root / "ptb-xl",
        ptbxl_metadata_csv=ptbxl_metadata_csv,
        ptbxl_original_root=ptbxl_original_root,
    )
    assignment_by_record = {item.challenge_record_id: item for item in assignments}
    assignment_hash = write_ptbxl_assignment(
        assignments,
        out_root / "ptbxl_verified_assignment.csv",
    )

    source_counts: Counter[str] = Counter()
    sampling_rates: dict[str, set[float]] = defaultdict(set)
    durations: dict[str, list[float]] = defaultdict(list)
    resampled = Counter()
    cropped = Counter()
    padded = Counter()
    invalid = Counter()
    header_hashers = {source: _corpus_hasher() for source in EXPECTED_SOURCES}
    waveform_hashers = {source: _corpus_hasher() for source in EXPECTED_SOURCES}
    stats_builder = StreamingLeadStats()
    fit_records = 0

    for source, expected_count in EXPECTED_SOURCES.items():
        paths = _source_header_paths(data_root, source)
        if len(paths) != expected_count:
            raise RuntimeError(
                f"Source {source} has {len(paths)} headers; expected {expected_count}. "
                "Refusing waveform execution on an incomplete or mixed release."
            )
        source_root = data_root / source
        for header_path in paths:
            source_counts[source] += 1
            relative_header = header_path.relative_to(source_root).as_posix()
            raw_header = header_path.read_bytes()
            _update_corpus_hash(header_hashers[source], relative_header, raw_header)
            mat_path = header_path.with_suffix(".mat")
            if not mat_path.is_file():
                invalid[source] += 1
                continue
            relative_mat = mat_path.relative_to(source_root).as_posix()
            raw_mat = mat_path.read_bytes()
            _update_corpus_hash(waveform_hashers[source], relative_mat, raw_mat)
            try:
                record_id, rate, header_samples, specs = parse_signal_header(header_path)
                digital = _load_mat_from_raw(raw_mat, mat_path)
                if digital.shape[1] != header_samples:
                    raise ValueError("MAT sample count differs from WFDB header.")
                standardized = standardize_signal(
                    digital,
                    specs,
                    source_sampling_rate_hz=rate,
                )
                sampling_rates[source].add(rate)
                durations[source].append(header_samples / rate)
                if abs(rate - standardized.target_sampling_rate_hz) > 1e-9:
                    resampled[source] += 1
                if standardized.crop_start_after_resampling is not None:
                    cropped[source] += 1
                if standardized.left_padding or standardized.right_padding:
                    padded[source] += 1
                if source == "ptb-xl":
                    assignment = assignment_by_record.get(record_id)
                    if assignment is None:
                        raise ValueError("PTB-XL record is absent from verified fold assignment.")
                    if assignment.strat_fold in _FIT_FOLDS:
                        stats_builder.update(standardized.waveform_mv, standardized.valid_mask)
                        fit_records += 1
            except (ValueError, RuntimeError, OSError):
                invalid[source] += 1

    blockers: list[str] = []
    if any(invalid.values()):
        blockers.append("invalid_or_missing_waveform_records")
    if fit_records == 0:
        blockers.append("no_ptbxl_fold_1_to_7_records_for_normalization")

    stats: NormalizationStats | None = None
    normalization_hash = ""
    if not blockers:
        stats = stats_builder.finalize(fit_folds=_FIT_FOLDS)
        write_normalization_stats(stats, out_root / "open_ecg_normalization_stats.json")
        normalization_hash = stats.stats_sha256

    duration_summary: dict[str, dict[str, float]] = {}
    for source in EXPECTED_SOURCES:
        values = durations[source]
        duration_summary[source] = {
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }

    payload: dict[str, Any] = {
        "audit_version": "0.1.0",
        "source_header_audit_sha256": str(header_audit["manifest_sha256"]),
        "label_manifest_sha256": str(label_manifest["manifest_sha256"]),
        "label_count": int(label_manifest["label_count"]),
        "source_record_counts": {source: source_counts[source] for source in EXPECTED_SOURCES},
        "source_sampling_rates_hz": {source: sorted(sampling_rates[source]) for source in EXPECTED_SOURCES},
        "source_duration_seconds": duration_summary,
        "source_resampled_records": {source: resampled[source] for source in EXPECTED_SOURCES},
        "source_cropped_records": {source: cropped[source] for source in EXPECTED_SOURCES},
        "source_padded_records": {source: padded[source] for source in EXPECTED_SOURCES},
        "source_invalid_records": {source: invalid[source] for source in EXPECTED_SOURCES},
        "source_header_corpus_sha256": {
            source: header_hashers[source].hexdigest() for source in EXPECTED_SOURCES
        },
        "source_waveform_corpus_sha256": {
            source: waveform_hashers[source].hexdigest() for source in EXPECTED_SOURCES
        },
        "ptbxl_assignment_sha256": assignment_hash,
        "normalization_stats_sha256": normalization_hash,
        "normalization_fit_folds": list(_FIT_FOLDS),
        "ready_for_model_stage": not blockers,
        "blockers": blockers,
        "audit_sha256": "",
    }
    payload["audit_sha256"] = _canonical_hash(payload, "audit_sha256")
    audit = EcgWaveformAudit(
        audit_version=str(payload["audit_version"]),
        source_header_audit_sha256=str(payload["source_header_audit_sha256"]),
        label_manifest_sha256=str(payload["label_manifest_sha256"]),
        label_count=int(payload["label_count"]),
        source_record_counts=payload["source_record_counts"],
        source_sampling_rates_hz=payload["source_sampling_rates_hz"],
        source_duration_seconds=payload["source_duration_seconds"],
        source_resampled_records=payload["source_resampled_records"],
        source_cropped_records=payload["source_cropped_records"],
        source_padded_records=payload["source_padded_records"],
        source_invalid_records=payload["source_invalid_records"],
        source_header_corpus_sha256=payload["source_header_corpus_sha256"],
        source_waveform_corpus_sha256=payload["source_waveform_corpus_sha256"],
        ptbxl_assignment_sha256=str(payload["ptbxl_assignment_sha256"]),
        normalization_stats_sha256=str(payload["normalization_stats_sha256"]),
        normalization_fit_folds=_FIT_FOLDS,
        ready_for_model_stage=bool(payload["ready_for_model_stage"]),
        blockers=tuple(payload["blockers"]),
        audit_sha256=str(payload["audit_sha256"]),
    )
    (out_root / "open_ecg_waveform_audit.json").write_text(
        json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def load_and_verify_waveform_audit(path: str | Path) -> dict[str, Any]:
    audit_path = Path(path).expanduser().resolve()
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ECG waveform audit must be a JSON object.")
    observed = str(payload.get("audit_sha256", ""))
    expected = _canonical_hash(payload, "audit_sha256")
    if not observed or observed != expected:
        raise ValueError("ECG waveform audit SHA-256 verification failed.")
    if payload.get("ready_for_model_stage") is not True:
        raise RuntimeError("ECG model stage is blocked by the waveform audit.")
    return payload
