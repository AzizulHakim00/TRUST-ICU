"""Secure waveform-stage orchestration for TRUST-ECG protocol v0.4.

The unified primary data root contains original PTB-XL v1.0.1 under ``ptb-xl/`` and the three
Challenge external sources under their source names. Challenge-renamed PTB-XL records are never
loaded as model inputs. Row-level PTB assignments remain local; public outputs are aggregate-only.
"""

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

from trust_icu.ecg_data import EXPECTED_SOURCES
from trust_icu.ecg_manifest import load_and_verify_header_audit, load_and_verify_label_manifest
from trust_icu.ecg_signal import (
    NormalizationStats,
    StreamingLeadStats,
    load_wfdb_physical_signal,
    parse_signal_header,
    standardize_physical_signal,
    standardize_signal,
    write_normalization_stats,
)

_FIT_FOLDS = (1, 2, 3, 4, 5, 6, 7)
_EXTERNAL_SOURCES = ("georgia", "cpsc_2018", "cpsc_2018_extra")


@dataclass(frozen=True)
class PtbxlAssignment:
    record_id: str
    ecg_id: int
    strat_fold: int
    relative_header_path: str
    relative_waveform_path: str


@dataclass(frozen=True)
class EcgWaveformAudit:
    audit_version: str
    development_source: str
    challenge_ptbxl_model_input: bool
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
    source_waveform_format: dict[str, str]
    ptbxl_assignment_sha256: str
    normalization_stats_sha256: str
    normalization_fit_folds: tuple[int, ...]
    ready_for_model_stage: bool
    blockers: tuple[str, ...]
    audit_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _numeric_record_id(record_id: str) -> int:
    match = re.search(r"(\d+)(?:_hr)?$", record_id)
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


def _safe_relative_stem(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe PTB-XL filename_hr value: {value!r}")
    return path


def _read_ptbxl_metadata(path: str | Path) -> list[dict[str, str]]:
    metadata_path = Path(path).expanduser().resolve()
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"ecg_id", "patient_id", "strat_fold", "filename_hr", "scp_codes"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("PTB-XL metadata is missing required development columns.")
        rows = [dict(row) for row in reader]
    if len(rows) != EXPECTED_SOURCES["ptb-xl"]:
        raise ValueError(
            f"PTB-XL metadata has {len(rows)} rows; expected {EXPECTED_SOURCES['ptb-xl']}."
        )
    if len({int(row["ecg_id"]) for row in rows}) != len(rows):
        raise ValueError("PTB-XL metadata contains duplicate ecg_id values.")
    patient_folds: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        fold = int(row["strat_fold"])
        if fold not in range(1, 11):
            raise ValueError("PTB-XL metadata contains a fold outside 1..10.")
        patient_folds[str(row["patient_id"])].add(fold)
    if any(len(folds) != 1 for folds in patient_folds.values()):
        raise ValueError("PTB-XL metadata violates patient-wise fold separation.")
    return rows


def build_verified_ptbxl_assignments(
    *,
    primary_data_root: str | Path,
    ptbxl_metadata_csv: str | Path,
) -> tuple[PtbxlAssignment, ...]:
    """Build original PTB-XL record-to-fold assignments directly from official metadata."""

    root = Path(primary_data_root).expanduser().resolve()
    ptb_root = root / "ptb-xl"
    if not ptb_root.is_dir():
        raise FileNotFoundError(f"Missing original PTB-XL source root: {ptb_root}")
    rows = _read_ptbxl_metadata(ptbxl_metadata_csv)
    assignments: list[PtbxlAssignment] = []
    seen_record_ids: set[str] = set()
    for row in rows:
        relative_stem = _safe_relative_stem(str(row["filename_hr"]))
        header_path = (ptb_root / relative_stem).with_suffix(".hea").resolve()
        waveform_path = (ptb_root / relative_stem).with_suffix(".dat").resolve()
        if ptb_root not in header_path.parents or ptb_root not in waveform_path.parents:
            raise ValueError("PTB-XL metadata path escaped the declared source root.")
        if not header_path.is_file() or not waveform_path.is_file():
            raise FileNotFoundError(f"Incomplete original PTB-XL WFDB record: {relative_stem}")
        record_id, rate, samples, _ = parse_signal_header(header_path)
        if rate != 500.0 or samples != 5000:
            raise ValueError(f"Original PTB-XL records500 contract drift detected: {header_path}")
        if record_id in seen_record_ids:
            raise ValueError("Original PTB-XL contains duplicate WFDB record IDs.")
        seen_record_ids.add(record_id)
        assignments.append(
            PtbxlAssignment(
                record_id=record_id,
                ecg_id=int(row["ecg_id"]),
                strat_fold=int(row["strat_fold"]),
                relative_header_path=header_path.relative_to(root).as_posix(),
                relative_waveform_path=waveform_path.relative_to(root).as_posix(),
            )
        )
    return tuple(sorted(assignments, key=lambda item: item.ecg_id))


def assignment_sha256(assignments: tuple[PtbxlAssignment, ...]) -> str:
    digest = hashlib.sha256()
    for item in sorted(assignments, key=lambda row: row.ecg_id):
        digest.update(
            (
                f"{item.record_id},{item.ecg_id},{item.strat_fold},"
                f"{item.relative_header_path},{item.relative_waveform_path}\n"
            ).encode()
        )
    return digest.hexdigest()


def write_ptbxl_assignment(assignments: tuple[PtbxlAssignment, ...], output: str | Path) -> str:
    """Write the local original-PTB-XL record/fold map without patient identifiers."""

    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_id",
                "ecg_id",
                "strat_fold",
                "relative_header_path",
                "relative_waveform_path",
            ],
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


def _external_header_paths(data_root: Path, source: str) -> list[Path]:
    source_root = data_root / source
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing Challenge external source directory: {source_root}")
    return sorted(source_root.rglob("*.hea"), key=lambda path: (_numeric_record_id(path.stem), str(path)))


def prepare_waveform_stage(
    *,
    primary_data_root: str | Path,
    ptbxl_metadata_csv: str | Path,
    header_audit_path: str | Path,
    label_manifest_path: str | Path,
    output_root: str | Path,
) -> EcgWaveformAudit:
    """Audit original PTB-XL + three Challenge external waveform corpora before modeling."""

    header_audit = load_and_verify_header_audit(header_audit_path)
    label_manifest = load_and_verify_label_manifest(label_manifest_path)
    if str(label_manifest["source_header_audit_sha256"]) != str(header_audit["manifest_sha256"]):
        raise ValueError("Label manifest is not tied to the supplied Challenge label-support audit.")
    if label_manifest.get("development_source") != "original_ptbxl_v1_0_1":
        raise ValueError("Waveform stage requires original PTB-XL v1.0.1 as development source.")
    if label_manifest.get("challenge_ptbxl_model_input") is not False:
        raise ValueError("Challenge PTB-XL is prohibited as a model input under protocol v0.4.")

    data_root = Path(primary_data_root).expanduser().resolve()
    out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    assignments = build_verified_ptbxl_assignments(
        primary_data_root=data_root,
        ptbxl_metadata_csv=ptbxl_metadata_csv,
    )
    assignment_hash = write_ptbxl_assignment(assignments, out_root / "ptbxl_verified_assignment.csv")

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

    # Development source: original PTB-XL WFDB .hea + .dat, never Challenge-renamed PTB-XL.
    for assignment in assignments:
        source = "ptb-xl"
        header_path = data_root / assignment.relative_header_path
        waveform_path = data_root / assignment.relative_waveform_path
        source_root = data_root / source
        source_counts[source] += 1
        _update_corpus_hash(
            header_hashers[source],
            header_path.relative_to(source_root).as_posix(),
            header_path.read_bytes(),
        )
        _update_corpus_hash(
            waveform_hashers[source],
            waveform_path.relative_to(source_root).as_posix(),
            waveform_path.read_bytes(),
        )
        try:
            physical, leads, rate = load_wfdb_physical_signal(waveform_path.with_suffix(""))
            standardized = standardize_physical_signal(
                physical,
                leads,
                source_sampling_rate_hz=rate,
            )
            sampling_rates[source].add(rate)
            durations[source].append(standardized.source_sample_count / rate)
            if abs(rate - standardized.target_sampling_rate_hz) > 1e-9:
                resampled[source] += 1
            if standardized.crop_start_after_resampling is not None:
                cropped[source] += 1
            if standardized.left_padding or standardized.right_padding:
                padded[source] += 1
            if assignment.strat_fold in _FIT_FOLDS:
                stats_builder.update(standardized.waveform_mv, standardized.valid_mask)
                fit_records += 1
        except (ValueError, RuntimeError, OSError):
            invalid[source] += 1

    # External sources: Challenge WFDB headers paired with MATLAB-v4 waveform matrices.
    for source in _EXTERNAL_SOURCES:
        expected_count = EXPECTED_SOURCES[source]
        paths = _external_header_paths(data_root, source)
        if len(paths) != expected_count:
            raise RuntimeError(
                f"Source {source} has {len(paths)} headers; expected {expected_count}. "
                "Refusing waveform execution on an incomplete or mixed release."
            )
        source_root = data_root / source
        for header_path in paths:
            source_counts[source] += 1
            raw_header = header_path.read_bytes()
            _update_corpus_hash(
                header_hashers[source],
                header_path.relative_to(source_root).as_posix(),
                raw_header,
            )
            mat_path = header_path.with_suffix(".mat")
            if not mat_path.is_file():
                invalid[source] += 1
                continue
            raw_mat = mat_path.read_bytes()
            _update_corpus_hash(
                waveform_hashers[source],
                mat_path.relative_to(source_root).as_posix(),
                raw_mat,
            )
            try:
                _, rate, header_samples, specs = parse_signal_header(header_path)
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
            except (ValueError, RuntimeError, OSError):
                invalid[source] += 1

    blockers: list[str] = []
    for source, expected_count in EXPECTED_SOURCES.items():
        if source_counts[source] != expected_count:
            blockers.append(f"source_count_mismatch:{source}")
    if any(invalid.values()):
        blockers.append("invalid_or_missing_waveform_records")
    if fit_records == 0:
        blockers.append("no_ptbxl_fold_1_to_7_records_for_normalization")

    normalization_hash = ""
    if not blockers:
        stats: NormalizationStats = stats_builder.finalize(fit_folds=_FIT_FOLDS)
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
        "audit_version": "0.2.0",
        "development_source": "original_ptbxl_v1_0_1",
        "challenge_ptbxl_model_input": False,
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
        "source_waveform_format": {
            "ptb-xl": "wfdb_dat_original_ptbxl_v1_0_1",
            "georgia": "challenge_mat_v4",
            "cpsc_2018": "challenge_mat_v4",
            "cpsc_2018_extra": "challenge_mat_v4",
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
        development_source=str(payload["development_source"]),
        challenge_ptbxl_model_input=bool(payload["challenge_ptbxl_model_input"]),
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
        source_waveform_format=payload["source_waveform_format"],
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
    if payload.get("development_source") != "original_ptbxl_v1_0_1":
        raise ValueError("ECG waveform audit does not use the protocol v0.4 development source.")
    if payload.get("challenge_ptbxl_model_input") is not False:
        raise ValueError("ECG waveform audit illegally includes Challenge PTB-XL as model input.")
    return payload
