"""Tamper-evident source-aware model index for TRUST-ECG protocol v0.4."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from trust_icu.ecg_baseline import external_partition
from trust_icu.ecg_data import EXPECTED_SOURCES, parse_challenge_header
from trust_icu.ecg_manifest import load_and_verify_label_manifest
from trust_icu.ecg_waveform import (
    PtbxlAssignment,
    assignment_sha256,
    load_and_verify_waveform_audit,
)

Role = Literal[
    "model_fit",
    "optimization_validation",
    "calibration",
    "internal_test",
    "external_certification",
    "external_recovery_pool",
]

_EXTERNAL_SOURCES = ("georgia", "cpsc_2018", "cpsc_2018_extra")
_SOURCE_FORMATS = {
    "ptb-xl": "wfdb_dat_original_ptbxl_v1_0_1",
    "georgia": "challenge_mat_v4",
    "cpsc_2018": "challenge_mat_v4",
    "cpsc_2018_extra": "challenge_mat_v4",
}


@dataclass(frozen=True)
class EcgIndexRow:
    source: str
    record_id: str
    relative_header_path: str
    relative_waveform_path: str
    waveform_format: str
    role: Role
    strat_fold: int | None
    labels: tuple[int, ...]


@dataclass(frozen=True)
class EcgIndexAudit:
    audit_version: str
    development_source: str
    challenge_ptbxl_model_input: bool
    waveform_audit_sha256: str
    label_manifest_sha256: str
    label_codes: tuple[str, ...]
    total_rows: int
    source_rows: dict[str, int]
    role_rows: dict[str, int]
    source_role_rows: dict[str, dict[str, int]]
    source_label_positives: dict[str, dict[str, int]]
    source_waveform_format: dict[str, str]
    rows_with_no_locked_positive_label: int
    duplicate_source_record_ids: int
    corpus_hashes_verified: bool
    index_sha256: str
    ready_for_baseline_execution: bool
    blockers: tuple[str, ...]
    audit_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_hash(payload: dict[str, Any], key: str) -> str:
    material = dict(payload)
    material[key] = ""
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _numeric_record_id(record_id: str) -> int:
    match = re.search(r"(\d+)(?:_hr)?$", record_id)
    if not match:
        raise ValueError(f"Record ID has no terminal numeric component: {record_id!r}")
    return int(match.group(1))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_files(source_root: Path, paths: list[Path]) -> str:
    """Hash a source corpus in the same canonical record order as the waveform audit.

    Challenge sources are stored in nested ``g1/``, ``g2/``, ... directories. A plain
    lexicographic path sort therefore places ``g10`` before ``g2`` and can produce a
    different digest from the waveform audit even when every file is byte-identical.
    Ordering by the terminal numeric record ID (then relative path as a deterministic
    tie-breaker) matches the audit traversal while preserving byte-level tamper detection.
    """

    digest = hashlib.sha256()
    ordered_paths = sorted(
        paths,
        key=lambda item: (
            _numeric_record_id(item.stem),
            item.relative_to(source_root).as_posix(),
        ),
    )
    for path in ordered_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing corpus file: {path}")
        relative = path.relative_to(source_root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(_file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_assignment(path: str | Path) -> tuple[PtbxlAssignment, ...]:
    assignment_path = Path(path).expanduser().resolve()
    with assignment_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "record_id",
            "ecg_id",
            "strat_fold",
            "relative_header_path",
            "relative_waveform_path",
        }
        if set(reader.fieldnames or []) != required:
            raise ValueError("Verified original PTB-XL assignment has unexpected columns.")
        rows = tuple(
            PtbxlAssignment(
                record_id=str(row["record_id"]),
                ecg_id=int(row["ecg_id"]),
                strat_fold=int(row["strat_fold"]),
                relative_header_path=str(row["relative_header_path"]),
                relative_waveform_path=str(row["relative_waveform_path"]),
            )
            for row in reader
        )
    if not rows:
        raise ValueError("Verified PTB-XL assignment is empty.")
    if len({row.record_id for row in rows}) != len(rows):
        raise ValueError("Verified PTB-XL assignment contains duplicate record IDs.")
    if len({row.ecg_id for row in rows}) != len(rows):
        raise ValueError("Verified PTB-XL assignment contains duplicate ecg_id values.")
    if any(row.strat_fold not in range(1, 11) for row in rows):
        raise ValueError("Verified PTB-XL assignment contains a fold outside 1..10.")
    return rows


def _ptb_role(fold: int) -> Role:
    if fold in range(1, 8):
        return "model_fit"
    if fold == 8:
        return "optimization_validation"
    if fold == 9:
        return "calibration"
    if fold == 10:
        return "internal_test"
    raise ValueError(f"Unsupported PTB-XL fold: {fold}")


def _parse_scp_codes(raw: str) -> set[str]:
    parsed = ast.literal_eval(raw)
    if not isinstance(parsed, dict):
        raise ValueError("PTB-XL scp_codes must parse to a dictionary.")
    return {str(code) for code in parsed}


def _read_ptb_metadata(path: str | Path) -> dict[int, set[str]]:
    metadata_path = Path(path).expanduser().resolve()
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"ecg_id", "scp_codes"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("PTB-XL metadata lacks ecg_id or scp_codes.")
        output: dict[int, set[str]] = {}
        for row in reader:
            ecg_id = int(row["ecg_id"])
            if ecg_id in output:
                raise ValueError("PTB-XL metadata contains duplicate ecg_id values.")
            output[ecg_id] = _parse_scp_codes(str(row["scp_codes"]))
    if not output:
        raise ValueError("PTB-XL metadata is empty.")
    return output


def _locked_label_maps(
    manifest: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], dict[str, int]]:
    labels = manifest.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("Locked ECG label manifest has no labels.")
    canonical: list[str] = []
    ptb_groups: list[tuple[str, ...]] = []
    challenge_members: dict[str, int] = {}
    for index, item in enumerate(labels):
        if not isinstance(item, dict):
            raise ValueError("Locked ECG label manifest contains a malformed entry.")
        code = str(item["canonical_code"])
        canonical.append(code)
        ptb = tuple(str(value) for value in item.get("ptbxl_scp_codes", []))
        challenge = tuple(str(value) for value in item.get("challenge_member_codes", []))
        if not ptb or len(set(ptb)) != len(ptb):
            raise ValueError(f"Invalid PTB-XL SCP mapping for {code}.")
        if not challenge:
            raise ValueError(f"Missing Challenge SNOMED mapping for {code}.")
        ptb_groups.append(ptb)
        for member in challenge:
            previous = challenge_members.setdefault(member, index)
            if previous != index:
                raise ValueError(f"Challenge diagnosis code {member} maps to multiple labels.")
    if len(set(canonical)) != len(canonical):
        raise ValueError("Locked ECG label manifest contains duplicate canonical codes.")
    return tuple(canonical), tuple(ptb_groups), challenge_members


def _ptb_label_vector(present: set[str], groups: tuple[tuple[str, ...], ...]) -> tuple[int, ...]:
    return tuple(int(bool(present.intersection(group))) for group in groups)


def _challenge_label_vector(
    dx_codes: tuple[str, ...],
    *,
    n_labels: int,
    member_to_index: dict[str, int],
) -> tuple[int, ...]:
    target = [0] * n_labels
    for code in dx_codes:
        index = member_to_index.get(str(code))
        if index is not None:
            target[index] = 1
    return tuple(target)


def model_index_sha256(rows: list[EcgIndexRow], label_codes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(("labels=" + ",".join(label_codes) + "\n").encode())
    for row in sorted(rows, key=lambda item: (item.source, _numeric_record_id(item.record_id))):
        fold = "" if row.strat_fold is None else str(row.strat_fold)
        bits = "".join(str(value) for value in row.labels)
        digest.update(
            (
                f"{row.source}|{row.record_id}|{row.relative_header_path}|"
                f"{row.relative_waveform_path}|{row.waveform_format}|{row.role}|{fold}|{bits}\n"
            ).encode()
        )
    return digest.hexdigest()


def _count_row(
    row: EcgIndexRow,
    *,
    source_counts: Counter[str],
    role_counts: Counter[str],
    source_role_counts: dict[str, Counter[str]],
    source_label_counts: dict[str, Counter[str]],
    label_codes: tuple[str, ...],
) -> int:
    source_counts[row.source] += 1
    role_counts[row.role] += 1
    source_role_counts[row.source][row.role] += 1
    for index, value in enumerate(row.labels):
        if value:
            source_label_counts[row.source][label_codes[index]] += 1
    return int(not any(row.labels))


def build_model_index(
    *,
    primary_data_root: str | Path,
    ptbxl_metadata_path: str | Path,
    waveform_audit_path: str | Path,
    label_manifest_path: str | Path,
    ptbxl_assignment_path: str | Path,
) -> tuple[list[EcgIndexRow], EcgIndexAudit]:
    """Build the complete v0.4 index before any baseline execution."""

    waveform_audit = load_and_verify_waveform_audit(waveform_audit_path)
    manifest = load_and_verify_label_manifest(label_manifest_path)
    if waveform_audit.get("development_source") != "original_ptbxl_v1_0_1":
        raise ValueError("Model index requires original PTB-XL v1.0.1 development data.")
    if waveform_audit.get("challenge_ptbxl_model_input") is not False:
        raise ValueError("Model index refuses Challenge PTB-XL model input.")
    if str(waveform_audit["label_manifest_sha256"]) != str(manifest["manifest_sha256"]):
        raise ValueError("Waveform audit and label manifest do not share the same locked state.")

    assignments = _load_assignment(ptbxl_assignment_path)
    if assignment_sha256(assignments) != str(waveform_audit["ptbxl_assignment_sha256"]):
        raise ValueError("PTB-XL assignment hash does not match the waveform audit.")
    ptb_metadata = _read_ptb_metadata(ptbxl_metadata_path)
    if set(ptb_metadata) != {row.ecg_id for row in assignments}:
        raise ValueError("PTB-XL metadata and assignment do not cover the same ecg_id set.")

    label_codes, ptb_groups, challenge_members = _locked_label_maps(manifest)
    root = Path(primary_data_root).expanduser().resolve()
    rows: list[EcgIndexRow] = []
    seen: set[tuple[str, str]] = set()
    duplicate_count = 0
    source_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    source_role_counts = {source: Counter() for source in EXPECTED_SOURCES}
    source_label_counts = {source: Counter() for source in EXPECTED_SOURCES}
    rows_without_label = 0
    corpus_hashes_verified = True

    ptb_root = root / "ptb-xl"
    ptb_headers: list[Path] = []
    ptb_waveforms: list[Path] = []
    for assignment in assignments:
        header_path = (root / assignment.relative_header_path).resolve()
        waveform_path = (root / assignment.relative_waveform_path).resolve()
        if root not in header_path.parents or root not in waveform_path.parents:
            raise ValueError("PTB-XL assignment escaped the primary data root.")
        if not header_path.is_file() or not waveform_path.is_file():
            raise FileNotFoundError(f"Incomplete PTB-XL record: {assignment.record_id}")
        key = ("ptb-xl", assignment.record_id)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        row = EcgIndexRow(
            source="ptb-xl",
            record_id=assignment.record_id,
            relative_header_path=assignment.relative_header_path,
            relative_waveform_path=assignment.relative_waveform_path,
            waveform_format=_SOURCE_FORMATS["ptb-xl"],
            role=_ptb_role(assignment.strat_fold),
            strat_fold=assignment.strat_fold,
            labels=_ptb_label_vector(ptb_metadata[assignment.ecg_id], ptb_groups),
        )
        rows.append(row)
        ptb_headers.append(header_path)
        ptb_waveforms.append(waveform_path)
        rows_without_label += _count_row(
            row,
            source_counts=source_counts,
            role_counts=role_counts,
            source_role_counts=source_role_counts,
            source_label_counts=source_label_counts,
            label_codes=label_codes,
        )

    if _hash_files(ptb_root, ptb_headers) != str(
        waveform_audit["source_header_corpus_sha256"]["ptb-xl"]
    ):
        corpus_hashes_verified = False
    if _hash_files(ptb_root, ptb_waveforms) != str(
        waveform_audit["source_waveform_corpus_sha256"]["ptb-xl"]
    ):
        corpus_hashes_verified = False

    for source in _EXTERNAL_SOURCES:
        source_root = root / source
        if not source_root.is_dir():
            raise FileNotFoundError(f"Missing external source directory: {source_root}")
        headers = sorted(
            source_root.rglob("*.hea"),
            key=lambda path: (_numeric_record_id(path.stem), str(path)),
        )
        if len(headers) != EXPECTED_SOURCES[source]:
            raise RuntimeError(
                f"Source {source} has {len(headers)} headers; expected {EXPECTED_SOURCES[source]}."
            )
        waveforms = [path.with_suffix(".mat") for path in headers]
        if _hash_files(source_root, headers) != str(
            waveform_audit["source_header_corpus_sha256"][source]
        ):
            corpus_hashes_verified = False
        if _hash_files(source_root, waveforms) != str(
            waveform_audit["source_waveform_corpus_sha256"][source]
        ):
            corpus_hashes_verified = False

        for header_path, waveform_path in zip(headers, waveforms, strict=True):
            parsed = parse_challenge_header(header_path, source=source)
            key = (source, parsed.record_id)
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            partition = external_partition(source=source, record_id=parsed.record_id)
            role: Role = (
                "external_certification" if partition == "certification" else "external_recovery_pool"
            )
            row = EcgIndexRow(
                source=source,
                record_id=parsed.record_id,
                relative_header_path=header_path.relative_to(root).as_posix(),
                relative_waveform_path=waveform_path.relative_to(root).as_posix(),
                waveform_format=_SOURCE_FORMATS[source],
                role=role,
                strat_fold=None,
                labels=_challenge_label_vector(
                    parsed.dx_codes,
                    n_labels=len(label_codes),
                    member_to_index=challenge_members,
                ),
            )
            rows.append(row)
            rows_without_label += _count_row(
                row,
                source_counts=source_counts,
                role_counts=role_counts,
                source_role_counts=source_role_counts,
                source_label_counts=source_label_counts,
                label_codes=label_codes,
            )

    blockers: list[str] = []
    if duplicate_count:
        blockers.append("duplicate_source_record_ids")
    if not corpus_hashes_verified:
        blockers.append("corpus_hash_changed_after_waveform_audit")
    for source, expected_count in EXPECTED_SOURCES.items():
        if source_counts[source] != expected_count:
            blockers.append(f"source_count_mismatch:{source}")
    for role in ("model_fit", "optimization_validation", "calibration", "internal_test"):
        if role_counts[role] == 0:
            blockers.append(f"development_role_empty:{role}")
    for source in _EXTERNAL_SOURCES:
        if source_role_counts[source]["external_certification"] == 0:
            blockers.append(f"external_certification_partition_empty:{source}")
        if source_role_counts[source]["external_recovery_pool"] == 0:
            blockers.append(f"external_recovery_partition_empty:{source}")
    if dict(waveform_audit.get("source_waveform_format", {})) != _SOURCE_FORMATS:
        blockers.append("waveform_format_contract_mismatch")

    index_hash = model_index_sha256(rows, label_codes)
    payload: dict[str, Any] = {
        "audit_version": "0.2.0",
        "development_source": "original_ptbxl_v1_0_1",
        "challenge_ptbxl_model_input": False,
        "waveform_audit_sha256": str(waveform_audit["audit_sha256"]),
        "label_manifest_sha256": str(manifest["manifest_sha256"]),
        "label_codes": list(label_codes),
        "total_rows": len(rows),
        "source_rows": {source: source_counts[source] for source in EXPECTED_SOURCES},
        "role_rows": dict(sorted(role_counts.items())),
        "source_role_rows": {
            source: dict(sorted(source_role_counts[source].items())) for source in EXPECTED_SOURCES
        },
        "source_label_positives": {
            source: {code: source_label_counts[source][code] for code in label_codes}
            for source in EXPECTED_SOURCES
        },
        "source_waveform_format": dict(_SOURCE_FORMATS),
        "rows_with_no_locked_positive_label": rows_without_label,
        "duplicate_source_record_ids": duplicate_count,
        "corpus_hashes_verified": corpus_hashes_verified,
        "index_sha256": index_hash,
        "ready_for_baseline_execution": not blockers,
        "blockers": blockers,
        "audit_sha256": "",
    }
    payload["audit_sha256"] = _canonical_hash(payload, "audit_sha256")
    audit = EcgIndexAudit(
        audit_version="0.2.0",
        development_source="original_ptbxl_v1_0_1",
        challenge_ptbxl_model_input=False,
        waveform_audit_sha256=str(payload["waveform_audit_sha256"]),
        label_manifest_sha256=str(payload["label_manifest_sha256"]),
        label_codes=label_codes,
        total_rows=len(rows),
        source_rows=payload["source_rows"],
        role_rows=payload["role_rows"],
        source_role_rows=payload["source_role_rows"],
        source_label_positives=payload["source_label_positives"],
        source_waveform_format=dict(_SOURCE_FORMATS),
        rows_with_no_locked_positive_label=rows_without_label,
        duplicate_source_record_ids=duplicate_count,
        corpus_hashes_verified=corpus_hashes_verified,
        index_sha256=index_hash,
        ready_for_baseline_execution=not blockers,
        blockers=tuple(blockers),
        audit_sha256=str(payload["audit_sha256"]),
    )
    return rows, audit


def write_model_index(
    rows: list[EcgIndexRow],
    audit: EcgIndexAudit,
    *,
    index_output: str | Path,
    audit_output: str | Path,
) -> None:
    """Write the local row-level index and aggregate audit."""

    index_path = Path(index_output).expanduser().resolve()
    audit_path = Path(audit_output).expanduser().resolve()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    label_columns = [f"label_{code}" for code in audit.label_codes]
    fieldnames = [
        "source",
        "record_id",
        "relative_header_path",
        "relative_waveform_path",
        "waveform_format",
        "role",
        "strat_fold",
        *label_columns,
    ]
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload: dict[str, Any] = {
                "source": row.source,
                "record_id": row.record_id,
                "relative_header_path": row.relative_header_path,
                "relative_waveform_path": row.relative_waveform_path,
                "waveform_format": row.waveform_format,
                "role": row.role,
                "strat_fold": "" if row.strat_fold is None else row.strat_fold,
            }
            payload.update(
                {column: row.labels[index] for index, column in enumerate(label_columns)}
            )
            writer.writerow(payload)
    audit_path.write_text(json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_and_verify_index_audit(path: str | Path) -> dict[str, Any]:
    audit_path = Path(path).expanduser().resolve()
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ECG model-index audit must be a JSON object.")
    observed = str(payload.get("audit_sha256", ""))
    if not observed or observed != _canonical_hash(payload, "audit_sha256"):
        raise ValueError("ECG model-index audit SHA-256 verification failed.")
    if payload.get("ready_for_baseline_execution") is not True:
        raise RuntimeError("ECG baseline execution is blocked by the model-index audit.")
    if payload.get("development_source") != "original_ptbxl_v1_0_1":
        raise ValueError("ECG model-index audit has an unexpected development source.")
    if payload.get("challenge_ptbxl_model_input") is not False:
        raise ValueError("ECG model-index audit illegally includes Challenge PTB-XL model input.")
    return payload
