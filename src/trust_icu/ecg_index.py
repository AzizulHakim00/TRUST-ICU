"""Tamper-evident local model index for the TRUST-ECG open-data study.

The index binds every primary record to a frozen statistical role and locked label vector. It is
record-level derived data and must remain local. The companion audit is aggregate-only.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class EcgIndexRow:
    source: str
    record_id: str
    relative_header_path: str
    relative_mat_path: str
    role: Role
    strat_fold: int | None
    labels: tuple[int, ...]


@dataclass(frozen=True)
class EcgIndexAudit:
    audit_version: str
    waveform_audit_sha256: str
    label_manifest_sha256: str
    label_codes: tuple[str, ...]
    total_rows: int
    source_rows: dict[str, int]
    role_rows: dict[str, int]
    source_role_rows: dict[str, dict[str, int]]
    source_label_positives: dict[str, dict[str, int]]
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
    match = re.search(r"(\d+)$", record_id)
    if not match:
        raise ValueError(f"Record ID has no terminal numeric component: {record_id!r}")
    return int(match.group(1))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _corpus_hash(source_root: Path, header_paths: list[Path], *, suffix: str) -> str:
    digest = hashlib.sha256()
    for header_path in header_paths:
        path = header_path if suffix == ".hea" else header_path.with_suffix(suffix)
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
        required = {"challenge_record_id", "ecg_id", "strat_fold"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Verified PTB-XL assignment has unexpected columns.")
        rows = tuple(
            PtbxlAssignment(
                challenge_record_id=str(row["challenge_record_id"]),
                ecg_id=int(row["ecg_id"]),
                strat_fold=int(row["strat_fold"]),
            )
            for row in reader
        )
    if not rows:
        raise ValueError("Verified PTB-XL assignment is empty.")
    if len({row.challenge_record_id for row in rows}) != len(rows):
        raise ValueError("Verified PTB-XL assignment contains duplicate Challenge record IDs.")
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


def _locked_label_map(manifest: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, int]]:
    labels = manifest.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("Locked ECG label manifest has no labels.")
    canonical_codes: list[str] = []
    member_to_index: dict[str, int] = {}
    for index, item in enumerate(labels):
        if not isinstance(item, dict):
            raise ValueError("Locked ECG label manifest contains a malformed label entry.")
        canonical = str(item["canonical_code"])
        canonical_codes.append(canonical)
        members = item.get("member_codes")
        if not isinstance(members, list) or not members:
            raise ValueError(f"Locked label {canonical} has no member_codes.")
        for member in members:
            code = str(member)
            previous = member_to_index.setdefault(code, index)
            if previous != index:
                raise ValueError(f"Diagnosis code {code} maps to multiple locked labels.")
    if len(set(canonical_codes)) != len(canonical_codes):
        raise ValueError("Locked ECG label manifest contains duplicate canonical codes.")
    return tuple(canonical_codes), member_to_index


def _label_vector(
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
        bits = "".join(str(value) for value in row.labels)
        fold = "" if row.strat_fold is None else str(row.strat_fold)
        digest.update(
            (
                f"{row.source}|{row.record_id}|{row.relative_header_path}|{row.relative_mat_path}|"
                f"{row.role}|{fold}|{bits}\n"
            ).encode()
        )
    return digest.hexdigest()


def build_model_index(
    *,
    challenge_training_root: str | Path,
    waveform_audit_path: str | Path,
    label_manifest_path: str | Path,
    ptbxl_assignment_path: str | Path,
) -> tuple[list[EcgIndexRow], EcgIndexAudit]:
    """Build and verify the complete local record index before any baseline is executed."""

    waveform_audit = load_and_verify_waveform_audit(waveform_audit_path)
    manifest = load_and_verify_label_manifest(label_manifest_path)
    if str(waveform_audit["label_manifest_sha256"]) != str(manifest["manifest_sha256"]):
        raise ValueError("Waveform audit and label manifest are not from the same locked study state.")

    assignments = _load_assignment(ptbxl_assignment_path)
    if assignment_sha256(assignments) != str(waveform_audit["ptbxl_assignment_sha256"]):
        raise ValueError("PTB-XL assignment does not match the hash frozen by the waveform audit.")
    fold_by_record = {row.challenge_record_id: row.strat_fold for row in assignments}

    label_codes, member_to_index = _locked_label_map(manifest)
    root = Path(challenge_training_root).expanduser().resolve()
    rows: list[EcgIndexRow] = []
    seen: set[tuple[str, str]] = set()
    duplicate_count = 0
    source_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    source_role_counts = {source: Counter() for source in EXPECTED_SOURCES}
    source_label_counts = {source: Counter() for source in EXPECTED_SOURCES}
    rows_without_label = 0
    corpus_hashes_verified = True

    for source, expected_count in EXPECTED_SOURCES.items():
        source_root = root / source
        if not source_root.is_dir():
            raise FileNotFoundError(f"Missing Challenge source directory: {source_root}")
        header_paths = sorted(
            source_root.rglob("*.hea"),
            key=lambda path: (_numeric_record_id(path.stem), str(path)),
        )
        if len(header_paths) != expected_count:
            raise RuntimeError(
                f"Source {source} has {len(header_paths)} headers; expected {expected_count}."
            )

        observed_header_hash = _corpus_hash(source_root, header_paths, suffix=".hea")
        observed_waveform_hash = _corpus_hash(source_root, header_paths, suffix=".mat")
        if observed_header_hash != str(waveform_audit["source_header_corpus_sha256"][source]):
            corpus_hashes_verified = False
        if observed_waveform_hash != str(waveform_audit["source_waveform_corpus_sha256"][source]):
            corpus_hashes_verified = False

        for header_path in header_paths:
            parsed = parse_challenge_header(header_path, source=source)
            key = (source, parsed.record_id)
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            mat_path = header_path.with_suffix(".mat")
            if source == "ptb-xl":
                fold = fold_by_record.get(parsed.record_id)
                if fold is None:
                    raise RuntimeError(
                        f"PTB-XL record is absent from verified fold assignment: {parsed.record_id}"
                    )
                role = _ptb_role(fold)
            else:
                fold = None
                partition = external_partition(source=source, record_id=parsed.record_id)
                role = (
                    "external_certification"
                    if partition == "certification"
                    else "external_recovery_pool"
                )

            labels = _label_vector(
                parsed.dx_codes,
                n_labels=len(label_codes),
                member_to_index=member_to_index,
            )
            if not any(labels):
                rows_without_label += 1
            for index, value in enumerate(labels):
                if value:
                    source_label_counts[source][label_codes[index]] += 1
            row = EcgIndexRow(
                source=source,
                record_id=parsed.record_id,
                relative_header_path=header_path.relative_to(root).as_posix(),
                relative_mat_path=mat_path.relative_to(root).as_posix(),
                role=role,
                strat_fold=fold,
                labels=labels,
            )
            rows.append(row)
            source_counts[source] += 1
            role_counts[role] += 1
            source_role_counts[source][role] += 1

    blockers: list[str] = []
    if duplicate_count:
        blockers.append("duplicate_source_record_ids")
    if not corpus_hashes_verified:
        blockers.append("corpus_hash_changed_after_waveform_audit")
    if source_counts["ptb-xl"] != len(assignments):
        blockers.append("ptbxl_assignment_does_not_cover_exact_development_source")
    for source, expected_count in EXPECTED_SOURCES.items():
        if source_counts[source] != expected_count:
            blockers.append(f"source_count_mismatch:{source}")
    for role in ("model_fit", "optimization_validation", "calibration", "internal_test"):
        if role_counts[role] == 0:
            blockers.append(f"development_role_empty:{role}")
    for source in ("georgia", "cpsc_2018", "cpsc_2018_extra"):
        if source_role_counts[source]["external_certification"] == 0:
            blockers.append(f"external_certification_partition_empty:{source}")
        if source_role_counts[source]["external_recovery_pool"] == 0:
            blockers.append(f"external_recovery_partition_empty:{source}")

    index_hash = model_index_sha256(rows, label_codes)
    payload: dict[str, Any] = {
        "audit_version": "0.1.0",
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
        audit_version=str(payload["audit_version"]),
        waveform_audit_sha256=str(payload["waveform_audit_sha256"]),
        label_manifest_sha256=str(payload["label_manifest_sha256"]),
        label_codes=label_codes,
        total_rows=int(payload["total_rows"]),
        source_rows=payload["source_rows"],
        role_rows=payload["role_rows"],
        source_role_rows=payload["source_role_rows"],
        source_label_positives=payload["source_label_positives"],
        rows_with_no_locked_positive_label=int(payload["rows_with_no_locked_positive_label"]),
        duplicate_source_record_ids=int(payload["duplicate_source_record_ids"]),
        corpus_hashes_verified=bool(payload["corpus_hashes_verified"]),
        index_sha256=str(payload["index_sha256"]),
        ready_for_baseline_execution=bool(payload["ready_for_baseline_execution"]),
        blockers=tuple(payload["blockers"]),
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
    """Write local record-level index plus aggregate audit. The index must never be committed."""

    index_path = Path(index_output).expanduser().resolve()
    audit_path = Path(audit_output).expanduser().resolve()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    label_columns = [f"label_{code}" for code in audit.label_codes]
    fieldnames = [
        "source",
        "record_id",
        "relative_header_path",
        "relative_mat_path",
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
                "relative_mat_path": row.relative_mat_path,
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
    expected = _canonical_hash(payload, "audit_sha256")
    if not observed or observed != expected:
        raise ValueError("ECG model-index audit SHA-256 verification failed.")
    if payload.get("ready_for_baseline_execution") is not True:
        raise RuntimeError("ECG baseline execution is blocked by the model-index audit.")
    return payload
