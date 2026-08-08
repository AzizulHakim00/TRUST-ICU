"""Header-only data audit and immutable label locking for the open TRUST-ECG pivot."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPECTED_SOURCES = {
    "ptb-xl": 21837,
    "georgia": 10344,
    "cpsc_2018": 6877,
    "cpsc_2018_extra": 3453,
}
EXTERNAL_SOURCES = ("georgia", "cpsc_2018", "cpsc_2018_extra")
EXPECTED_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


@dataclass(frozen=True)
class HeaderRecord:
    source: str
    record_id: str
    sampling_rate_hz: float
    sample_count: int
    lead_names: tuple[str, ...]
    age: str | None
    sex: str | None
    dx_codes: tuple[str, ...]
    signal_checksums: tuple[int, ...] = ()

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sampling_rate_hz


@dataclass(frozen=True)
class LabelDecision:
    canonical_code: str
    abbreviation: str
    member_codes: tuple[str, ...]
    development_positives: int
    external_positives: dict[str, int]
    external_domains_meeting_threshold: int
    eligible: bool


@dataclass(frozen=True)
class EcgHeaderAudit:
    protocol_version: str
    source_record_counts: dict[str, int]
    source_expected_counts: dict[str, int]
    source_count_matches_expected: dict[str, bool]
    source_sampling_rates_hz: dict[str, list[float]]
    source_duration_seconds: dict[str, dict[str, float]]
    records_with_non_12_lead_contract: dict[str, int]
    records_missing_dx: dict[str, int]
    unscored_dx_occurrences: dict[str, int]
    ptbxl_fold_integrity: dict[str, Any]
    ptbxl_crosswalk: dict[str, Any]
    labels: tuple[LabelDecision, ...]
    eligible_labels: tuple[str, ...]
    ready_for_waveform_stage: bool
    blockers: tuple[str, ...]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _parse_rate(token: str) -> float:
    return float(token.split("/", 1)[0])


def _numeric_record_id(record_id: str) -> int:
    digits = "".join(char for char in record_id if char.isdigit())
    if not digits:
        raise ValueError(f"Record ID does not contain a numeric component: {record_id!r}")
    return int(digits)


def parse_challenge_header(path: str | Path, *, source: str) -> HeaderRecord:
    """Parse one WFDB .hea file without loading waveform samples."""

    header_path = Path(path)
    lines = [
        line.strip()
        for line in header_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"Empty ECG header: {header_path}")
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f"Malformed WFDB first line: {header_path}")
    record_id = first[0]
    n_sig = int(first[1])
    sampling_rate = _parse_rate(first[2])
    sample_count = int(first[3])
    if n_sig <= 0 or sampling_rate <= 0 or sample_count <= 0:
        raise ValueError(f"Invalid signal dimensions: {header_path}")
    if len(lines) < 1 + n_sig:
        raise ValueError(f"Missing WFDB signal specification lines: {header_path}")

    signal_tokens = [lines[index].split() for index in range(1, 1 + n_sig)]
    lead_names = tuple(tokens[-1] for tokens in signal_tokens)
    checksums: list[int] = []
    for tokens in signal_tokens:
        if len(tokens) < 9:
            checksums = []
            break
        try:
            checksums.append(int(tokens[-3]))
        except ValueError:
            checksums = []
            break

    comments: dict[str, str] = {}
    for line in lines[1 + n_sig :]:
        if not line.startswith("#") or ":" not in line:
            continue
        key, value = line[1:].split(":", 1)
        comments[key.strip().lower()] = value.strip()
    dx = tuple(sorted({code.strip() for code in comments.get("dx", "").split(",") if code.strip()}))
    return HeaderRecord(
        source=source,
        record_id=record_id,
        sampling_rate_hz=sampling_rate,
        sample_count=sample_count,
        lead_names=lead_names,
        age=comments.get("age"),
        sex=comments.get("sex"),
        dx_codes=dx,
        signal_checksums=tuple(checksums),
    )


def scan_headers(root: str | Path, *, sources: tuple[str, ...] | None = None) -> list[HeaderRecord]:
    """Recursively scan local Challenge source directories for .hea files."""

    data_root = Path(root).expanduser().resolve()
    selected = sources or tuple(EXPECTED_SOURCES)
    records: list[HeaderRecord] = []
    for source in selected:
        source_root = data_root / source
        if not source_root.is_dir():
            raise FileNotFoundError(f"Missing ECG source directory: {source_root}")
        for path in sorted(source_root.rglob("*.hea")):
            records.append(parse_challenge_header(path, source=source))
    return records


def load_scored_mapping(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the repository-pinned scored-code equivalence mapping."""

    groups: dict[str, dict[str, Any]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            canonical = str(row["canonical_code"])
            entry = groups.setdefault(
                canonical,
                {"abbreviation": str(row["abbreviation"]), "member_codes": set()},
            )
            if entry["abbreviation"] != str(row["abbreviation"]):
                raise ValueError(f"Inconsistent abbreviation for canonical code {canonical}")
            entry["member_codes"].add(str(row["member_code"]))
    return groups


def _count_group_positive_records(
    records: list[HeaderRecord],
    mapping: dict[str, dict[str, Any]],
) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = {source: Counter() for source in EXPECTED_SOURCES}
    code_to_canonical: dict[str, str] = {}
    for canonical, meta in mapping.items():
        for code in meta["member_codes"]:
            previous = code_to_canonical.setdefault(code, canonical)
            if previous != canonical:
                raise ValueError(f"Scored code {code} maps to multiple canonical groups")
    for record in records:
        seen_groups = {code_to_canonical[code] for code in record.dx_codes if code in code_to_canonical}
        for canonical in seen_groups:
            counts[record.source][canonical] += 1
    return counts


def _read_ptbxl_metadata(metadata_csv: str | Path | None) -> tuple[list[dict[str, str]], str | None]:
    if metadata_csv is None:
        return [], "ptbxl_metadata_missing"
    path = Path(metadata_csv).expanduser().resolve()
    if not path.is_file():
        return [], "ptbxl_metadata_not_found"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"ecg_id", "patient_id", "strat_fold", "filename_hr"}
        if not required.issubset(reader.fieldnames or []):
            return [], "ptbxl_metadata_columns_missing"
        rows = [dict(row) for row in reader]
    return rows, None


def validate_ptbxl_folds(metadata_csv: str | Path | None) -> dict[str, Any]:
    """Verify the official PTB-XL patient-wise fold invariant from metadata only."""

    rows, error = _read_ptbxl_metadata(metadata_csv)
    if error:
        return {"provided": metadata_csv is not None, "valid": False, "blocker": error}
    patient_folds: dict[str, set[int]] = defaultdict(set)
    fold_counts: Counter[int] = Counter()
    for row in rows:
        patient_id = str(row["patient_id"])
        fold = int(row["strat_fold"])
        patient_folds[patient_id].add(fold)
        fold_counts[fold] += 1
    leaking_patients = sum(1 for folds in patient_folds.values() if len(folds) > 1)
    folds_present = sorted(fold_counts)
    valid = leaking_patients == 0 and folds_present == list(range(1, 11))
    return {
        "provided": True,
        "valid": valid,
        "rows": len(rows),
        "unique_patients": len(patient_folds),
        "patients_spanning_multiple_folds": leaking_patients,
        "folds_present": folds_present,
        "fold_record_counts": {str(key): fold_counts[key] for key in folds_present},
    }


def validate_ptbxl_crosswalk(
    *,
    challenge_records: list[HeaderRecord],
    metadata_csv: str | Path | None,
    original_ptbxl_root: str | Path | None,
) -> dict[str, Any]:
    """Verify a rank-based Challenge/PTB-XL record crosswalk using all WFDB lead checksums.

    The routine does not trust a filename formula. It pairs the numerically sorted Challenge
    PTB-XL records with metadata rows sorted by official ``ecg_id`` and then requires the paired
    original PTB-XL v1.0.1 header to agree on all 12 signal checksums, sampling rate, sample count,
    and lead order. The crosswalk is valid only when every record verifies.
    """

    rows, error = _read_ptbxl_metadata(metadata_csv)
    if error:
        return {"required": True, "valid": False, "blocker": error}
    if original_ptbxl_root is None:
        return {"required": True, "valid": False, "blocker": "ptbxl_original_header_root_missing"}
    root = Path(original_ptbxl_root).expanduser().resolve()
    if not root.is_dir():
        return {"required": True, "valid": False, "blocker": "ptbxl_original_header_root_not_found"}

    challenge = sorted(
        (record for record in challenge_records if record.source == "ptb-xl"),
        key=lambda record: _numeric_record_id(record.record_id),
    )
    metadata = sorted(rows, key=lambda row: int(row["ecg_id"]))
    if len(challenge) != len(metadata):
        return {
            "required": True,
            "valid": False,
            "blocker": "ptbxl_crosswalk_row_count_mismatch",
            "challenge_records": len(challenge),
            "metadata_rows": len(metadata),
        }

    challenge_numbers = [_numeric_record_id(record.record_id) for record in challenge]
    consecutive_ids = bool(challenge_numbers) and challenge_numbers == list(
        range(challenge_numbers[0], challenge_numbers[0] + len(challenge_numbers))
    )
    missing_original_headers = 0
    checksum_unavailable = 0
    structural_mismatches = 0
    checksum_mismatches = 0
    verified = 0

    for challenge_record, row in zip(challenge, metadata, strict=True):
        relative = Path(str(row["filename_hr"]) + ".hea")
        original_path = root / relative
        if not original_path.is_file():
            missing_original_headers += 1
            continue
        original = parse_challenge_header(original_path, source="ptbxl_original")
        structural_match = (
            challenge_record.sampling_rate_hz == original.sampling_rate_hz
            and challenge_record.sample_count == original.sample_count
            and challenge_record.lead_names == original.lead_names
        )
        if not structural_match:
            structural_mismatches += 1
            continue
        if len(challenge_record.signal_checksums) != len(EXPECTED_LEADS) or len(
            original.signal_checksums
        ) != len(EXPECTED_LEADS):
            checksum_unavailable += 1
            continue
        if challenge_record.signal_checksums != original.signal_checksums:
            checksum_mismatches += 1
            continue
        verified += 1

    valid = (
        consecutive_ids
        and missing_original_headers == 0
        and checksum_unavailable == 0
        and structural_mismatches == 0
        and checksum_mismatches == 0
        and verified == len(challenge)
    )
    return {
        "required": True,
        "valid": valid,
        "method": "numeric_rank_pairing_verified_by_all_12_wfdb_checksums",
        "challenge_records": len(challenge),
        "metadata_rows": len(metadata),
        "challenge_numeric_ids_consecutive": consecutive_ids,
        "missing_original_headers": missing_original_headers,
        "checksum_unavailable": checksum_unavailable,
        "structural_mismatches": structural_mismatches,
        "checksum_mismatches": checksum_mismatches,
        "verified_pairs": verified,
    }


def build_header_audit(
    *,
    records: list[HeaderRecord],
    scored_mapping_path: str | Path,
    ptbxl_metadata_csv: str | Path | None,
    ptbxl_original_root: str | Path | None = None,
    require_ptbxl_crosswalk: bool = True,
    protocol_version: str = "0.1.0",
    minimum_development_positives: int = 500,
    minimum_external_positives: int = 100,
    minimum_external_domains: int = 2,
) -> EcgHeaderAudit:
    """Build an aggregate, hash-locked feasibility audit before waveform training."""

    mapping = load_scored_mapping(scored_mapping_path)
    group_counts = _count_group_positive_records(records, mapping)
    source_counts = Counter(record.source for record in records)
    sampling_rates: dict[str, set[float]] = defaultdict(set)
    durations: dict[str, list[float]] = defaultdict(list)
    bad_leads = Counter()
    missing_dx = Counter()
    scored_member_codes = {code for meta in mapping.values() for code in meta["member_codes"]}
    unscored_dx = Counter()
    for record in records:
        sampling_rates[record.source].add(record.sampling_rate_hz)
        durations[record.source].append(record.duration_seconds)
        if record.lead_names != EXPECTED_LEADS:
            bad_leads[record.source] += 1
        if not record.dx_codes:
            missing_dx[record.source] += 1
        for code in record.dx_codes:
            if code.isdigit() and code not in scored_member_codes:
                unscored_dx[record.source] += 1

    labels: list[LabelDecision] = []
    for canonical in sorted(mapping, key=int):
        development = group_counts["ptb-xl"][canonical]
        external = {source: group_counts[source][canonical] for source in EXTERNAL_SOURCES}
        meeting = sum(value >= minimum_external_positives for value in external.values())
        eligible = development >= minimum_development_positives and meeting >= minimum_external_domains
        labels.append(
            LabelDecision(
                canonical_code=canonical,
                abbreviation=str(mapping[canonical]["abbreviation"]),
                member_codes=tuple(sorted(mapping[canonical]["member_codes"], key=int)),
                development_positives=development,
                external_positives=external,
                external_domains_meeting_threshold=meeting,
                eligible=eligible,
            )
        )

    ptbxl_fold_integrity = validate_ptbxl_folds(ptbxl_metadata_csv)
    ptbxl_crosswalk = (
        validate_ptbxl_crosswalk(
            challenge_records=records,
            metadata_csv=ptbxl_metadata_csv,
            original_ptbxl_root=ptbxl_original_root,
        )
        if require_ptbxl_crosswalk
        else {"required": False, "valid": True, "method": "disabled_for_test_or_nonproduction_audit"}
    )
    source_count_matches = {
        source: source_counts[source] == expected for source, expected in EXPECTED_SOURCES.items()
    }
    blockers: list[str] = []
    if not all(source_count_matches.values()):
        blockers.append("source_record_count_mismatch")
    if any(bad_leads.values()):
        blockers.append("nonstandard_12_lead_headers_present")
    if any(missing_dx.values()):
        blockers.append("headers_missing_diagnosis_codes")
    if not ptbxl_fold_integrity.get("valid", False):
        blockers.append("ptbxl_patientwise_fold_integrity_not_verified")
    if not ptbxl_crosswalk.get("valid", False):
        blockers.append("challenge_ptbxl_crosswalk_not_verified")
    eligible_labels = tuple(decision.canonical_code for decision in labels if decision.eligible)
    if not eligible_labels:
        blockers.append("no_labels_meet_prespecified_transportability_counts")

    duration_summary: dict[str, dict[str, float]] = {}
    for source in EXPECTED_SOURCES:
        values = durations[source]
        duration_summary[source] = {
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }

    material: dict[str, Any] = {
        "protocol_version": protocol_version,
        "source_record_counts": {source: source_counts[source] for source in EXPECTED_SOURCES},
        "source_expected_counts": EXPECTED_SOURCES,
        "source_count_matches_expected": source_count_matches,
        "source_sampling_rates_hz": {source: sorted(sampling_rates[source]) for source in EXPECTED_SOURCES},
        "source_duration_seconds": duration_summary,
        "records_with_non_12_lead_contract": {source: bad_leads[source] for source in EXPECTED_SOURCES},
        "records_missing_dx": {source: missing_dx[source] for source in EXPECTED_SOURCES},
        "unscored_dx_occurrences": {source: unscored_dx[source] for source in EXPECTED_SOURCES},
        "ptbxl_fold_integrity": ptbxl_fold_integrity,
        "ptbxl_crosswalk": ptbxl_crosswalk,
        "labels": [asdict(label) for label in labels],
        "eligible_labels": eligible_labels,
        "ready_for_waveform_stage": not blockers,
        "blockers": blockers,
        "manifest_sha256": "",
    }
    manifest_sha256 = _sha256_json(material)
    return EcgHeaderAudit(
        protocol_version=protocol_version,
        source_record_counts=material["source_record_counts"],
        source_expected_counts=dict(EXPECTED_SOURCES),
        source_count_matches_expected=source_count_matches,
        source_sampling_rates_hz=material["source_sampling_rates_hz"],
        source_duration_seconds=duration_summary,
        records_with_non_12_lead_contract=material["records_with_non_12_lead_contract"],
        records_missing_dx=material["records_missing_dx"],
        unscored_dx_occurrences=material["unscored_dx_occurrences"],
        ptbxl_fold_integrity=ptbxl_fold_integrity,
        ptbxl_crosswalk=ptbxl_crosswalk,
        labels=tuple(labels),
        eligible_labels=eligible_labels,
        ready_for_waveform_stage=not blockers,
        blockers=tuple(blockers),
        manifest_sha256=manifest_sha256,
    )


def write_header_audit(audit: EcgHeaderAudit, output: str | Path) -> None:
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
