"""Content-identity crosswalk between Challenge PTB-XL and original PTB-XL headers.

The Challenge record names are intentionally not assumed to encode original PTB-XL ``ecg_id``
values. Records are joined only by the complete ordered 12-lead WFDB checksum signature. The
resulting row-level mapping remains local; public audit output is aggregate-only.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from trust_icu.ecg_data import (
    EXPECTED_LEADS,
    EcgHeaderAudit,
    HeaderRecord,
    build_header_audit,
    parse_challenge_header,
)

_AUGMENTED_LEAD_CANONICAL = {
    "AVR": "aVR",
    "AVL": "aVL",
    "AVF": "aVF",
}


@dataclass(frozen=True)
class PtbxlResolvedPair:
    """Local-only content-verified Challenge/original PTB-XL mapping row."""

    challenge_record_id: str
    ecg_id: int
    strat_fold: int
    filename_hr: str


@dataclass(frozen=True)
class _OriginalRecord:
    metadata: dict[str, str]
    header: HeaderRecord


def _canonical_hash(payload: dict[str, Any], hash_key: str) -> str:
    material = dict(payload)
    material[hash_key] = ""
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_leads(leads: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_AUGMENTED_LEAD_CANONICAL.get(lead, lead) for lead in leads)


def _safe_relative_header(filename_hr: str) -> Path:
    relative = Path(filename_hr + ".hea")
    if relative.is_absolute() or not filename_hr or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Unsafe PTB-XL filename_hr value: {filename_hr!r}")
    return relative


def _read_metadata(path: str | Path) -> list[dict[str, str]]:
    metadata_path = Path(path).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"PTB-XL metadata not found: {metadata_path}")
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"ecg_id", "patient_id", "strat_fold", "filename_hr"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("PTB-XL metadata lacks required crosswalk columns.")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("PTB-XL metadata is empty.")
    if len({int(row["ecg_id"]) for row in rows}) != len(rows):
        raise ValueError("PTB-XL metadata contains duplicate ecg_id values.")
    return rows


def _checksum_signature(record: HeaderRecord) -> tuple[int, ...] | None:
    if len(record.signal_checksums) != len(EXPECTED_LEADS):
        return None
    return tuple(int(value) for value in record.signal_checksums)


def resolve_ptbxl_checksum_crosswalk(
    *,
    challenge_records: list[HeaderRecord],
    metadata_csv: str | Path,
    original_ptbxl_root: str | Path,
) -> tuple[tuple[PtbxlResolvedPair, ...], dict[str, Any]]:
    """Resolve PTB-XL identities by exact 12-lead checksum signatures and fail closed on ambiguity."""

    challenge = [record for record in challenge_records if record.source == "ptb-xl"]
    metadata = _read_metadata(metadata_csv)
    root = Path(original_ptbxl_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Original PTB-XL header root not found: {root}")

    original_records: list[_OriginalRecord] = []
    missing_original_headers = 0
    for row in metadata:
        relative = _safe_relative_header(str(row["filename_hr"]))
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("PTB-XL metadata filename_hr escaped the declared original root.")
        if not path.is_file():
            missing_original_headers += 1
            continue
        original_records.append(
            _OriginalRecord(
                metadata=row,
                header=parse_challenge_header(path, source="ptbxl_original"),
            )
        )

    challenge_by_signature: dict[tuple[int, ...], list[HeaderRecord]] = defaultdict(list)
    original_by_signature: dict[tuple[int, ...], list[_OriginalRecord]] = defaultdict(list)
    challenge_checksum_unavailable = 0
    original_checksum_unavailable = 0

    for record in challenge:
        signature = _checksum_signature(record)
        if signature is None:
            challenge_checksum_unavailable += 1
        else:
            challenge_by_signature[signature].append(record)
    for record in original_records:
        signature = _checksum_signature(record.header)
        if signature is None:
            original_checksum_unavailable += 1
        else:
            original_by_signature[signature].append(record)

    duplicate_challenge_signatures = sum(
        len(records) for records in challenge_by_signature.values() if len(records) != 1
    )
    duplicate_original_signatures = sum(
        len(records) for records in original_by_signature.values() if len(records) != 1
    )

    unmatched = 0
    ambiguous = 0
    sampling_rate_mismatches = 0
    sample_count_mismatches = 0
    lead_order_mismatches = 0
    augmented_case_normalization_records = 0
    verified_pairs = 0
    resolved: list[PtbxlResolvedPair] = []
    matched_ecg_ids: set[int] = set()

    for signature, challenge_matches in challenge_by_signature.items():
        if len(challenge_matches) != 1:
            ambiguous += len(challenge_matches)
            continue
        original_matches = original_by_signature.get(signature, [])
        if not original_matches:
            unmatched += 1
            continue
        if len(original_matches) != 1:
            ambiguous += 1
            continue

        challenge_record = challenge_matches[0]
        original = original_matches[0]
        rate_match = challenge_record.sampling_rate_hz == original.header.sampling_rate_hz
        count_match = challenge_record.sample_count == original.header.sample_count
        challenge_leads = _canonical_leads(challenge_record.lead_names)
        original_leads = _canonical_leads(original.header.lead_names)
        lead_match = challenge_leads == original_leads == tuple(EXPECTED_LEADS)
        if challenge_record.lead_names != original.header.lead_names and lead_match:
            augmented_case_normalization_records += 1
        if not rate_match:
            sampling_rate_mismatches += 1
        if not count_match:
            sample_count_mismatches += 1
        if not lead_match:
            lead_order_mismatches += 1
        if not (rate_match and count_match and lead_match):
            continue

        ecg_id = int(original.metadata["ecg_id"])
        if ecg_id in matched_ecg_ids:
            ambiguous += 1
            continue
        matched_ecg_ids.add(ecg_id)
        resolved.append(
            PtbxlResolvedPair(
                challenge_record_id=challenge_record.record_id,
                ecg_id=ecg_id,
                strat_fold=int(original.metadata["strat_fold"]),
                filename_hr=str(original.metadata["filename_hr"]),
            )
        )
        verified_pairs += 1

    metadata_rows_unmatched = len(metadata) - len(matched_ecg_ids)
    valid = (
        len(challenge) == len(metadata)
        and missing_original_headers == 0
        and challenge_checksum_unavailable == 0
        and original_checksum_unavailable == 0
        and duplicate_challenge_signatures == 0
        and duplicate_original_signatures == 0
        and unmatched == 0
        and ambiguous == 0
        and sampling_rate_mismatches == 0
        and sample_count_mismatches == 0
        and lead_order_mismatches == 0
        and metadata_rows_unmatched == 0
        and verified_pairs == len(challenge)
    )
    report: dict[str, Any] = {
        "required": True,
        "valid": valid,
        "method": "all_12_wfdb_checksum_signature_join",
        "challenge_records": len(challenge),
        "metadata_rows": len(metadata),
        "missing_original_headers": missing_original_headers,
        "challenge_checksum_unavailable": challenge_checksum_unavailable,
        "original_checksum_unavailable": original_checksum_unavailable,
        "duplicate_challenge_checksum_signature_records": duplicate_challenge_signatures,
        "duplicate_original_checksum_signature_records": duplicate_original_signatures,
        "checksum_signature_unmatched": unmatched,
        "checksum_signature_ambiguous": ambiguous,
        "sampling_rate_mismatches": sampling_rate_mismatches,
        "sample_count_mismatches": sample_count_mismatches,
        "lead_order_mismatches_after_canonicalization": lead_order_mismatches,
        "augmented_lead_case_normalization_records": augmented_case_normalization_records,
        "metadata_rows_unmatched": metadata_rows_unmatched,
        "verified_pairs": verified_pairs,
        # Compatibility summary keys retained for existing aggregate consumers.
        "checksum_unavailable": challenge_checksum_unavailable + original_checksum_unavailable,
        "checksum_mismatches": unmatched,
        "structural_mismatches": (
            sampling_rate_mismatches + sample_count_mismatches + lead_order_mismatches
        ),
    }
    return tuple(sorted(resolved, key=lambda row: row.challenge_record_id)), report


def build_checksum_verified_header_audit(
    *,
    records: list[HeaderRecord],
    scored_mapping_path: str | Path,
    ptbxl_metadata_csv: str | Path,
    ptbxl_original_root: str | Path,
    protocol_version: str,
    minimum_development_positives: int,
    minimum_external_positives: int,
    minimum_external_domains: int,
) -> EcgHeaderAudit:
    """Build the standard header audit but replace the obsolete rank crosswalk with content identity."""

    base = build_header_audit(
        records=records,
        scored_mapping_path=scored_mapping_path,
        ptbxl_metadata_csv=ptbxl_metadata_csv,
        ptbxl_original_root=ptbxl_original_root,
        require_ptbxl_crosswalk=False,
        protocol_version=protocol_version,
        minimum_development_positives=minimum_development_positives,
        minimum_external_positives=minimum_external_positives,
        minimum_external_domains=minimum_external_domains,
    )
    _, crosswalk = resolve_ptbxl_checksum_crosswalk(
        challenge_records=records,
        metadata_csv=ptbxl_metadata_csv,
        original_ptbxl_root=ptbxl_original_root,
    )
    blockers = [
        blocker
        for blocker in base.blockers
        if blocker != "challenge_ptbxl_crosswalk_not_verified"
    ]
    if crosswalk.get("valid") is not True:
        blockers.append("challenge_ptbxl_crosswalk_not_verified")

    candidate = replace(
        base,
        ptbxl_crosswalk=crosswalk,
        ready_for_waveform_stage=not blockers,
        blockers=tuple(blockers),
        manifest_sha256="",
    )
    material = candidate.to_dict()
    material["manifest_sha256"] = ""
    return replace(candidate, manifest_sha256=_canonical_hash(material, "manifest_sha256"))


def assignment_payload_sha256(rows: tuple[PtbxlResolvedPair, ...]) -> str:
    """Hash the local resolved assignment without exposing it in public aggregate output."""

    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item.challenge_record_id):
        digest.update(
            f"{row.challenge_record_id},{row.ecg_id},{row.strat_fold},{row.filename_hr}\n".encode()
        )
    return digest.hexdigest()


def resolved_pair_to_dict(pair: PtbxlResolvedPair) -> dict[str, Any]:
    """Explicit helper for trusted local consumers; do not publish row-level mappings."""

    return asdict(pair)
