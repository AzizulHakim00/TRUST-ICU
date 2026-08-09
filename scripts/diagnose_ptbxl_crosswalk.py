#!/usr/bin/env python3
"""Diagnose a failed PTB-XL Challenge/original header crosswalk without relaxing the gate.

The report is aggregate-only: it records mismatch counts and lead-order frequency summaries, but
never emits record identifiers or per-record checksum values.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from trust_icu.ecg_data import EXPECTED_LEADS, HeaderRecord, parse_challenge_header, scan_headers

_NUMERIC_RE = re.compile(r"(\d+)$")


def _numeric_id(record_id: str) -> int:
    match = _NUMERIC_RE.search(record_id)
    if not match:
        raise ValueError(f"Record ID has no terminal numeric component: {record_id!r}")
    return int(match.group(1))


def _load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"ecg_id", "filename_hr"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("PTB-XL metadata lacks required ecg_id/filename_hr columns.")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("PTB-XL metadata is empty.")
    return sorted(rows, key=lambda row: int(row["ecg_id"]))


def _lead_order_key(leads: tuple[str, ...]) -> str:
    return "|".join(leads)


def diagnose_crosswalk(
    *,
    challenge_records: list[HeaderRecord],
    metadata_rows: list[dict[str, str]],
    original_root: Path,
) -> dict[str, Any]:
    challenge = sorted(
        (record for record in challenge_records if record.source == "ptb-xl"),
        key=lambda record: _numeric_id(record.record_id),
    )
    metadata = sorted(metadata_rows, key=lambda row: int(row["ecg_id"]))
    if len(challenge) != len(metadata):
        raise ValueError(
            f"Crosswalk diagnostic row-count mismatch: challenge={len(challenge)}, metadata={len(metadata)}"
        )

    rate_mismatches = 0
    sample_count_mismatches = 0
    lead_order_mismatches = 0
    structural_mismatches = 0
    checksum_unavailable = 0
    checksum_mismatches = 0
    checksum_matches = 0
    fully_verified = 0
    missing_original_headers = 0
    challenge_lead_orders: Counter[str] = Counter()
    original_lead_orders: Counter[str] = Counter()

    for challenge_record, row in zip(challenge, metadata, strict=True):
        challenge_lead_orders[_lead_order_key(challenge_record.lead_names)] += 1
        original_path = original_root / (str(row["filename_hr"]) + ".hea")
        if not original_path.is_file():
            missing_original_headers += 1
            continue
        original = parse_challenge_header(original_path, source="ptbxl_original")
        original_lead_orders[_lead_order_key(original.lead_names)] += 1

        rate_match = challenge_record.sampling_rate_hz == original.sampling_rate_hz
        sample_match = challenge_record.sample_count == original.sample_count
        lead_match = challenge_record.lead_names == original.lead_names
        if not rate_match:
            rate_mismatches += 1
        if not sample_match:
            sample_count_mismatches += 1
        if not lead_match:
            lead_order_mismatches += 1
        if not (rate_match and sample_match and lead_match):
            structural_mismatches += 1

        checksum_available = (
            len(challenge_record.signal_checksums) == len(EXPECTED_LEADS)
            and len(original.signal_checksums) == len(EXPECTED_LEADS)
        )
        checksum_match = False
        if not checksum_available:
            checksum_unavailable += 1
        elif challenge_record.signal_checksums == original.signal_checksums:
            checksum_matches += 1
            checksum_match = True
        else:
            checksum_mismatches += 1

        if rate_match and sample_match and lead_match and checksum_match:
            fully_verified += 1

    challenge_numbers = [_numeric_id(record.record_id) for record in challenge]
    consecutive = bool(challenge_numbers) and challenge_numbers == list(
        range(challenge_numbers[0], challenge_numbers[0] + len(challenge_numbers))
    )
    return {
        "diagnostic_version": "0.1.0",
        "aggregate_only": True,
        "gate_relaxed": False,
        "challenge_records": len(challenge),
        "metadata_rows": len(metadata),
        "challenge_numeric_ids_consecutive": consecutive,
        "missing_original_headers": missing_original_headers,
        "sampling_rate_mismatches": rate_mismatches,
        "sample_count_mismatches": sample_count_mismatches,
        "lead_order_mismatches": lead_order_mismatches,
        "structural_mismatches": structural_mismatches,
        "checksum_unavailable": checksum_unavailable,
        "checksum_matches": checksum_matches,
        "checksum_mismatches": checksum_mismatches,
        "fully_verified_pairs": fully_verified,
        "challenge_lead_orders": dict(sorted(challenge_lead_orders.items())),
        "original_lead_orders": dict(sorted(original_lead_orders.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge-root")
    parser.add_argument("--ptbxl-metadata")
    parser.add_argument("--ptbxl-original-root")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "ptbxl_crosswalk_component_diagnostic",
                    "aggregate_only": True,
                    "gate_relaxed": False,
                    "components": [
                        "sampling_rate",
                        "sample_count",
                        "lead_order",
                        "checksum",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    required = {
        "--challenge-root": args.challenge_root,
        "--ptbxl-metadata": args.ptbxl_metadata,
        "--ptbxl-original-root": args.ptbxl_original_root,
        "--output": args.output,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")

    challenge_root = Path(args.challenge_root).expanduser().resolve()
    metadata_path = Path(args.ptbxl_metadata).expanduser().resolve()
    original_root = Path(args.ptbxl_original_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    records = scan_headers(challenge_root, sources=("ptb-xl",))
    metadata_rows = _load_metadata(metadata_path)
    report = diagnose_crosswalk(
        challenge_records=records,
        metadata_rows=metadata_rows,
        original_root=original_root,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
