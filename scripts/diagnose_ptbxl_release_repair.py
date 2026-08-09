#!/usr/bin/env python3
"""Diagnose whether the PTB-XL v1.0.0→v1.0.1 ID repair explains Challenge HR IDs.

PTB-XL v1.0.1 states that it fixed mismatching IDs between waveform data and metadata while
retaining the same content. This tool compares both official metadata releases and, when Challenge
headers are supplied, tests candidate order/permutation relationships using aggregate demographics.
It never declares a crosswalk valid and never emits row-level identifiers.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from trust_icu.ecg_data import HeaderRecord, scan_headers

_ID_RE = re.compile(r"(\d+)(?:_hr)?$")


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path).expanduser().resolve()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Metadata CSV is empty: {csv_path}")
    return rows


def _waveform_id(filename_hr: str) -> int:
    stem = Path(filename_hr).name
    match = _ID_RE.search(stem)
    if not match:
        raise ValueError(f"Cannot parse numeric waveform ID from filename_hr={filename_hr!r}")
    return int(match.group(1))


def _normalized_scalar(value: str | None) -> str:
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text.lower()
    if math.isnan(numeric):
        return ""
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.8g}"


def _challenge_age(record: HeaderRecord) -> str:
    return _normalized_scalar(record.age)


def _challenge_sex(record: HeaderRecord) -> str:
    text = _normalized_scalar(record.sex)
    aliases = {"m": "male", "f": "female"}
    return aliases.get(text, text)


def _metadata_sex(row: dict[str, str]) -> str:
    raw = _normalized_scalar(row.get("sex"))
    aliases = {"0": "male", "1": "female", "m": "male", "f": "female"}
    return aliases.get(raw, raw)


def _metadata_age(row: dict[str, str]) -> str:
    return _normalized_scalar(row.get("age"))


def _stable_row_signature(row: dict[str, str], *, exclude: set[str]) -> str:
    pairs = [(key, _normalized_scalar(value)) for key, value in sorted(row.items()) if key not in exclude]
    return json.dumps(pairs, separators=(",", ":"), ensure_ascii=True)


def _parse_scp_keys(raw: str) -> tuple[str, ...]:
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return ()
    if not isinstance(value, dict):
        return ()
    return tuple(sorted(str(key) for key in value))


def _challenge_numeric(record_id: str) -> int:
    match = re.search(r"(\d+)$", record_id)
    if not match:
        raise ValueError(f"Challenge record ID lacks numeric suffix: {record_id!r}")
    return int(match.group(1))


def _demographic_agreement(
    challenge: list[HeaderRecord],
    candidate_rows: list[dict[str, str]],
) -> dict[str, Any]:
    if len(challenge) != len(candidate_rows):
        return {
            "rows_compared": 0,
            "age_comparable": 0,
            "age_matches": 0,
            "sex_comparable": 0,
            "sex_matches": 0,
            "both_comparable": 0,
            "both_match": 0,
        }
    age_comparable = age_matches = 0
    sex_comparable = sex_matches = 0
    both_comparable = both_match = 0
    for record, row in zip(challenge, candidate_rows, strict=True):
        ca = _challenge_age(record)
        ma = _metadata_age(row)
        cs = _challenge_sex(record)
        ms = _metadata_sex(row)
        age_ok = bool(ca and ma)
        sex_ok = bool(cs and ms)
        if age_ok:
            age_comparable += 1
            age_matches += int(ca == ma)
        if sex_ok:
            sex_comparable += 1
            sex_matches += int(cs == ms)
        if age_ok and sex_ok:
            both_comparable += 1
            both_match += int(ca == ma and cs == ms)
    return {
        "rows_compared": len(challenge),
        "age_comparable": age_comparable,
        "age_matches": age_matches,
        "age_match_fraction": age_matches / age_comparable if age_comparable else None,
        "sex_comparable": sex_comparable,
        "sex_matches": sex_matches,
        "sex_match_fraction": sex_matches / sex_comparable if sex_comparable else None,
        "both_comparable": both_comparable,
        "both_match": both_match,
        "both_match_fraction": both_match / both_comparable if both_comparable else None,
    }


def diagnose_release_repair(
    *,
    v100_rows: list[dict[str, str]],
    v101_rows: list[dict[str, str]],
    challenge_records: list[HeaderRecord] | None = None,
) -> dict[str, Any]:
    required = {"ecg_id", "patient_id", "filename_hr", "filename_lr", "age", "sex", "strat_fold"}
    for version, rows in (("1.0.0", v100_rows), ("1.0.1", v101_rows)):
        missing = required - set(rows[0])
        if missing:
            raise ValueError(f"PTB-XL {version} metadata missing required columns: {sorted(missing)}")

    v100_by_ecg = {int(row["ecg_id"]): row for row in v100_rows}
    v101_by_ecg = {int(row["ecg_id"]): row for row in v101_rows}
    if len(v100_by_ecg) != len(v100_rows) or len(v101_by_ecg) != len(v101_rows):
        raise ValueError("PTB-XL metadata contains duplicate ecg_id values.")

    shared_ids = sorted(set(v100_by_ecg) & set(v101_by_ecg))
    filename_hr_changed_same_ecg = 0
    filename_lr_changed_same_ecg = 0
    patient_changed_same_ecg = 0
    fold_changed_same_ecg = 0
    age_changed_same_ecg = 0
    sex_changed_same_ecg = 0
    scp_changed_same_ecg = 0
    old_filename_matches_ecg = 0
    new_filename_matches_ecg = 0
    old_waveform_ids: list[int] = []
    new_waveform_ids: list[int] = []

    for ecg_id in shared_ids:
        old = v100_by_ecg[ecg_id]
        new = v101_by_ecg[ecg_id]
        old_wave = _waveform_id(old["filename_hr"])
        new_wave = _waveform_id(new["filename_hr"])
        old_waveform_ids.append(old_wave)
        new_waveform_ids.append(new_wave)
        old_filename_matches_ecg += int(old_wave == ecg_id)
        new_filename_matches_ecg += int(new_wave == ecg_id)
        filename_hr_changed_same_ecg += int(old["filename_hr"] != new["filename_hr"])
        filename_lr_changed_same_ecg += int(old["filename_lr"] != new["filename_lr"])
        patient_changed_same_ecg += int(_normalized_scalar(old["patient_id"]) != _normalized_scalar(new["patient_id"]))
        fold_changed_same_ecg += int(_normalized_scalar(old["strat_fold"]) != _normalized_scalar(new["strat_fold"]))
        age_changed_same_ecg += int(_metadata_age(old) != _metadata_age(new))
        sex_changed_same_ecg += int(_metadata_sex(old) != _metadata_sex(new))
        if "scp_codes" in old and "scp_codes" in new:
            scp_changed_same_ecg += int(_parse_scp_keys(old["scp_codes"]) != _parse_scp_keys(new["scp_codes"]))

    stable_exclude = {"ecg_id", "filename_hr", "filename_lr"}
    old_signatures = Counter(_stable_row_signature(row, exclude=stable_exclude) for row in v100_rows)
    new_signatures = Counter(_stable_row_signature(row, exclude=stable_exclude) for row in v101_rows)
    stable_signature_multiset_equal = old_signatures == new_signatures
    unique_stable_signatures = sum(1 for value in old_signatures.values() if value == 1 and new_signatures.get(next((k for k, v in old_signatures.items() if v == value), ""), 0) == 1)

    report: dict[str, Any] = {
        "diagnostic_version": "0.1.0",
        "aggregate_only": True,
        "gate_relaxed": False,
        "v100_rows": len(v100_rows),
        "v101_rows": len(v101_rows),
        "shared_ecg_ids": len(shared_ids),
        "filename_hr_changed_same_ecg_id": filename_hr_changed_same_ecg,
        "filename_lr_changed_same_ecg_id": filename_lr_changed_same_ecg,
        "patient_id_changed_same_ecg_id": patient_changed_same_ecg,
        "strat_fold_changed_same_ecg_id": fold_changed_same_ecg,
        "age_changed_same_ecg_id": age_changed_same_ecg,
        "sex_changed_same_ecg_id": sex_changed_same_ecg,
        "scp_keyset_changed_same_ecg_id": scp_changed_same_ecg,
        "v100_filename_hr_numeric_equals_ecg_id": old_filename_matches_ecg,
        "v101_filename_hr_numeric_equals_ecg_id": new_filename_matches_ecg,
        "v100_waveform_ids_unique": len(set(old_waveform_ids)),
        "v101_waveform_ids_unique": len(set(new_waveform_ids)),
        "stable_metadata_signature_multiset_equal_excluding_ecg_and_filenames": stable_signature_multiset_equal,
        "stable_metadata_unique_signature_count_hint": unique_stable_signatures,
    }

    if challenge_records is not None:
        challenge = sorted(challenge_records, key=lambda row: _challenge_numeric(row.record_id))
        old_by_ecg_order = [v100_by_ecg[ecg_id] for ecg_id in sorted(v100_by_ecg)]
        new_by_ecg_order = [v101_by_ecg[ecg_id] for ecg_id in sorted(v101_by_ecg)]
        old_by_filename_order = sorted(v100_rows, key=lambda row: _waveform_id(row["filename_hr"]))
        new_by_filename_order = sorted(v101_rows, key=lambda row: _waveform_id(row["filename_hr"]))
        # If Challenge generation iterated v1.0.0 metadata by ecg_id and loaded its then-current
        # filename_hr, map each old filename to the corrected v1.0.1 owner of that waveform ID.
        new_by_waveform_id = {_waveform_id(row["filename_hr"]): row for row in v101_rows}
        repaired_from_old_ecg_order = [
            new_by_waveform_id[_waveform_id(row["filename_hr"])] for row in old_by_ecg_order
        ]
        report["challenge_records"] = len(challenge)
        report["challenge_numeric_range"] = (
            [
                _challenge_numeric(challenge[0].record_id),
                _challenge_numeric(challenge[-1].record_id),
            ]
            if challenge
            else []
        )
        report["candidate_demographic_agreement"] = {
            "v100_ecg_id_order": _demographic_agreement(challenge, old_by_ecg_order),
            "v100_filename_hr_order": _demographic_agreement(challenge, old_by_filename_order),
            "v101_ecg_id_order": _demographic_agreement(challenge, new_by_ecg_order),
            "v101_filename_hr_order": _demographic_agreement(challenge, new_by_filename_order),
            "v100_ecg_order_then_repaired_waveform_owner": _demographic_agreement(
                challenge,
                repaired_from_old_ecg_order,
            ),
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptbxl-v100-metadata")
    parser.add_argument("--ptbxl-v101-metadata")
    parser.add_argument("--challenge-root")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "ptbxl_v100_to_v101_release_repair_diagnostic",
                    "aggregate_only": True,
                    "gate_relaxed": False,
                    "tests": [
                        "same-ecg-id metadata changes",
                        "old/new filename_hr permutation",
                        "stable metadata multiset consistency",
                        "Challenge HR-order demographic agreement for candidate permutations",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    required = {
        "--ptbxl-v100-metadata": args.ptbxl_v100_metadata,
        "--ptbxl-v101-metadata": args.ptbxl_v101_metadata,
        "--output": args.output,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")

    challenge_records = None
    if args.challenge_root:
        challenge_records = scan_headers(args.challenge_root, sources=("ptb-xl",))
    report = diagnose_release_repair(
        v100_rows=_read_csv(args.ptbxl_v100_metadata),
        v101_rows=_read_csv(args.ptbxl_v101_metadata),
        challenge_records=challenge_records,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
