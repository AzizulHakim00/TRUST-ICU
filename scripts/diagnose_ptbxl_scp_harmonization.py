#!/usr/bin/env python3
"""Aggregate-only diagnostic for the two unresolved PTB-XL label harmonizations.

This tool never selects labels from model performance. It enumerates SCP-ECG codes whose union
with the already justified base statement exactly reproduces the corresponding public Challenge
PTB-XL aggregate label count, and joins candidate codes to official PTB-XL statement descriptions.
Clinical/semantic acceptance remains a separate explicit decision.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

TARGETS = {
    "PAC_equivalent": {"base": "PAC", "target": 555},
    "NSR": {"base": "SR", "target": 18092},
}


def _read_metadata(path: Path) -> list[set[str]]:
    rows: list[set[str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "scp_codes" not in (reader.fieldnames or []):
            raise ValueError("PTB-XL metadata lacks scp_codes.")
        for row in reader:
            parsed = ast.literal_eval(str(row["scp_codes"]))
            if not isinstance(parsed, dict):
                raise ValueError("scp_codes must parse to a dictionary.")
            rows.append({str(code) for code in parsed})
    return rows


def _read_statements(path: Path) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Empty scp_statements CSV.")
        first = reader.fieldnames[0]
        for row in reader:
            code = str(row.get(first, "")).strip()
            if code:
                descriptions[code] = str(row.get("description", "")).strip()
    return descriptions


def diagnose(metadata: Path, statements: Path) -> dict[str, Any]:
    record_codes = _read_metadata(metadata)
    descriptions = _read_statements(statements)
    counts = Counter(code for codes in record_codes for code in codes)
    all_codes = sorted(counts)
    report: dict[str, Any] = {
        "diagnostic_version": "0.1.0",
        "aggregate_only": True,
        "model_performance_touched": False,
        "records": len(record_codes),
        "targets": {},
    }
    for name, spec in TARGETS.items():
        base = str(spec["base"])
        target = int(spec["target"])
        base_records = {index for index, codes in enumerate(record_codes) if base in codes}
        candidates = []
        for code in all_codes:
            if code == base:
                continue
            union_count = sum(1 for codes in record_codes if base in codes or code in codes)
            if union_count == target:
                candidate_records = {index for index, codes in enumerate(record_codes) if code in codes}
                candidates.append(
                    {
                        "scp_code": code,
                        "description": descriptions.get(code, ""),
                        "candidate_count": counts[code],
                        "overlap_with_base": len(base_records & candidate_records),
                        "union_count": union_count,
                    }
                )
        report["targets"][name] = {
            "base_scp_code": base,
            "base_description": descriptions.get(base, ""),
            "base_count": counts[base],
            "target_challenge_count": target,
            "additional_unique_records_needed": target - counts[base],
            "single_code_exact_union_candidates": candidates,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl-metadata")
    parser.add_argument("--scp-statements")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"stage": "ptbxl_scp_harmonization_diagnostic", "targets": TARGETS}, indent=2))
        return 0
    if not args.ptbxl_metadata or not args.scp_statements or not args.output:
        raise SystemExit("--ptbxl-metadata, --scp-statements and --output are required")
    result = diagnose(Path(args.ptbxl_metadata), Path(args.scp_statements))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
