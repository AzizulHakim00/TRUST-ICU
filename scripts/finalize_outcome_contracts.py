#!/usr/bin/env python3
"""Validate reviewer evidence and prepare a private locked Phase 0 runtime context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trust_icu.outcome_evidence import prepare_locked_runtime_context


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--mimic-summary", type=Path, required=True)
    parser.add_argument("--eicu-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = prepare_locked_runtime_context(
        repo_root=args.repo_root,
        evidence_path=args.evidence,
        mimic_summary_path=args.mimic_summary,
        eicu_summary_path=args.eicu_summary,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.ready_for_runtime_lock else 2


if __name__ == "__main__":
    raise SystemExit(main())
