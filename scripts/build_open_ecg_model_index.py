#!/usr/bin/env python3
"""Build the tamper-evident local TRUST-ECG model index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trust_icu.ecg_index import build_model_index, write_model_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge-training-root")
    parser.add_argument("--waveform-audit")
    parser.add_argument("--label-manifest")
    parser.add_argument("--ptbxl-assignment")
    parser.add_argument("--output-root", default="open_ecg_model_index_outputs")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "study": "TRUST-ECG",
                    "stage": "verified_model_index_before_baseline_execution",
                    "record_level_output": "local_only_open_ecg_model_index.csv",
                    "aggregate_output": "open_ecg_model_index_audit.json",
                    "checks": [
                        "waveform audit SHA-256 valid and ready",
                        "locked label manifest SHA-256 valid",
                        "PTB-XL assignment hash matches waveform audit",
                        "all four primary source record counts exact",
                        "raw header corpus hashes unchanged",
                        "raw waveform corpus hashes unchanged",
                        "PTB-XL folds map only to locked development roles",
                        "external records use label-blind deterministic 60/40 partition",
                    ],
                    "prohibited": [
                        "outcome-informed external repartition",
                        "external records in model-fitting folds",
                        "training after raw corpus mutation",
                        "committing record-level model index",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    required = {
        "--challenge-training-root": args.challenge_training_root,
        "--waveform-audit": args.waveform_audit,
        "--label-manifest": args.label_manifest,
        "--ptbxl-assignment": args.ptbxl_assignment,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")

    output_root = Path(args.output_root).expanduser().resolve()
    rows, audit = build_model_index(
        challenge_training_root=args.challenge_training_root,
        waveform_audit_path=args.waveform_audit,
        label_manifest_path=args.label_manifest,
        ptbxl_assignment_path=args.ptbxl_assignment,
    )
    write_model_index(
        rows,
        audit,
        index_output=output_root / "open_ecg_model_index.csv",
        audit_output=output_root / "open_ecg_model_index_audit.json",
    )
    print(
        json.dumps(
            {
                "ready_for_baseline_execution": audit.ready_for_baseline_execution,
                "total_rows": audit.total_rows,
                "label_count": len(audit.label_codes),
                "index_sha256": audit.index_sha256,
                "audit_sha256": audit.audit_sha256,
                "blockers": list(audit.blockers),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit.ready_for_baseline_execution else 2


if __name__ == "__main__":
    raise SystemExit(main())
