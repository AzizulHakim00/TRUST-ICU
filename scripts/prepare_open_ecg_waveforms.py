#!/usr/bin/env python3
"""Run the deterministic TRUST-ECG waveform audit before any model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trust_icu.ecg_waveform import prepare_waveform_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-training-root", help="Challenge training root with four primary source folders.")
    parser.add_argument("--ptbxl-metadata", help="PTB-XL v1.0.1 ptbxl_database.csv.")
    parser.add_argument("--ptbxl-original-root", help="PTB-XL v1.0.1 root containing records500 headers.")
    parser.add_argument("--header-audit", help="Verified open_ecg_header_audit.json.")
    parser.add_argument("--label-manifest", help="Locked open_ecg_label_manifest.json.")
    parser.add_argument("--output-root", default="open_ecg_waveform_outputs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "study": "TRUST-ECG",
                    "stage": "deterministic_waveform_audit_before_model_training",
                    "required_inputs": [
                        "verified header audit",
                        "locked label manifest",
                        "complete Challenge PTB-XL/Georgia/CPSC/CPSC-Extra waveforms",
                        "PTB-XL v1.0.1 metadata",
                        "PTB-XL v1.0.1 original records500 headers",
                    ],
                    "operations": [
                        "reverify Challenge/PTB-XL checksum crosswalk",
                        "write local record-to-fold assignment without patient IDs",
                        "hash raw header and waveform corpora by source",
                        "convert digital ECGs to physical mV",
                        "resample to 500 Hz when required",
                        "center crop or symmetric zero pad to 10 seconds",
                        "fit per-lead normalization only on PTB-XL folds 1-7",
                        "write aggregate waveform audit and hashed normalization stats",
                    ],
                    "prohibited": [
                        "external-domain normalization fitting",
                        "source-specific filtering",
                        "waveform model training before waveform audit passes",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    required = {
        "--challenge-training-root": args.challenge_training_root,
        "--ptbxl-metadata": args.ptbxl_metadata,
        "--ptbxl-original-root": args.ptbxl_original_root,
        "--header-audit": args.header_audit,
        "--label-manifest": args.label_manifest,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"Required outside --dry-run: {', '.join(missing)}")

    audit = prepare_waveform_stage(
        challenge_training_root=args.challenge_training_root,
        ptbxl_metadata_csv=args.ptbxl_metadata,
        ptbxl_original_root=args.ptbxl_original_root,
        header_audit_path=args.header_audit,
        label_manifest_path=args.label_manifest,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "ready_for_model_stage": audit.ready_for_model_stage,
                "blockers": audit.blockers,
                "normalization_stats_sha256": audit.normalization_stats_sha256,
                "ptbxl_assignment_sha256": audit.ptbxl_assignment_sha256,
                "audit_sha256": audit.audit_sha256,
                "output_root": str(Path(args.output_root).expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if args.require_ready and not audit.ready_for_model_stage else 0


if __name__ == "__main__":
    raise SystemExit(main())
