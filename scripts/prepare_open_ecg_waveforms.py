#!/usr/bin/env python3
"""Run the deterministic TRUST-ECG v0.4 waveform audit before model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trust_icu.ecg_waveform import prepare_waveform_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary-data-root",
        help=(
            "Unified root: ptb-xl/ is original PTB-XL v1.0.1; georgia/, cpsc_2018/, "
            "cpsc_2018_extra/ are Challenge external sources."
        ),
    )
    parser.add_argument("--ptbxl-metadata", help="Original PTB-XL v1.0.1 ptbxl_database.csv.")
    parser.add_argument("--header-audit", help="Verified v0.4 Challenge label-support audit JSON.")
    parser.add_argument("--label-manifest", help="Locked v0.4 two-source seven-label manifest JSON.")
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
                    "protocol_version": "0.4.0",
                    "stage": "deterministic_waveform_audit_before_model_training",
                    "required_layout": {
                        "ptb-xl": "original PTB-XL v1.0.1 metadata + records500 WFDB .hea/.dat",
                        "georgia": "Challenge 2020 external .hea/.mat",
                        "cpsc_2018": "Challenge 2020 external .hea/.mat",
                        "cpsc_2018_extra": "Challenge 2020 external .hea/.mat",
                    },
                    "required_inputs": [
                        "verified Challenge label-support audit",
                        "locked two-source label manifest",
                        "complete original PTB-XL v1.0.1 records500 waveforms",
                        "complete Georgia/CPSC/CPSC-Extra Challenge waveforms",
                        "original PTB-XL v1.0.1 metadata",
                    ],
                    "operations": [
                        "build official metadata-to-fold assignment directly for original PTB-XL",
                        "never load Challenge-renamed PTB-XL as model input",
                        "hash raw header and waveform corpora by source",
                        "read original PTB-XL WFDB physical mV using wfdb",
                        "convert external Challenge digital samples to physical mV",
                        "resample to 500 Hz when required",
                        "center crop or symmetric zero pad to 10 seconds",
                        "fit per-lead normalization only on original PTB-XL folds 1-7",
                        "write aggregate waveform audit and hashed normalization stats",
                    ],
                    "prohibited": [
                        "Challenge PTB-XL model input",
                        "reverse Challenge/PTB-XL crosswalk",
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
        "--primary-data-root": args.primary_data_root,
        "--ptbxl-metadata": args.ptbxl_metadata,
        "--header-audit": args.header_audit,
        "--label-manifest": args.label_manifest,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"Required outside --dry-run: {', '.join(missing)}")

    audit = prepare_waveform_stage(
        primary_data_root=args.primary_data_root,
        ptbxl_metadata_csv=args.ptbxl_metadata,
        header_audit_path=args.header_audit,
        label_manifest_path=args.label_manifest,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "ready_for_model_stage": audit.ready_for_model_stage,
                "development_source": audit.development_source,
                "challenge_ptbxl_model_input": audit.challenge_ptbxl_model_input,
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
