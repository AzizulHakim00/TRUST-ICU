#!/usr/bin/env python3
"""Prepare TRUST-ECG external label-support evidence before waveform training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from trust_icu.ecg_data import build_header_audit, scan_headers, write_header_audit

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "schemas/open_ecg_protocol.yaml"
SCORED_MAPPING = ROOT / "schemas/challenge2020_scored_classes.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header-root", help="Local directory containing the four primary Challenge source folders.")
    parser.add_argument("--ptbxl-metadata", help="Path to PTB-XL v1.0.1 ptbxl_database.csv for fold-integrity evidence.")
    parser.add_argument(
        "--output",
        default="open_ecg_header_audit.json",
        help="Aggregate-only Challenge label-support audit output path.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the immutable preparation plan only.")
    parser.add_argument("--require-ready", action="store_true", help="Exit 2 unless all header/label-support gates pass.")
    return parser.parse_args()


def _protocol() -> dict:
    raw = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Open ECG protocol must be a mapping.")
    return raw


def _download_plan() -> dict:
    base = "https://physionet.org/files/challenge-2020/1.0.2/training"
    return {
        "study": "TRUST-ECG",
        "stage": "challenge_header_label_support_reference_before_waveforms",
        "challenge_version": "1.0.2",
        "sources": {
            "ptb-xl": f"{base}/ptb-xl/",
            "georgia": f"{base}/georgia/",
            "cpsc_2018": f"{base}/cpsc_2018/",
            "cpsc_2018_extra": f"{base}/cpsc_2018_extra/",
        },
        "ptbxl_v1_0_1_metadata_url": "https://physionet.org/files/ptb-xl/1.0.1/ptbxl_database.csv",
        "challenge_ptbxl_model_input": False,
        "challenge_ptbxl_role": "aggregate_label_concordance_reference_only",
        "crosswalk_required": False,
        "reason": (
            "Original PTB-XL v1.0.1 is the development release. Challenge PTB-XL contributes only "
            "aggregate label-support evidence, so reverse record linkage is neither required nor permitted."
        ),
    }


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(_download_plan(), indent=2, sort_keys=True))
        return 0
    required = {"--header-root": args.header_root, "--ptbxl-metadata": args.ptbxl_metadata}
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"Required outside --dry-run: {', '.join(missing)}")

    protocol = _protocol()
    rules = protocol["prediction_task"]["labels"]["inclusion_rules"]
    records = scan_headers(args.header_root)
    audit = build_header_audit(
        records=records,
        scored_mapping_path=SCORED_MAPPING,
        ptbxl_metadata_csv=args.ptbxl_metadata,
        ptbxl_original_root=None,
        require_ptbxl_crosswalk=False,
        protocol_version=str(protocol["version"]),
        minimum_development_positives=int(rules["minimum_development_positive_records"]),
        minimum_external_positives=int(rules["minimum_external_positive_records_per_domain"]),
        minimum_external_domains=int(rules["minimum_external_domains_meeting_positive_threshold"]),
    )
    write_header_audit(audit, args.output)
    print(
        json.dumps(
            {
                "ready_for_label_lock": audit.ready_for_waveform_stage,
                "eligible_label_count": len(audit.eligible_labels),
                "eligible_labels": audit.eligible_labels,
                "challenge_ptbxl_model_input": False,
                "ptbxl_crosswalk_required": False,
                "blockers": audit.blockers,
                "manifest_sha256": audit.manifest_sha256,
                "output": str(Path(args.output).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if args.require_ready and not audit.ready_for_waveform_stage else 0


if __name__ == "__main__":
    raise SystemExit(main())
