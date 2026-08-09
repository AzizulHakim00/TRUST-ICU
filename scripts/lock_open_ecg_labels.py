#!/usr/bin/env python3
"""Lock TRUST-ECG labels from Challenge support and original PTB-XL concordance evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trust_icu.ecg_manifest import build_label_manifest, write_label_manifest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "schemas/open_ecg_protocol.yaml"
MAPPING = ROOT / "schemas/challenge2020_scored_classes.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header-audit", help="Verified Challenge label-support audit JSON.")
    parser.add_argument(
        "--ptbxl-label-concordance",
        help="Verified original PTB-XL label-concordance audit JSON.",
    )
    parser.add_argument("--output", default="open_ecg_label_manifest.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "study": "TRUST-ECG",
                    "stage": "lock_common_label_manifest_before_waveform_training",
                    "requires": [
                        "Challenge v1.0.2 aggregate label-support audit SHA-256 valid",
                        "Challenge header/source/label-support gates ready",
                        "original PTB-XL v1.0.1 label-concordance audit SHA-256 valid",
                        "all seven PTB-XL SCP unions exactly concordant",
                        "protocol v0.4.0 exact canonical mapping match",
                        "Challenge PTB-XL prohibited as model input",
                    ],
                    "output": "aggregate-only deterministic two-source label manifest",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    missing = []
    if not args.header_audit:
        missing.append("--header-audit")
    if not args.ptbxl_label_concordance:
        missing.append("--ptbxl-label-concordance")
    if missing:
        raise SystemExit(f"Required outside --dry-run: {', '.join(missing)}")
    manifest = build_label_manifest(
        header_audit_path=args.header_audit,
        ptbxl_label_concordance_path=args.ptbxl_label_concordance,
        protocol_path=PROTOCOL,
        scored_mapping_path=MAPPING,
    )
    report = write_label_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "label_count": report.label_count,
                "canonical_codes": report.canonical_codes,
                "source_header_audit_sha256": report.source_header_audit_sha256,
                "ptbxl_label_concordance_audit_sha256": report.ptbxl_label_concordance_audit_sha256,
                "manifest_sha256": report.manifest_sha256,
                "output": report.output_path,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
