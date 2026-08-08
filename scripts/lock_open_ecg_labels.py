#!/usr/bin/env python3
"""Lock the TRUST-ECG common label set from a verified aggregate header audit."""

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
    parser.add_argument("--header-audit", help="Verified open_ecg_header_audit.json.")
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
                        "header audit SHA-256 valid",
                        "ready_for_waveform_stage=true",
                        "Challenge/PTB-XL checksum crosswalk valid",
                        "protocol version match",
                    ],
                    "output": "aggregate-only deterministic label manifest",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.header_audit:
        raise SystemExit("--header-audit is required outside --dry-run")
    manifest = build_label_manifest(
        header_audit_path=args.header_audit,
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
