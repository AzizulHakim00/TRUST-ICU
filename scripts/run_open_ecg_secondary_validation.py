#!/usr/bin/env python3
"""Assemble aggregate-only TRUST-ECG v0.4 secondary validation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trust_icu.ecg_secondary_reporting import write_secondary_aggregate_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase0-report", required=True)
    parser.add_argument("--phase1-report", required=True)
    parser.add_argument("--output-root", default="secondary_results/trust_ecg_v04")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "study": "TRUST-ECG",
                    "stage": "secondary_validation_aggregate_only",
                    "primary_analysis_unchanged": True,
                    "requires": [
                        "verified fixed-ResNet Phase-0 report",
                        "verified conditional Phase-1 recovery report",
                    ],
                    "outputs": [
                        "secondary_summary.json",
                        "envelope_sensitivity.csv",
                        "framework_ablation.csv",
                        "phase1_estimability.csv",
                        "secondary_sha256_manifest.json",
                    ],
                    "prohibited": [
                        "primary artifact overwrite",
                        "record-level predictions/logits",
                        "record/patient identifiers",
                        "raw waveforms",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    phase0 = _load_json(args.phase0_report)
    phase1 = _load_json(args.phase1_report)
    summary = write_secondary_aggregate_package(phase0, phase1, args.output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
