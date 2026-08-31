#!/usr/bin/env python3
"""Run aggregate-only TRUST-ECG matched Phase-1 recovery statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trust_icu.ecg_manifest import load_and_verify_label_manifest
from trust_icu.ecg_phase0_v04 import (
    load_and_verify_model_index,
    load_and_verify_normalization_stats,
)
from trust_icu.ecg_phase1 import build_phase1_plan, load_and_verify_phase0_report
from trust_icu.ecg_statistical_core import write_csv
from trust_icu.ecg_statistical_phase1 import execute_phase1_with_matched_capture
from trust_icu.ecg_statistical_plots import plot_phase1_recovery
from trust_icu.ecg_statistical_stages import (
    LOCKED_LABEL_CODES,
    build_phase1_stage_payload,
    write_stage_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0-report", required=True)
    parser.add_argument("--primary-data-root", required=True)
    parser.add_argument("--model-index", required=True)
    parser.add_argument("--model-index-audit", required=True)
    parser.add_argument("--label-manifest", required=True)
    parser.add_argument("--normalization-stats", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--global-calibration", required=True)
    parser.add_argument("--protocol", default="schemas/open_ecg_protocol.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device != "cpu":
        raise SystemExit("The matched Phase-1 stage is locked to CPU.")
    if args.num_workers < 0:
        raise SystemExit("num-workers cannot be negative.")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    phase0_path = Path(args.phase0_report).expanduser().resolve()
    report = load_and_verify_phase0_report(phase0_path, args.protocol)
    manifest = load_and_verify_label_manifest(args.label_manifest)
    _, index_audit = load_and_verify_model_index(
        index_csv=args.model_index,
        index_audit_path=args.model_index_audit,
    )
    normalization_stats = load_and_verify_normalization_stats(args.normalization_stats)
    label_codes = tuple(str(code) for code in report["label_codes"])
    if label_codes != LOCKED_LABEL_CODES:
        raise RuntimeError("Locked TRUST-ECG label order changed.")
    if tuple(str(code) for code in index_audit["label_codes"]) != label_codes:
        raise RuntimeError("Model-index label order differs from the primary report.")
    if str(index_audit["index_sha256"]) != str(report["model_index_sha256"]):
        raise RuntimeError("Model-index SHA-256 differs from the primary report.")
    if str(manifest["manifest_sha256"]) != str(report["label_manifest_sha256"]):
        raise RuntimeError("Label-manifest SHA-256 differs from the primary report.")

    phase1_path = output_root / "source_phase1_recovery_report.json"
    phase1_report, matched_results, phase1_rows = execute_phase1_with_matched_capture(
        phase0_report_path=phase0_path,
        primary_data_root=Path(args.primary_data_root).expanduser().resolve(),
        index_csv=args.model_index,
        index_audit_path=args.model_index_audit,
        label_manifest_path=args.label_manifest,
        normalization_stats_path=args.normalization_stats,
        checkpoint_path=args.checkpoint,
        global_calibration_path=args.global_calibration,
        protocol_path=args.protocol,
        phase1_output_path=phase1_path,
        device_name="cpu",
        num_workers=args.num_workers,
    )
    plan = build_phase1_plan(phase0_path, args.protocol)
    (output_root / "source_phase1_plan.json").write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_root / "phase1_matched_method_comparisons.csv", phase1_rows)
    plot_phase1_recovery(phase1_report, output_root)

    payload = build_phase1_stage_payload(
        protocol_version=str(report["protocol_version"]),
        protocol_sha256=str(report["protocol_sha256"]),
        phase0_report_sha256=str(report["report_sha256"]),
        phase0_model_sha256=str(report["model_sha256"]),
        model_index_sha256=str(report["model_index_sha256"]),
        label_manifest_sha256=str(report["label_manifest_sha256"]),
        normalization_stats_sha256=str(normalization_stats.stats_sha256),
        phase1_report_sha256=str(phase1_report["report_sha256"]),
        matched_results=matched_results,
    )
    digest = write_stage_payload(output_root / "phase1_stage.json", payload)
    print(
        json.dumps(
            {
                "stage": "phase1_matched",
                "stage_sha256": digest,
                "output_root": str(output_root),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
