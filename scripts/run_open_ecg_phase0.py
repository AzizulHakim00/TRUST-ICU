#!/usr/bin/env python3
"""Run the locked TRUST-ECG Phase 0 reference or primary model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trust_icu.ecg_phase0 import (
    build_phase0_dry_run_plan,
    execute_logistic_reference_phase0,
    write_phase0_report,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "schemas/open_ecg_protocol.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("logistic", "resnet"), default="logistic")
    parser.add_argument("--challenge-training-root")
    parser.add_argument("--model-index")
    parser.add_argument("--model-index-audit")
    parser.add_argument("--label-manifest")
    parser.add_argument("--waveform-audit")
    parser.add_argument("--normalization-stats")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-root", default="open_ecg_phase0_outputs")
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _require(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not getattr(args, name)]
    if missing:
        formatted = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise SystemExit(f"Missing required arguments: {formatted}")


def main() -> int:
    args = parse_args()
    if args.dry_run:
        plan = build_phase0_dry_run_plan(args.protocol)
        plan["requested_model"] = args.model
        plan["resnet_optional_dependency_required"] = args.model == "resnet"
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    common = (
        "challenge_training_root",
        "model_index",
        "model_index_audit",
        "label_manifest",
    )
    _require(args, common)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.model == "logistic":
        report = execute_logistic_reference_phase0(
            challenge_training_root=args.challenge_training_root,
            index_csv=args.model_index,
            index_audit_path=args.model_index_audit,
            label_manifest_path=args.label_manifest,
            protocol_path=args.protocol,
        )
        report_path = output_root / "open_ecg_phase0_logistic_report.json"
    else:
        _require(args, ("waveform_audit", "normalization_stats"))
        from trust_icu.ecg_deep_phase0 import execute_fixed_resnet_phase0

        report = execute_fixed_resnet_phase0(
            challenge_training_root=args.challenge_training_root,
            index_csv=args.model_index,
            index_audit_path=args.model_index_audit,
            label_manifest_path=args.label_manifest,
            waveform_audit_path=args.waveform_audit,
            normalization_stats_path=args.normalization_stats,
            protocol_path=args.protocol,
            output_root=output_root,
            device_name=args.device,
            num_workers=args.num_workers,
        )
        report_path = output_root / "open_ecg_phase0_resnet_report.json"

    write_phase0_report(report, report_path)
    print(
        json.dumps(
            {
                "study": report.study,
                "model": report.model_name,
                "primary_gate_eligible": report.primary_gate_eligible,
                "external_recovery_pool_used": report.external_recovery_pool_used,
                "internal_macro_pr_auc": report.internal_test.macro_pr_auc,
                "report_sha256": report.report_sha256,
                "output": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
