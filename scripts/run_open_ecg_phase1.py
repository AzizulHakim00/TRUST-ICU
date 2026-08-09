#!/usr/bin/env python3
"""Plan or execute conditional TRUST-ECG v0.4 Phase-1 probability recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trust_icu.ecg_phase1 import build_phase1_plan, execute_phase1_probability_recovery

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "schemas/open_ecg_protocol.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--phase0-report")
    parser.add_argument("--plan-output")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--primary-data-root")
    parser.add_argument("--model-index")
    parser.add_argument("--model-index-audit")
    parser.add_argument("--label-manifest")
    parser.add_argument("--normalization-stats")
    parser.add_argument("--checkpoint")
    parser.add_argument("--global-calibration")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _require(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not getattr(args, name)]
    if missing:
        formatted = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise SystemExit(f"Missing required arguments: {formatted}")


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "study": "TRUST-ECG",
                    "stage": "conditional_phase1_label_efficient_probability_recovery",
                    "protocol_version": "0.4.0",
                    "activation_requires": (
                        "verified fixed-ResNet Phase-0 report with recovery candidates "
                        "in at least two external domains"
                    ),
                    "recovery_pool_only": True,
                    "target_label_budgets": [0, 50, 100, 250, 500, 1000],
                    "repeats": 100,
                    "sampling": "uniform_without_replacement_no_label_stratification",
                    "allowed_methods": [
                        "frozen_no_update",
                        "intercept_only_recalibration",
                        "platt_recalibration",
                    ],
                    "prohibited": [
                        "target_domain_model_retraining",
                        "target_domain_feature_selection",
                        "target_domain_normalization_refit",
                        "posthoc_threshold_search",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    _require(args, ("phase0_report",))
    plan = build_phase1_plan(args.phase0_report, args.protocol)
    plan_payload = plan.to_dict()
    if args.plan_output:
        path = Path(args.plan_output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.plan_only:
        print(json.dumps(plan_payload, indent=2, sort_keys=True))
        return 0

    if not plan.activated:
        print(json.dumps(plan_payload, indent=2, sort_keys=True))
        return 0

    _require(
        args,
        (
            "primary_data_root",
            "model_index",
            "model_index_audit",
            "label_manifest",
            "normalization_stats",
            "checkpoint",
            "global_calibration",
            "output",
        ),
    )
    report = execute_phase1_probability_recovery(
        phase0_report_path=args.phase0_report,
        primary_data_root=args.primary_data_root,
        index_csv=args.model_index,
        index_audit_path=args.model_index_audit,
        label_manifest_path=args.label_manifest,
        normalization_stats_path=args.normalization_stats,
        checkpoint_path=args.checkpoint,
        global_calibration_path=args.global_calibration,
        protocol_path=args.protocol,
        output_path=args.output,
        device_name=args.device,
        num_workers=args.num_workers,
    )
    print(
        json.dumps(
            {
                "study": report["study"],
                "stage": report["stage"],
                "candidate_pair_count": report["candidate_pair_count"],
                "candidate_domain_count": report["candidate_domain_count"],
                "recovery_pool_only": report["recovery_pool_only"],
                "report_sha256": report["report_sha256"],
                "output": str(Path(args.output).expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
