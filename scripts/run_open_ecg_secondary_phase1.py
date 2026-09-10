#!/usr/bin/env python3
"""Run low-compute aggregate-only TRUST-ECG secondary validation analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trust_icu.ecg_secondary_reporting import (
    load_and_verify_primary_reports,
    validate_secondary_output_root,
    write_secondary_bundle,
)
from trust_icu.ecg_secondary_sensitivity import (
    evaluate_envelope_grid,
    evaluate_framework_ablations,
)
from trust_icu.ecg_secondary_statistics import summarize_phase1_estimability


def _dry_run_plan() -> dict[str, Any]:
    return {
        "study": "TRUST-ECG",
        "stage": "secondary_validation_phase1_low_compute",
        "analysis_role": "secondary_only_primary_immutable",
        "primary_seed": 20260808,
        "envelope_grid": {
            "pr_auc_to_prevalence_ratio_min": [1.5, 2.0, 2.5, 3.0],
            "maximum_absolute_slope_deviation": [0.25, 0.35, 0.50],
            "maximum_absolute_intercept": [0.50, 0.75, 1.00],
            "positive_brier_skill_required": [True, False],
        },
        "framework_ablations": [
            "full_official_gate",
            "discrimination_only",
            "calibration_only",
            "without_slope",
            "without_intercept",
            "without_brier_skill",
        ],
        "aggregate_outputs": [
            "envelope_sensitivity",
            "framework_ablation",
            "phase1_estimability",
            "secondary_provenance",
            "secondary_sha256_manifest",
        ],
        "deferred_to_prediction_level_adapter": [
            "certification_uncertainty",
            "real_cross_partition_overlap_audit",
        ],
        "prohibited": [
            "primary_report_overwrite",
            "primary_model_replacement",
            "protocol_change",
            "record_level_output",
            "raw_waveform_output",
            "external_tuning",
        ],
    }


def _phase1_estimability_rows(phase1_report: dict[str, Any]) -> list[dict[str, Any]]:
    pair_results = phase1_report.get("pair_results")
    if not isinstance(pair_results, dict):
        raise ValueError("Phase-1 report is missing pair_results.")

    output: list[dict[str, Any]] = []
    for pair_key, pair_payload in sorted(pair_results.items()):
        if not isinstance(pair_payload, dict):
            raise ValueError(f"Invalid Phase-1 pair payload: {pair_key}")
        source = str(pair_payload.get("source"))
        label_code = str(pair_payload.get("label_code"))
        budgets = pair_payload.get("budgets")
        if not isinstance(budgets, dict):
            raise ValueError(f"Phase-1 pair is missing budgets: {pair_key}")
        for budget_text, budget_payload in sorted(budgets.items(), key=lambda item: int(item[0])):
            if not isinstance(budget_payload, dict):
                raise ValueError(f"Invalid Phase-1 budget payload: {pair_key}/{budget_text}")
            methods = budget_payload.get("methods")
            if not isinstance(methods, dict):
                raise ValueError(f"Phase-1 budget is missing methods: {pair_key}/{budget_text}")
            for method, summary in sorted(methods.items()):
                if not isinstance(summary, dict):
                    raise ValueError(
                        f"Invalid Phase-1 method summary: {pair_key}/{budget_text}/{method}"
                    )
                estimability = summarize_phase1_estimability(
                    repeats_requested=int(summary["repeats_requested"]),
                    estimable_repeats=int(summary["estimable_repeats"]),
                    recovered_repeats=int(summary["recovery_envelope_met_count"]),
                    nonestimable_reasons={
                        str(reason): int(count)
                        for reason, count in dict(summary["nonestimable_reasons"]).items()
                    },
                )
                output.append(
                    {
                        "source": source,
                        "label_code": label_code,
                        "budget": int(budget_text),
                        "method": str(method),
                        **estimability,
                    }
                )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--phase0-report")
    parser.add_argument("--phase1-report")
    parser.add_argument("--protocol", default="schemas/open_ecg_protocol.yaml")
    parser.add_argument(
        "--output-root",
        default="secondary_results/trust_ecg_v04/phase1_low_compute",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.dry_run:
        print(json.dumps(_dry_run_plan(), indent=2, sort_keys=True))
        return

    if not args.phase0_report or not args.phase1_report:
        raise SystemExit("--phase0-report and --phase1-report are required outside --dry-run.")

    phase0_path = Path(args.phase0_report).expanduser().resolve()
    phase1_path = Path(args.phase1_report).expanduser().resolve()
    output_root = validate_secondary_output_root(
        args.output_root,
        primary_roots=[phase0_path.parent, phase1_path.parent],
    )
    phase0, phase1, provenance = load_and_verify_primary_reports(
        phase0_report_path=phase0_path,
        phase1_report_path=phase1_path,
        protocol_path=args.protocol,
    )
    if phase1 is None:
        raise RuntimeError("Low-compute Phase-1 reporting requires the frozen Phase-1 report.")

    external = phase0.get("external_certification")
    if not isinstance(external, dict):
        raise ValueError("Phase-0 report is missing the external certification matrix.")

    payloads = {
        "envelope_sensitivity": evaluate_envelope_grid(external),
        "framework_ablation": evaluate_framework_ablations(external),
        "phase1_estimability": _phase1_estimability_rows(phase1),
    }
    manifest = write_secondary_bundle(
        output_root=output_root,
        payloads=payloads,
        provenance=provenance,
    )
    print(
        json.dumps(
            {
                "study": "TRUST-ECG",
                "stage": "secondary_validation_phase1_low_compute",
                "output_root": str(output_root),
                "artifact_count": len(manifest),
                "primary_phase0_report_sha256": provenance["primary_phase0_report_sha256"],
                "primary_phase1_report_sha256": provenance["primary_phase1_report_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
