#!/usr/bin/env python3
"""Run aggregate-only TRUST-ECG paired model-comparison statistics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from trust_icu import ecg_statistical_models as models
from trust_icu.ecg_manifest import load_and_verify_label_manifest
from trust_icu.ecg_phase0_v04 import (
    load_and_verify_model_index,
    load_and_verify_normalization_stats,
)
from trust_icu.ecg_phase1 import build_phase1_plan, load_and_verify_phase0_report
from trust_icu.ecg_protocol import load_open_ecg_protocol
from trust_icu.ecg_statistical_core import write_csv
from trust_icu.ecg_statistical_plots import (
    calibration_records,
    plot_external_candidate_calibration,
    plot_internal_calibration,
    plot_internal_pr_auc_differences,
)
from trust_icu.ecg_statistical_reconstruction import (
    apply_phase0_calibration_payload,
    validate_resnet_report_metrics_audited,
)
from trust_icu.ecg_statistical_stages import (
    LOCKED_LABEL_CODES,
    build_model_stage_payload,
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
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_repeats < 500:
        raise SystemExit("At least 500 bootstrap replicates are required.")
    if args.calibration_bins < 5:
        raise SystemExit("At least five calibration bins are required.")
    if args.num_workers < 0:
        raise SystemExit("num-workers cannot be negative.")
    device = torch.device(args.device)
    if device.type != "cpu":
        raise SystemExit("The statistical model stage is locked to CPU.")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ["ADDENDUM_ROOT"] = str(output_root)
    phase0_path = Path(args.phase0_report).expanduser().resolve()
    report = load_and_verify_phase0_report(phase0_path, args.protocol)
    protocol = load_open_ecg_protocol(args.protocol)
    manifest = load_and_verify_label_manifest(args.label_manifest)
    rows, index_audit = load_and_verify_model_index(
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

    fit_rows = [row for row in rows if row.role == "model_fit"]
    calibration_rows = [row for row in rows if row.role == "calibration"]
    internal_rows = [row for row in rows if row.role == "internal_test"]
    external_rows = [row for row in rows if row.role == "external_certification"]
    if not all((fit_rows, calibration_rows, internal_rows, external_rows)):
        raise RuntimeError("Required frozen statistical roles are empty.")

    model, calibration_payload = models.load_verified_resnet(
        checkpoint_path=Path(args.checkpoint).expanduser().resolve(),
        calibration_path=Path(args.global_calibration).expanduser().resolve(),
        report=report,
        normalization_stats=normalization_stats,
        label_codes=label_codes,
        device=device,
    )
    models._global_calibrated_probabilities = apply_phase0_calibration_payload
    batch_size = int(protocol["phase0_models"]["resnet1d_fixed"]["batch_size"])
    data_root = Path(args.primary_data_root).expanduser().resolve()
    resnet_internal, internal_targets = models.collect_resnet_probabilities(
        model=model,
        rows=internal_rows,
        data_root=data_root,
        normalization_stats=normalization_stats,
        calibration_payload=calibration_payload,
        label_codes=label_codes,
        batch_size=batch_size,
        device=device,
        num_workers=args.num_workers,
    )
    resnet_external, external_targets = models.collect_resnet_probabilities(
        model=model,
        rows=external_rows,
        data_root=data_root,
        normalization_stats=normalization_stats,
        calibration_payload=calibration_payload,
        label_codes=label_codes,
        batch_size=batch_size,
        device=device,
        num_workers=args.num_workers,
    )
    validate_resnet_report_metrics_audited(
        report=report,
        internal_targets=internal_targets,
        internal_probabilities=resnet_internal,
        external_rows=external_rows,
        external_targets=external_targets,
        external_probabilities=resnet_external,
        label_codes=label_codes,
    )

    logistic_internal, logistic_external = models.fit_logistic_probabilities(
        data_root=data_root,
        fit_rows=fit_rows,
        calibration_rows=calibration_rows,
        internal_rows=internal_rows,
        external_rows=external_rows,
        label_codes=label_codes,
    )
    model_results, internal_table, external_table = models.paired_model_results(
        internal_targets=internal_targets,
        resnet_internal=resnet_internal,
        logistic_internal=logistic_internal,
        external_rows=external_rows,
        external_targets=external_targets,
        resnet_external=resnet_external,
        logistic_external=logistic_external,
        label_codes=label_codes,
        repeats=args.bootstrap_repeats,
    )
    plan = build_phase1_plan(phase0_path, args.protocol)
    candidate_pairs = {(item.source, item.label_code) for item in plan.candidate_pairs}
    bins_table = calibration_records(
        internal_targets=internal_targets,
        resnet_internal=resnet_internal,
        logistic_internal=logistic_internal,
        external_rows=external_rows,
        external_targets=external_targets,
        resnet_external=resnet_external,
        logistic_external=logistic_external,
        label_codes=label_codes,
        candidate_pairs=candidate_pairs,
        bins=args.calibration_bins,
    )

    write_csv(output_root / "paired_resnet_logistic_internal.csv", internal_table)
    write_csv(output_root / "paired_resnet_logistic_external.csv", external_table)
    write_csv(output_root / "calibration_bins.csv", bins_table)
    plot_internal_calibration(bins_table, output_root, label_codes)
    ordered_pairs = [(item.source, item.label_code) for item in plan.candidate_pairs]
    plot_external_candidate_calibration(bins_table, output_root, ordered_pairs)
    plot_internal_pr_auc_differences(internal_table, output_root)

    payload = build_model_stage_payload(
        protocol_version=str(report["protocol_version"]),
        protocol_sha256=str(report["protocol_sha256"]),
        phase0_report_sha256=str(report["report_sha256"]),
        phase0_model_sha256=str(report["model_sha256"]),
        model_index_sha256=str(report["model_index_sha256"]),
        label_manifest_sha256=str(report["label_manifest_sha256"]),
        normalization_stats_sha256=str(normalization_stats.stats_sha256),
        bootstrap_repeats=args.bootstrap_repeats,
        model_results=model_results,
    )
    digest = write_stage_payload(output_root / "model_comparison_stage.json", payload)
    print(
        json.dumps(
            {
                "stage": "model_comparison",
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
