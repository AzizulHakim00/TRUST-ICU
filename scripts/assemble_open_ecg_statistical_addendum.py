#!/usr/bin/env python3
"""Assemble the split TRUST-ECG aggregate-only statistical publication addendum."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from trust_icu.ecg_phase1 import load_and_verify_phase0_report
from trust_icu.ecg_statistical_core import canonical_hash, json_ready, write_manifest
from trust_icu.ecg_statistical_plots import write_summary
from trust_icu.ecg_statistical_stages import (
    LOCKED_LABEL_CODES,
    load_stage_payload,
    verify_common_identity,
)

PRIMARY_SEED = 20260808


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-stage-root", required=True)
    parser.add_argument("--phase1-stage-root", required=True)
    parser.add_argument("--phase0-report", required=True)
    parser.add_argument("--training-history", required=True)
    parser.add_argument("--protocol", default="schemas/open_ecg_protocol.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser.parse_args()


def _copy_stage_files(source: Path, target: Path, excluded: set[str]) -> None:
    for path in source.iterdir():
        if not path.is_file() or path.name in excluded:
            continue
        destination = target / path.name
        if destination.exists():
            if destination.read_bytes() != path.read_bytes():
                raise RuntimeError(f"Conflicting aggregate stage file: {path.name}")
            continue
        shutil.copy2(path, destination)


def _read_phase1_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            normalized: dict[str, Any] = dict(row)
            normalized["q_value_bh"] = None if row.get("q_value_bh", "") == "" else float(row["q_value_bh"])
            rows.append(normalized)
    return rows


def main() -> int:
    args = parse_args()
    if args.bootstrap_repeats < 500:
        raise SystemExit("At least 500 bootstrap replicates are required.")
    if args.calibration_bins < 5:
        raise SystemExit("At least five calibration bins are required.")

    model_root = Path(args.model_stage_root).expanduser().resolve()
    phase1_root = Path(args.phase1_stage_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model_stage = load_stage_payload(model_root / "model_comparison_stage.json", "model_comparison_statistics")
    phase1_stage = load_stage_payload(phase1_root / "phase1_stage.json", "phase1_matched_statistics")
    verify_common_identity(model_stage, phase1_stage)
    if int(model_stage["bootstrap_repeats"]) != int(args.bootstrap_repeats):
        raise RuntimeError("Bootstrap repeat count differs from the model-comparison stage.")

    phase0_path = Path(args.phase0_report).expanduser().resolve()
    phase0_report = load_and_verify_phase0_report(phase0_path, args.protocol)
    if tuple(str(code) for code in phase0_report["label_codes"]) != LOCKED_LABEL_CODES:
        raise RuntimeError("Locked TRUST-ECG label order changed.")
    for field, report_key in (
        ("protocol_version", "protocol_version"),
        ("protocol_sha256", "protocol_sha256"),
        ("phase0_report_sha256", "report_sha256"),
        ("phase0_model_sha256", "model_sha256"),
        ("model_index_sha256", "model_index_sha256"),
        ("label_manifest_sha256", "label_manifest_sha256"),
    ):
        if str(model_stage[field]) != str(phase0_report[report_key]):
            raise RuntimeError(f"Phase-0 identity mismatch during assembly: {field}")

    phase1_report_path = phase1_root / "source_phase1_recovery_report.json"
    phase1_report = json.loads(phase1_report_path.read_text(encoding="utf-8"))
    observed_phase1_hash = str(phase1_report.get("report_sha256", ""))
    if not observed_phase1_hash or canonical_hash(phase1_report, "report_sha256") != observed_phase1_hash:
        raise RuntimeError("Phase-1 report SHA-256 verification failed.")
    if observed_phase1_hash != str(phase1_stage["phase1_report_sha256"]):
        raise RuntimeError("Phase-1 report does not match the Phase-1 stage envelope.")

    required_model = {
        "paired_resnet_logistic_internal.csv",
        "paired_resnet_logistic_external.csv",
        "calibration_bins.csv",
        "resnet_metric_reconstruction_audit.csv",
        "internal_calibration.png",
        "internal_calibration.pdf",
        "external_candidate_calibration.png",
        "external_candidate_calibration.pdf",
        "internal_paired_pr_auc_differences.png",
        "internal_paired_pr_auc_differences.pdf",
    }
    required_phase1 = {
        "phase1_matched_method_comparisons.csv",
        "phase1_recovery_curves.png",
        "phase1_recovery_curves.pdf",
        "source_phase1_recovery_report.json",
        "source_phase1_plan.json",
    }
    missing = sorted(name for name in required_model if not (model_root / name).is_file())
    missing += sorted(name for name in required_phase1 if not (phase1_root / name).is_file())
    if missing:
        raise RuntimeError(f"Missing split statistical stage files: {missing}")

    _copy_stage_files(model_root, output_root, {"model_comparison_stage.json"})
    _copy_stage_files(phase1_root, output_root, {"phase1_stage.json"})
    (output_root / "source_phase0_resnet_report.json").write_bytes(phase0_path.read_bytes())
    history_path = Path(args.training_history).expanduser().resolve()
    (output_root / "source_resnet_training_history.json").write_bytes(history_path.read_bytes())

    phase1_rows = _read_phase1_rows(output_root / "phase1_matched_method_comparisons.csv")
    write_summary(
        output_root=output_root,
        model_results=model_stage["paired_resnet_vs_logistic"],
        phase1_rows=phase1_rows,
        bootstrap_repeats=args.bootstrap_repeats,
    )

    payload: dict[str, Any] = {
        "report_version": "0.1.0",
        "study": "TRUST-ECG",
        "stage": "aggregate_only_statistical_publication_addendum",
        "protocol_version": phase0_report["protocol_version"],
        "protocol_sha256": phase0_report["protocol_sha256"],
        "phase0_report_sha256": phase0_report["report_sha256"],
        "phase0_model_sha256": phase0_report["model_sha256"],
        "phase1_report_sha256": phase1_report["report_sha256"],
        "model_index_sha256": phase0_report["model_index_sha256"],
        "label_manifest_sha256": phase0_report["label_manifest_sha256"],
        "normalization_stats_sha256": model_stage["normalization_stats_sha256"],
        "bootstrap": {
            "repeats": args.bootstrap_repeats,
            "seed": PRIMARY_SEED,
            "internal_macro_unit": "ordinary_record_bootstrap",
            "per_label_unit": "class_stratified_record_bootstrap",
            "paired_models": True,
            "percentile_interval": [0.025, 0.975],
            "multiple_comparison_control": "Benjamini-Hochberg",
        },
        "paired_resnet_vs_logistic": model_stage["paired_resnet_vs_logistic"],
        "matched_phase1_method_comparisons": phase1_stage["matched_phase1_method_comparisons"],
        "calibration": {
            "binning": "equal_frequency",
            "bins_requested": args.calibration_bins,
            "observed_interval": "Wilson 95 percent",
        },
        "privacy": {
            "record_level_output_persisted": False,
            "patient_identifiers_persisted": False,
            "raw_predictions_persisted": False,
            "raw_logits_persisted": False,
            "waveforms_persisted": False,
            "adaptation_indices_persisted": False,
            "checkpoint_in_addendum_artifact": False,
        },
        "report_sha256": "",
    }
    payload["report_sha256"] = canonical_hash(payload, "report_sha256")
    (output_root / "trust_ecg_statistical_addendum.json").write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = write_manifest(output_root)
    print(json.dumps({"stage": payload["stage"], "report_sha256": payload["report_sha256"], "aggregate_file_count": len(manifest), "output_root": str(output_root)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
