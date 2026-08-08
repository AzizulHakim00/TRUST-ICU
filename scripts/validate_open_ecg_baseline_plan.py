#!/usr/bin/env python3
"""Validate and print the locked TRUST-ECG baseline/certification plan."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from trust_icu.ecg_baseline import FEATURE_STATISTICS, feature_names
from trust_icu.ecg_protocol import validate_open_ecg_protocol

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "schemas/open_ecg_protocol.yaml"


def main() -> int:
    protocol_report = validate_open_ecg_protocol(PROTOCOL)
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    models = protocol["phase0_models"]
    calibration = protocol["calibration"]
    gate = protocol["phase0_go_no_go"]
    partition = protocol["external_partition"]
    phase1 = protocol["phase1_if_phase0_passes"]
    payload = {
        "valid": True,
        "study": "TRUST-ECG",
        "protocol_version": protocol_report["version"],
        "protocol_sha256": protocol_report["protocol_sha256"],
        "primary_model": models["primary_model"],
        "architecture_search_allowed": models["architecture_search"]["allowed"],
        "development_roles": {
            "fit_folds": protocol_report["model_fit_folds"],
            "optimization_fold": protocol_report["optimization_validation_fold"],
            "calibration_fold": protocol_report["calibration_fold"],
            "internal_test_fold": protocol_report["internal_test_fold"],
        },
        "logistic_reference": {
            "feature_statistics": list(FEATURE_STATISTICS),
            "feature_count": len(feature_names()),
            "C": models["logistic_regression_handcrafted"]["C"],
            "class_weight": models["logistic_regression_handcrafted"]["class_weight"],
        },
        "calibration": {
            "method": calibration["method"],
            "fit_source": calibration["fit_source"],
            "external_recalibration_in_phase0": calibration["external_recalibration_in_phase0"],
        },
        "external_partition": {
            "method": partition["method"],
            "certification_fraction": partition["certification_fraction"],
            "recovery_pool_fraction": partition["recovery_pool_fraction"],
            "label_stratification": partition["label_stratification"],
        },
        "phase0_gate": {
            "unit": gate["evaluation_unit"],
            "minimum_positive_records": gate["minimum_positive_records_in_certification_partition"],
            "minimum_negative_records": gate["minimum_negative_records_in_certification_partition"],
            "minimum_pr_auc_to_prevalence_ratio": gate["discrimination_viability"][
                "minimum_pr_auc_to_prevalence_ratio"
            ],
            "maximum_absolute_calibration_slope_deviation": gate["calibration_envelope"][
                "maximum_absolute_calibration_slope_deviation"
            ],
            "maximum_absolute_calibration_intercept": gate["calibration_envelope"][
                "maximum_absolute_calibration_intercept"
            ],
            "require_positive_brier_skill_vs_prevalence": gate["calibration_envelope"][
                "require_positive_brier_skill_vs_prevalence"
            ],
        },
        "phase1": {
            "data_source": phase1["data_source"],
            "label_budgets": phase1["target_label_budgets"],
            "repeats": phase1["sampling"]["repeats"],
        },
        "execution_blocker": "real header, label, and waveform audits must pass before model execution",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
