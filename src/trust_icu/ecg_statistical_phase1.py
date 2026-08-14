"""Matched Phase-1 comparisons without persisting repeat-level results."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, wilcoxon

from trust_icu import ecg_phase1
from trust_icu.ecg_statistical_core import benjamini_hochberg, quantile_summary
from trust_icu.ecg_statistical_models import LABEL_NAMES


def safe_wilcoxon(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0 or np.allclose(array, 0.0):
        return None
    try:
        result = wilcoxon(array, zero_method="wilcox", alternative="two-sided", method="auto")
    except ValueError:
        return None
    return float(result.pvalue)


def paired_phase1_summary(
    candidate_records: list[dict[str, Any]],
    reference_records: list[dict[str, Any]],
) -> dict[str, Any]:
    paired = [
        (candidate, reference)
        for candidate, reference in zip(candidate_records, reference_records, strict=True)
        if candidate["estimable"] and reference["estimable"]
    ]
    if not paired:
        return {
            "paired_estimable_repeats": 0,
            "continuous_improvements": {},
            "recovery_envelope": {},
        }
    continuous = {
        "brier_reduction": [
            float(reference["metrics"]["brier"]) - float(candidate["metrics"]["brier"])
            for candidate, reference in paired
        ],
        "absolute_calibration_intercept_reduction": [
            abs(float(reference["metrics"]["calibration_intercept"]))
            - abs(float(candidate["metrics"]["calibration_intercept"]))
            for candidate, reference in paired
        ],
        "absolute_calibration_slope_deviation_reduction": [
            abs(float(reference["metrics"]["calibration_slope"]) - 1.0)
            - abs(float(candidate["metrics"]["calibration_slope"]) - 1.0)
            for candidate, reference in paired
        ],
        "pr_auc_delta": [
            float(candidate["metrics"]["pr_auc"]) - float(reference["metrics"]["pr_auc"])
            for candidate, reference in paired
        ],
    }
    candidate_only = sum(
        bool(candidate["recovery_envelope_met"]) and not bool(reference["recovery_envelope_met"])
        for candidate, reference in paired
    )
    reference_only = sum(
        bool(reference["recovery_envelope_met"]) and not bool(candidate["recovery_envelope_met"])
        for candidate, reference in paired
    )
    discordant = candidate_only + reference_only
    mcnemar_p = (
        float(binomtest(candidate_only, discordant, p=0.5, alternative="two-sided").pvalue)
        if discordant
        else None
    )
    return {
        "paired_estimable_repeats": len(paired),
        "continuous_improvements": {
            name: {
                **quantile_summary(values),
                "positive_win_rate": float(np.mean(np.asarray(values) > 0.0)),
                "wilcoxon_two_sided_p": safe_wilcoxon(values),
            }
            for name, values in continuous.items()
        },
        "recovery_envelope": {
            "candidate_successes": sum(
                bool(candidate["recovery_envelope_met"]) for candidate, _ in paired
            ),
            "reference_successes": sum(
                bool(reference["recovery_envelope_met"]) for _, reference in paired
            ),
            "candidate_only_successes": candidate_only,
            "reference_only_successes": reference_only,
            "paired_success_rate_difference": float(
                np.mean(
                    [
                        int(bool(candidate["recovery_envelope_met"]))
                        - int(bool(reference["recovery_envelope_met"]))
                        for candidate, reference in paired
                    ]
                )
            ),
            "mcnemar_exact_p": mcnemar_p,
        },
    }


def execute_phase1_with_matched_capture(
    *,
    phase0_report_path: str | Path,
    primary_data_root: str | Path,
    index_csv: str | Path,
    index_audit_path: str | Path,
    label_manifest_path: str | Path,
    normalization_stats_path: str | Path,
    checkpoint_path: str | Path,
    global_calibration_path: str | Path,
    protocol_path: str | Path,
    phase1_output_path: str | Path,
    device_name: str = "cpu",
    num_workers: int = 0,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Run frozen Phase 1 while capturing repeat records only in process memory."""

    plan = ecg_phase1.build_phase1_plan(phase0_report_path, protocol_path)
    captured: list[list[dict[str, Any]]] = []
    original = ecg_phase1._summarize_repeats

    def capture(records: list[dict[str, Any]]) -> dict[str, Any]:
        captured.append(records)
        return original(records)

    ecg_phase1._summarize_repeats = capture
    try:
        report = ecg_phase1.execute_phase1_probability_recovery(
            phase0_report_path=phase0_report_path,
            primary_data_root=primary_data_root,
            index_csv=index_csv,
            index_audit_path=index_audit_path,
            label_manifest_path=label_manifest_path,
            normalization_stats_path=normalization_stats_path,
            checkpoint_path=checkpoint_path,
            global_calibration_path=global_calibration_path,
            protocol_path=protocol_path,
            output_path=phase1_output_path,
            device_name=device_name,
            num_workers=num_workers,
        )
    finally:
        ecg_phase1._summarize_repeats = original

    expected_capture_count = len(plan.candidate_pairs) * len(plan.target_label_budgets) * len(plan.methods)
    if len(captured) != expected_capture_count:
        raise RuntimeError(
            f"Frozen Phase-1 capture count changed: {len(captured)} != {expected_capture_count}."
        )

    cursor = 0
    nested: dict[str, Any] = {}
    output_rows: list[dict[str, Any]] = []
    comparisons = (
        ("intercept_only_recalibration", "frozen_no_update"),
        ("platt_recalibration", "frozen_no_update"),
        ("platt_recalibration", "intercept_only_recalibration"),
    )
    for candidate in plan.candidate_pairs:
        pair_key = f"{candidate.source}/{candidate.label_code}"
        nested[pair_key] = {}
        for budget in plan.target_label_budgets:
            method_records: dict[str, list[dict[str, Any]]] = {}
            for method in plan.methods:
                method_records[method] = captured[cursor]
                cursor += 1
            nested[pair_key][str(budget)] = {}
            for candidate_method, reference_method in comparisons:
                key = f"{candidate_method}_vs_{reference_method}"
                summary = paired_phase1_summary(
                    method_records[candidate_method],
                    method_records[reference_method],
                )
                nested[pair_key][str(budget)][key] = summary
                for outcome, outcome_summary in summary["continuous_improvements"].items():
                    output_rows.append(
                        {
                            "source": candidate.source,
                            "label_code": candidate.label_code,
                            "label_name": LABEL_NAMES[candidate.label_code],
                            "budget": int(budget),
                            "candidate_method": candidate_method,
                            "reference_method": reference_method,
                            "outcome": outcome,
                            "paired_estimable_repeats": summary["paired_estimable_repeats"],
                            "improvement_mean": outcome_summary["mean"],
                            "improvement_median": outcome_summary["median"],
                            "improvement_q025": outcome_summary["q025"],
                            "improvement_q975": outcome_summary["q975"],
                            "positive_win_rate": outcome_summary["positive_win_rate"],
                            "p_value": outcome_summary["wilcoxon_two_sided_p"],
                        }
                    )
                recovery = summary["recovery_envelope"]
                output_rows.append(
                    {
                        "source": candidate.source,
                        "label_code": candidate.label_code,
                        "label_name": LABEL_NAMES[candidate.label_code],
                        "budget": int(budget),
                        "candidate_method": candidate_method,
                        "reference_method": reference_method,
                        "outcome": "recovery_envelope_success",
                        "paired_estimable_repeats": summary["paired_estimable_repeats"],
                        "improvement_mean": recovery.get("paired_success_rate_difference"),
                        "improvement_median": None,
                        "improvement_q025": None,
                        "improvement_q975": None,
                        "positive_win_rate": None,
                        "p_value": recovery.get("mcnemar_exact_p"),
                    }
                )
    q_values = benjamini_hochberg([row["p_value"] for row in output_rows])
    for row, q_value in zip(output_rows, q_values, strict=True):
        row["q_value_bh"] = q_value
    return report, nested, output_rows
