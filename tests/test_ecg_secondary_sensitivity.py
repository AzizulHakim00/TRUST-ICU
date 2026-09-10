from trust_icu.ecg_secondary_sensitivity import (
    SecondaryEnvelope,
    classify_pair_secondary,
    evaluate_envelope_grid,
    evaluate_framework_ablations,
)


def _pair(
    *,
    pr_ratio: float = 2.1,
    slope: float = 0.80,
    intercept: float = 0.20,
    brier_skill: float = 0.08,
):
    return {
        "status": "certified",
        "metrics": {
            "n": 200,
            "positives": 80,
            "negatives": 120,
            "prevalence": 0.4,
            "pr_auc": pr_ratio * 0.4,
            "pr_auc_to_prevalence_ratio": pr_ratio,
            "roc_auc": 0.90,
            "brier": 0.12,
            "brier_skill_vs_prevalence": brier_skill,
            "calibration_slope": slope,
            "calibration_intercept": intercept,
        },
        "reasons": [],
    }


def test_official_secondary_envelope_reproduces_certified_status():
    assert classify_pair_secondary(_pair(), SecondaryEnvelope.official()) == "certified"


def test_official_secondary_envelope_detects_discrimination_failure_first():
    pair = _pair(pr_ratio=1.9, slope=0.1, intercept=3.0, brier_skill=-1.0)
    assert classify_pair_secondary(pair, SecondaryEnvelope.official()) == "discrimination_failure"


def test_official_secondary_envelope_detects_calibration_failure():
    pair = _pair(slope=0.5)
    assert classify_pair_secondary(pair, SecondaryEnvelope.official()) == "calibration_recovery_candidate"


def test_insufficient_support_report_state_remains_insufficient():
    pair = {
        "status": "insufficient_support",
        "metrics": None,
        "reasons": ["certification_partition_support_below_threshold"],
    }
    assert classify_pair_secondary(pair, SecondaryEnvelope.official()) == "insufficient_support"


def test_grid_keeps_one_official_reference_condition():
    rows = evaluate_envelope_grid({"georgia": {"59118001": _pair()}})
    official = [row for row in rows if row["is_official"]]
    assert len(official) == 1
    assert official[0]["pr_ratio_min"] == 2.0
    assert official[0]["slope_deviation_max"] == 0.35
    assert official[0]["intercept_abs_max"] == 0.75
    assert official[0]["require_positive_brier_skill"] is True
    assert official[0]["certified"] == 1


def test_grid_reports_all_fixed_secondary_settings():
    rows = evaluate_envelope_grid({"georgia": {"59118001": _pair()}})
    assert len(rows) == 4 * 3 * 3 * 2


def test_framework_ablation_does_not_mutate_input():
    matrix = {"georgia": {"59118001": _pair()}}
    before = repr(matrix)
    rows = evaluate_framework_ablations(matrix)
    assert rows
    assert repr(matrix) == before


def test_framework_ablation_contains_full_gate_and_each_calibration_removal():
    rows = evaluate_framework_ablations({"georgia": {"59118001": _pair()}})
    names = {row["ablation"] for row in rows}
    assert {
        "full_official_gate",
        "discrimination_only",
        "calibration_only",
        "without_slope",
        "without_intercept",
        "without_brier_skill",
    }.issubset(names)
