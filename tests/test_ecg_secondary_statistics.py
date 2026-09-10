import numpy as np
import pytest

from trust_icu.ecg_secondary_statistics import (
    bootstrap_gate_uncertainty,
    summarize_phase1_estimability,
    wilson_rate_interval,
)


def _well_separated_example():
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    p = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.70, 0.78, 0.85, 0.90, 0.95])
    return y, p


def test_gate_uncertainty_is_deterministic():
    y, p = _well_separated_example()
    first = bootstrap_gate_uncertainty(y, p, repeats=50, seed=17)
    second = bootstrap_gate_uncertainty(y, p, repeats=50, seed=17)
    assert first == second
    assert first["valid_bootstrap_repeats"] == 50
    assert 0.0 <= first["gate_satisfaction_rate"] <= 1.0


def test_gate_uncertainty_reports_all_gate_metrics():
    y, p = _well_separated_example()
    result = bootstrap_gate_uncertainty(y, p, repeats=30, seed=9)
    assert set(result["metric_intervals"]) == {
        "pr_auc_to_prevalence_ratio",
        "calibration_slope",
        "calibration_intercept",
        "brier_skill_vs_prevalence",
    }
    for summary in result["metric_intervals"].values():
        assert summary["n"] == 30
        assert summary["q025"] <= summary["median"] <= summary["q975"]


def test_wilson_rate_interval_bounds_rate():
    low, high = wilson_rate_interval(90, 100)
    assert low < 0.9 < high


def test_phase1_estimability_preserves_nonestimable_counts():
    summary = summarize_phase1_estimability(
        repeats_requested=100,
        estimable_repeats=83,
        recovered_repeats=75,
        nonestimable_reasons={"single_class_evaluation": 17},
    )
    assert summary["nonestimable_repeats"] == 17
    assert summary["recovery_rate_among_estimable"] == 75 / 83
    assert summary["nonestimable_reasons"] == {"single_class_evaluation": 17}
    assert summary["recovery_rate_wilson_low"] < 75 / 83 < summary["recovery_rate_wilson_high"]


def test_phase1_estimability_rejects_inconsistent_counts():
    with pytest.raises(ValueError, match="counts"):
        summarize_phase1_estimability(
            repeats_requested=100,
            estimable_repeats=90,
            recovered_repeats=91,
            nonestimable_reasons={"single_class_evaluation": 10},
        )


def test_gate_uncertainty_rejects_single_class_targets():
    y = np.zeros(10, dtype=np.int64)
    p = np.linspace(0.1, 0.9, 10)
    with pytest.raises(ValueError, match="both classes"):
        bootstrap_gate_uncertainty(y, p, repeats=10, seed=1)
