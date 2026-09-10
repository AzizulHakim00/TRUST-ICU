from __future__ import annotations

import numpy as np
from trust_icu.ecg_secondary_overlap import audit_cross_partition_overlap
from trust_icu.ecg_secondary_robustness import (
    add_baseline_wander,
    add_gaussian_noise,
    apply_gain,
    dropout_leads,
    temporal_shift,
)
from trust_icu.ecg_secondary_sensitivity import (
    EnvelopeDefinition,
    classify_metric_payload,
    envelope_sensitivity_rows,
    framework_gate_ablation_rows,
)
from trust_icu.ecg_secondary_statistics import (
    aggregate_seed_stability,
    phase1_estimability_rows,
    wilson_interval,
)


def _metrics(
    *,
    positives: int = 100,
    negatives: int = 100,
    ratio: float = 3.0,
    slope: float = 1.0,
    intercept: float = 0.0,
    brier_skill: float = 0.25,
) -> dict[str, float | int]:
    return {
        "n": positives + negatives,
        "positives": positives,
        "negatives": negatives,
        "prevalence": positives / (positives + negatives),
        "pr_auc": 0.75,
        "pr_auc_to_prevalence_ratio": ratio,
        "roc_auc": 0.9,
        "brier": 0.1,
        "brier_skill_vs_prevalence": brier_skill,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def _phase0_report() -> dict[str, object]:
    return {
        "label_codes": ["A", "B", "C", "D"],
        "external_certification": {
            "georgia": {
                "A": {"status": "certified", "metrics": _metrics()},
                "B": {
                    "status": "calibration_recovery_candidate",
                    "metrics": _metrics(intercept=1.1),
                },
                "C": {
                    "status": "discrimination_failure",
                    "metrics": _metrics(ratio=1.5),
                },
                "D": {"status": "insufficient_support", "metrics": None},
            }
        },
    }


def test_official_envelope_reproduces_pair_statuses() -> None:
    report = _phase0_report()
    official = EnvelopeDefinition()
    pairs = report["external_certification"]["georgia"]  # type: ignore[index]
    observed = {
        code: classify_metric_payload(pair["metrics"], official)  # type: ignore[index]
        for code, pair in pairs.items()  # type: ignore[union-attr]
    }
    expected = {code: pair["status"] for code, pair in pairs.items()}  # type: ignore[union-attr]
    assert observed == expected


def test_envelope_grid_contains_official_row_and_expected_counts() -> None:
    rows = envelope_sensitivity_rows(_phase0_report())
    official_rows = [
        row
        for row in rows
        if row["minimum_pr_auc_to_prevalence_ratio"] == 2.0
        and row["maximum_absolute_slope_deviation"] == 0.35
        and row["maximum_absolute_intercept"] == 0.75
        and row["require_positive_brier_skill"] is True
    ]
    assert len(rows) == 72
    assert len(official_rows) == 1
    assert official_rows[0]["certified"] == 1
    assert official_rows[0]["calibration_recovery_candidate"] == 1
    assert official_rows[0]["discrimination_failure"] == 1
    assert official_rows[0]["insufficient_support"] == 1


def test_framework_ablation_can_remove_individual_calibration_criteria() -> None:
    rows = framework_gate_ablation_rows(_phase0_report())
    by_name = {row["ablation"]: row for row in rows}
    assert by_name["official_full_gate"]["calibration_recovery_candidate"] == 1
    assert by_name["minus_intercept_criterion"]["certified"] == 2
    assert by_name["discrimination_only"]["certified"] == 2


def test_robustness_perturbations_are_deterministic_and_shape_safe() -> None:
    waveform = np.linspace(-1.0, 1.0, 12 * 5000, dtype=np.float64).reshape(12, 5000)
    noisy_a = add_gaussian_noise(waveform, snr_db=20.0, seed=17)
    noisy_b = add_gaussian_noise(waveform, snr_db=20.0, seed=17)
    assert noisy_a.shape == waveform.shape
    assert np.array_equal(noisy_a, noisy_b)
    assert not np.array_equal(noisy_a, waveform)

    assert np.allclose(apply_gain(waveform, 1.1), waveform * 1.1)
    wandered = add_baseline_wander(waveform, sample_rate_hz=500, amplitude_mv=0.05)
    assert wandered.shape == waveform.shape
    assert np.isfinite(wandered).all()

    shifted = temporal_shift(waveform, shift_samples=50)
    assert shifted.shape == waveform.shape
    assert np.allclose(shifted[:, :50], 0.0)
    assert np.allclose(shifted[:, 50:], waveform[:, :-50])

    dropped = dropout_leads(waveform, [0, 5, 11])
    assert np.allclose(dropped[[0, 5, 11]], 0.0)
    assert np.array_equal(dropped[1], waveform[1])


def test_overlap_audit_detects_exact_duplicate_without_exposing_records() -> None:
    rng = np.random.default_rng(5)
    first = [rng.normal(size=(12, 500)) for _ in range(3)]
    second = [rng.normal(size=(12, 500)), first[1].copy()]
    result = audit_cross_partition_overlap(first, second, near_duplicate_threshold=0.995)
    assert result["certification_records"] == 3
    assert result["recovery_records"] == 2
    assert result["exact_cross_partition_duplicates"] == 1
    assert result["maximum_near_duplicate_similarity"] >= 0.999
    serialized = repr(result).lower()
    assert "record_id" not in serialized
    assert "waveform" not in serialized


def test_wilson_interval_and_phase1_estimability_preserve_missing_repeats() -> None:
    lower, upper = wilson_interval(90, 100)
    assert 0.82 < lower < 0.84
    assert 0.95 < upper < 0.97

    phase1 = {
        "phase1_plan": {"repeats": 100},
        "pair_results": {
            "georgia/A": {
                "source": "georgia",
                "label_code": "A",
                "budgets": {
                    "50": {
                        "methods": {
                            "platt_recalibration": {
                                "estimable_repeats": 92,
                                "nonestimable_repeats": 8,
                                "nonestimable_reasons": {"single_class_adaptation_sample": 8},
                                "recovery_envelope_met_count": 83,
                                "recovery_envelope_met_rate_among_estimable": 83 / 92,
                            }
                        }
                    }
                },
            }
        },
    }
    rows = phase1_estimability_rows(phase1)
    assert rows[0]["repeats_requested"] == 100
    assert rows[0]["estimable_repeats"] == 92
    assert rows[0]["nonestimable_repeats"] == 8
    assert rows[0]["nonestimable_reasons"] == {"single_class_adaptation_sample": 8}
    assert rows[0]["recovery_rate_wilson_low"] < rows[0]["recovery_success_rate"]
    assert rows[0]["recovery_rate_wilson_high"] > rows[0]["recovery_success_rate"]


def test_seed_stability_keeps_primary_seed_fixed() -> None:
    def report(seed: int, b_status: str) -> dict[str, object]:
        return {
            "random_seed": seed,
            "internal_test": {
                "macro_pr_auc": 0.75 + (seed - 20260808) * 0.01,
                "macro_roc_auc": 0.95,
                "macro_brier": 0.02,
            },
            "external_certification": {
                "georgia": {
                    "A": {"status": "certified"},
                    "B": {"status": b_status},
                }
            },
        }

    summary = aggregate_seed_stability(
        {
            20260808: report(20260808, "calibration_recovery_candidate"),
            20260809: report(20260809, "calibration_recovery_candidate"),
            20260810: report(20260810, "certified"),
        }
    )
    assert summary["primary_seed"] == 20260808
    assert summary["replication_seeds"] == [20260809, 20260810]
    assert summary["primary_seed_replaced"] is False
    assert summary["pair_status_agreement_with_primary"]["20260809"] == 1.0
    assert summary["pair_status_agreement_with_primary"]["20260810"] == 0.5
