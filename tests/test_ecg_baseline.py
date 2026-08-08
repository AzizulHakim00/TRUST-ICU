from __future__ import annotations

import numpy as np

from trust_icu.ecg_baseline import (
    apply_platt_calibrators,
    certify_label_domain_pair,
    external_partition,
    extract_handcrafted_features,
    feature_names,
    fit_logistic_reference,
    fit_platt_calibrators,
    logistic_decision_scores,
    raw_scores_to_probabilities,
)


def test_handcrafted_features_have_locked_shape_and_ignore_padding() -> None:
    waveform = np.tile(np.array([1.0, 2.0, 3.0, 999.0, 999.0]), (12, 1))
    mask = np.array([True, True, True, False, False])
    features = extract_handcrafted_features(waveform, mask)
    assert features.shape == (144,)
    assert len(feature_names()) == 144
    assert np.isclose(features[0], 2.0)
    assert np.isclose(features[2], 1.0)
    assert np.isclose(features[3], 3.0)
    assert not np.any(np.isclose(features, 999.0))


def test_external_partition_is_deterministic_and_approximately_60_40() -> None:
    first = external_partition(source="georgia", record_id="G00001")
    second = external_partition(source="georgia", record_id="G00001")
    assert first == second
    assignments = [
        external_partition(source="georgia", record_id=f"G{index:05d}")
        for index in range(10000)
    ]
    certification_fraction = assignments.count("certification") / len(assignments)
    assert 0.57 <= certification_fraction <= 0.63


def test_logistic_reference_and_platt_calibration_have_locked_shapes() -> None:
    rng = np.random.default_rng(20260808)
    X = rng.normal(size=(400, 144))
    y = np.column_stack(
        [
            (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int),
            (X[:, 2] - 0.25 * X[:, 3] > 0).astype(int),
        ]
    )
    model = fit_logistic_reference(X, y, label_codes=("a", "b"))
    scores = logistic_decision_scores(model, X[:100])
    assert scores.shape == (100, 2)
    calibration_y = y[:100]
    calibrator = fit_platt_calibrators(scores, calibration_y, label_codes=("a", "b"))
    calibrated = apply_platt_calibrators(calibrator, scores)
    assert calibrated.shape == scores.shape
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))
    assert np.isfinite(calibrated).all()


def test_insufficient_support_is_not_silently_evaluated() -> None:
    y = np.array([1] * 20 + [0] * 180)
    probabilities = np.linspace(0.9, 0.1, y.size)
    result = certify_label_domain_pair(y, probabilities)
    assert result.status == "insufficient_support"
    assert result.metrics is None


def test_constant_probability_causes_discrimination_failure() -> None:
    y = np.array([1] * 100 + [0] * 100)
    probabilities = np.full(y.size, 0.5)
    result = certify_label_domain_pair(y, probabilities)
    assert result.status == "discrimination_failure"
    assert result.metrics is not None
    assert np.isclose(result.metrics.pr_auc_to_prevalence_ratio, 1.0)


def test_good_ranking_but_bad_calibration_becomes_recovery_candidate() -> None:
    y = np.array([1] * 100 + [0] * 100)
    probabilities = np.array([0.55] * 100 + [0.45] * 100)
    result = certify_label_domain_pair(y, probabilities)
    assert result.status == "calibration_recovery_candidate"
    assert result.metrics is not None
    assert result.metrics.pr_auc_to_prevalence_ratio >= 2.0
    assert result.reasons


def test_certified_state_requires_all_requested_checks() -> None:
    y = np.array([1] * 100 + [0] * 100)
    probabilities = np.array([0.8] * 100 + [0.2] * 100)
    result = certify_label_domain_pair(
        y,
        probabilities,
        minimum_pr_auc_to_prevalence_ratio=1.0,
        maximum_absolute_slope_deviation=100.0,
        maximum_absolute_intercept=100.0,
    )
    assert result.status == "certified"
    assert result.metrics is not None
    assert result.metrics.brier_skill_vs_prevalence > 0.0


def test_raw_score_sigmoid_is_finite_for_extreme_logits() -> None:
    probabilities = raw_scores_to_probabilities(np.array([[-1000.0, 0.0, 1000.0]]))
    assert np.isfinite(probabilities).all()
    assert probabilities[0, 0] < 1e-20
    assert np.isclose(probabilities[0, 1], 0.5)
    assert probabilities[0, 2] > 1.0 - 1e-20
