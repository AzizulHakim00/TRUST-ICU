import numpy as np
import pandas as pd
import pytest

from trust_icu.baseline import assert_matrix_is_leakage_safe, fit_logistic_baseline


def _matrix(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(65, 12, n)
    lactate = rng.lognormal(0.3, 0.5, n)
    logits = -4.0 + 0.03 * age + 0.8 * lactate
    target = rng.binomial(1, 1 / (1 + np.exp(-logits)))
    return pd.DataFrame(
        {
            "age": age,
            "lactate__last": lactate,
            "lactate__missing": 0,
            "sex": rng.choice(["female", "male"], n),
            "y": target,
        }
    )


def test_logistic_baseline_runs_on_synthetic_data() -> None:
    train = _matrix(600, 1)
    test = _matrix(300, 2)
    _, probabilities, metrics = fit_logistic_baseline(
        train,
        test,
        target_column="y",
        data_classification="synthetic",
    )
    assert len(probabilities) == len(test)
    assert 0 <= metrics.pr_auc <= 1
    assert 0 <= metrics.brier <= 1
    assert metrics.pr_auc_prevalence_ratio > 0
    assert np.isfinite(metrics.calibration_slope)
    assert np.isfinite(metrics.calibration_intercept)


def test_identifier_is_blocked() -> None:
    frame = _matrix(50, 3)
    frame["stay_id"] = range(len(frame))
    with pytest.raises(ValueError, match="Prohibited"):
        assert_matrix_is_leakage_safe(frame, target_column="y")


def test_future_metadata_is_blocked() -> None:
    frame = _matrix(50, 3)
    frame["icu_discharge_time"] = 1
    with pytest.raises(ValueError, match="Prohibited"):
        assert_matrix_is_leakage_safe(frame, target_column="y")


def test_unapproved_real_data_classification_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="allowed only"):
        fit_logistic_baseline(
            _matrix(100, 4),
            _matrix(100, 5),
            target_column="y",
            data_classification="raw_credentialed",  # type: ignore[arg-type]
        )
