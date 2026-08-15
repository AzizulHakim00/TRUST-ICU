from __future__ import annotations

import numpy as np
from scipy.special import expit

from trust_icu.ecg_statistical_reconstruction import (
    _metric_match,
    apply_phase0_calibration_payload,
)


def test_phase0_calibration_payload_matches_expit_contract() -> None:
    logits = np.asarray([[-2.0, 0.5], [1.5, -0.25]], dtype=np.float64)
    payload = {
        "parameters": {
            "a": {"coefficient": 1.2, "intercept": -0.3},
            "b": {"coefficient": -0.7, "intercept": 0.4},
        }
    }
    observed = apply_phase0_calibration_payload(logits, payload, ("a", "b"))
    expected = np.column_stack(
        [
            expit(1.2 * logits[:, 0] - 0.3),
            expit(-0.7 * logits[:, 1] + 0.4),
        ]
    )
    assert np.array_equal(observed, expected)


def test_reconstruction_tolerance_accepts_tiny_brier_drift() -> None:
    within, delta, allowed = _metric_match("brier", 0.0208312, 0.0208309)
    assert within is True
    assert delta < allowed


def test_reconstruction_tolerance_rejects_material_brier_drift() -> None:
    within, delta, allowed = _metric_match("brier", 0.0220, 0.0208)
    assert within is False
    assert delta > allowed


def test_reconstruction_support_counts_remain_exact() -> None:
    assert _metric_match("positives", 50.0, 50.0)[0] is True
    assert _metric_match("positives", 51.0, 50.0)[0] is False
