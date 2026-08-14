from __future__ import annotations

import numpy as np

from trust_icu.ecg_statistical_core import (
    benjamini_hochberg,
    equal_frequency_calibration_bins,
    paired_binary_bootstrap,
    paired_macro_bootstrap,
)


def test_benjamini_hochberg_preserves_missing_values() -> None:
    observed = benjamini_hochberg([0.01, 0.03, None, 0.20])
    assert observed[2] is None
    assert np.allclose([observed[0], observed[1], observed[3]], [0.03, 0.045, 0.20])


def test_equal_frequency_bins_cover_every_record() -> None:
    targets = np.asarray([0] * 80 + [1] * 20, dtype=np.int64)
    probabilities = np.linspace(0.01, 0.99, targets.size)
    bins = equal_frequency_calibration_bins(targets, probabilities, bins=10)
    assert sum(int(row["n"]) for row in bins) == targets.size
    assert len(bins) == 10
    assert all(0.0 <= float(row["observed_wilson_low"]) <= 1.0 for row in bins)
    assert all(0.0 <= float(row["observed_wilson_high"]) <= 1.0 for row in bins)


def test_paired_binary_bootstrap_uses_positive_improvement_direction() -> None:
    targets = np.asarray([0] * 80 + [1] * 20, dtype=np.int64)
    candidate = np.where(targets == 1, 0.80, 0.05)
    reference = np.full(targets.size, 0.20)
    result = paired_binary_bootstrap(
        targets,
        candidate,
        reference,
        repeats=100,
        seed=7,
    )
    assert float(result["paired_improvement"]["pr_auc"]["median"]) > 0.0
    assert float(result["paired_improvement"]["brier"]["median"]) > 0.0


def test_paired_macro_bootstrap_returns_requested_repeats() -> None:
    first = np.asarray([0] * 80 + [1] * 20, dtype=np.int64)
    second = np.roll(first, 10)
    targets = np.column_stack([first, second])
    candidate = np.column_stack(
        [np.linspace(0.01, 0.99, first.size), np.roll(np.linspace(0.01, 0.99, first.size), 10)]
    )
    reference = np.full_like(candidate, 0.20)
    result = paired_macro_bootstrap(
        targets,
        candidate,
        reference,
        repeats=50,
        seed=11,
    )
    assert result["paired_improvement"]["pr_auc"]["n"] == 50
    assert result["bootstrap_attempts"] >= 50
