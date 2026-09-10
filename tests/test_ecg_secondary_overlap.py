import numpy as np

from trust_icu.ecg_secondary_overlap import (
    cosine_similarity,
    summarize_cross_partition_overlap,
    waveform_fingerprint,
)


def test_exact_duplicate_is_detected_without_identifier_output():
    x = np.arange(120, dtype=float).reshape(12, 10)
    summary = summarize_cross_partition_overlap([x], [x.copy()], similarity_threshold=0.9999)
    assert summary["exact_duplicate_pairs"] == 1
    assert summary["near_duplicate_pairs"] >= 1
    assert summary["audit_status"] == "cross_partition_overlap_detected"
    assert "record_id" not in repr(summary)


def test_fingerprint_is_deterministic_and_compact():
    x = np.ones((12, 5000), dtype=float)
    first = waveform_fingerprint(x)
    second = waveform_fingerprint(x)
    assert np.array_equal(first, second)
    assert first.ndim == 1
    assert first.size <= 512


def test_dissimilar_signals_do_not_create_exact_duplicate():
    a = np.tile(np.linspace(-1.0, 1.0, 100), (12, 1))
    b = np.tile(np.sin(np.linspace(0.0, 8.0, 100)), (12, 1))
    summary = summarize_cross_partition_overlap([a], [b], similarity_threshold=0.99999)
    assert summary["exact_duplicate_pairs"] == 0
    assert 0.0 <= summary["maximum_cross_partition_similarity"] <= 1.0


def test_cosine_similarity_has_safe_zero_vector_behavior():
    assert cosine_similarity(np.zeros(5), np.ones(5)) == 0.0


def test_invalid_waveform_shape_is_rejected():
    bad = np.ones((11, 5000))
    try:
        waveform_fingerprint(bad)
    except ValueError as exc:
        assert "12" in str(exc)
    else:
        raise AssertionError("waveform with the wrong lead count was accepted")
