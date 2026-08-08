from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trust_icu.ecg_data import EXPECTED_LEADS
from trust_icu.ecg_signal import (
    SignalSpec,
    StreamingLeadStats,
    digital_to_physical_mv,
    normalize_signal,
    parse_signal_header,
    standardize_signal,
)


def _specs(*, reversed_order: bool = False) -> tuple[SignalSpec, ...]:
    leads = tuple(reversed(EXPECTED_LEADS)) if reversed_order else EXPECTED_LEADS
    return tuple(
        SignalSpec(lead=lead, gain_per_unit=1000.0, baseline_digital=100.0, physical_unit="mV")
        for lead in leads
    )


def test_parse_signal_header_supports_wfdb_gain_baseline(tmp_path: Path) -> None:
    header = tmp_path / "A.hea"
    lines = ["A 12 500 5000"]
    for index, lead in enumerate(EXPECTED_LEADS):
        lines.append(f"A.mat 16+24 1000(100)/mV 16 0 0 {index} 0 {lead}")
    header.write_text("\n".join(lines) + "\n", encoding="utf-8")
    record_id, rate, samples, specs = parse_signal_header(header)
    assert record_id == "A"
    assert rate == 500
    assert samples == 5000
    assert specs[0].gain_per_unit == 1000
    assert specs[0].baseline_digital == 100
    assert specs[0].physical_unit == "mV"


def test_digital_conversion_reorders_leads_and_applies_gain() -> None:
    specs = _specs(reversed_order=True)
    digital = np.zeros((12, 3), dtype=float)
    for index, spec in enumerate(specs):
        canonical_index = EXPECTED_LEADS.index(spec.lead)
        digital[index] = 100.0 + 1000.0 * canonical_index
    physical = digital_to_physical_mv(digital, specs)
    for index in range(12):
        assert np.allclose(physical[index], float(index))


def test_standardization_resamples_250hz_to_500hz() -> None:
    digital = np.tile(np.linspace(100, 1100, 2500), (12, 1))
    standardized = standardize_signal(
        digital,
        _specs(),
        source_sampling_rate_hz=250,
    )
    assert standardized.waveform_mv.shape == (12, 5000)
    assert standardized.valid_mask.all()
    assert standardized.left_padding == 0
    assert standardized.right_padding == 0


def test_standardization_symmetrically_pads_short_recording() -> None:
    digital = np.full((12, 4000), 1100.0)
    standardized = standardize_signal(
        digital,
        _specs(),
        source_sampling_rate_hz=500,
    )
    assert standardized.left_padding == 500
    assert standardized.right_padding == 500
    assert standardized.valid_mask.sum() == 4000
    assert np.allclose(standardized.waveform_mv[:, :500], 0.0)
    assert np.allclose(standardized.waveform_mv[:, 500:4500], 1.0)


def test_standardization_center_crops_long_recording() -> None:
    digital = np.tile(np.arange(6000, dtype=float) + 100.0, (12, 1))
    standardized = standardize_signal(
        digital,
        _specs(),
        source_sampling_rate_hz=500,
    )
    assert standardized.crop_start_after_resampling == 500
    assert standardized.waveform_mv.shape == (12, 5000)
    assert np.isclose(standardized.waveform_mv[0, 0], 0.5)
    assert np.isclose(standardized.waveform_mv[0, -1], 5.499)


def test_streaming_stats_exclude_padding_and_normalization_resets_padding() -> None:
    stats_builder = StreamingLeadStats()
    waveform_a = np.tile(np.array([1.0, 3.0, 0.0, 0.0]), (12, 1))
    waveform_b = np.tile(np.array([5.0, 7.0, 0.0, 0.0]), (12, 1))
    mask = np.array([True, True, False, False])
    stats_builder.update(waveform_a, mask)
    stats_builder.update(waveform_b, mask)
    stats = stats_builder.finalize()
    assert np.allclose(stats.means_mv, 4.0)
    assert np.allclose(stats.stds_mv, np.sqrt(5.0))
    assert stats.valid_sample_counts == (4,) * 12
    normalized = normalize_signal(waveform_a, mask, stats)
    assert np.allclose(normalized[:, ~mask], 0.0)
    assert np.isfinite(normalized).all()
    assert len(stats.stats_sha256) == 64


def test_nonfinite_signal_is_rejected() -> None:
    digital = np.zeros((12, 5000), dtype=float)
    digital[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        standardize_signal(
            digital,
            _specs(),
            source_sampling_rate_hz=500,
        )


def test_non_mv_unit_is_rejected(tmp_path: Path) -> None:
    header = tmp_path / "A.hea"
    lines = ["A 12 500 5000"]
    for index, lead in enumerate(EXPECTED_LEADS):
        lines.append(f"A.mat 16+24 1000/uV 16 0 0 {index} 0 {lead}")
    header.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires mV"):
        parse_signal_header(header)
