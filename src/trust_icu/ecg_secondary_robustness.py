"""Deterministic signal perturbations for secondary TRUST-ECG robustness tests."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

LIMB_LEAD_INDICES = (0, 1, 2, 3, 4, 5)
PRECORDIAL_LEAD_INDICES = (6, 7, 8, 9, 10, 11)


def _validated_waveform(waveform: np.ndarray) -> np.ndarray:
    array = np.asarray(waveform, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("ECG waveform must have shape (leads, time).")
    if array.shape[0] != 12:
        raise ValueError("TRUST-ECG robustness analysis requires 12 leads.")
    if array.shape[1] < 2:
        raise ValueError("ECG waveform must contain at least two time samples.")
    if not np.isfinite(array).all():
        raise ValueError("ECG waveform must contain only finite values.")
    return array


def apply_gain(waveform: np.ndarray, factor: float) -> np.ndarray:
    """Scale all leads by a fixed gain factor."""

    array = _validated_waveform(waveform)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Gain factor must be finite and positive.")
    return array * float(factor)


def add_gaussian_noise(
    waveform: np.ndarray,
    *,
    snr_db: float = 20.0,
    seed: int = 20260808,
) -> np.ndarray:
    """Add deterministic white Gaussian noise at a requested global SNR."""

    array = _validated_waveform(waveform)
    if not np.isfinite(snr_db):
        raise ValueError("SNR must be finite.")
    rms_signal = float(np.sqrt(np.mean(np.square(array))))
    if rms_signal == 0.0:
        return array.copy()
    noise_rms = rms_signal / (10.0 ** (float(snr_db) / 20.0))
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(loc=0.0, scale=noise_rms, size=array.shape)
    return array + noise


def add_baseline_wander(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int = 500,
    amplitude_mv: float = 0.05,
    frequency_hz: float = 0.33,
) -> np.ndarray:
    """Add a fixed low-frequency sinusoidal baseline perturbation."""

    array = _validated_waveform(waveform)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    if amplitude_mv < 0 or not np.isfinite(amplitude_mv):
        raise ValueError("amplitude_mv must be finite and non-negative.")
    if frequency_hz <= 0 or not np.isfinite(frequency_hz):
        raise ValueError("frequency_hz must be finite and positive.")
    time = np.arange(array.shape[1], dtype=np.float64) / float(sample_rate_hz)
    wander = float(amplitude_mv) * np.sin(2.0 * np.pi * float(frequency_hz) * time)
    return array + wander[None, :]


def temporal_shift(waveform: np.ndarray, *, shift_samples: int) -> np.ndarray:
    """Shift a signal in time and zero-pad the newly exposed samples."""

    array = _validated_waveform(waveform)
    shift = int(shift_samples)
    if abs(shift) >= array.shape[1]:
        raise ValueError("Absolute shift must be smaller than signal length.")
    shifted = np.zeros_like(array)
    if shift > 0:
        shifted[:, shift:] = array[:, :-shift]
    elif shift < 0:
        shifted[:, :shift] = array[:, -shift:]
    else:
        shifted[:] = array
    return shifted


def dropout_leads(waveform: np.ndarray, lead_indices: Iterable[int]) -> np.ndarray:
    """Zero only the requested lead indices while preserving tensor shape."""

    array = _validated_waveform(waveform)
    indices = tuple(sorted(set(int(index) for index in lead_indices)))
    if not indices:
        raise ValueError("At least one lead index is required.")
    if indices[0] < 0 or indices[-1] >= array.shape[0]:
        raise ValueError("Lead index is outside the 12-lead signal.")
    dropped = array.copy()
    dropped[list(indices), :] = 0.0
    return dropped


def robustness_suite(waveform: np.ndarray, *, seed: int = 20260808) -> dict[str, np.ndarray]:
    """Generate the fixed secondary perturbation suite for one physical-mV ECG."""

    array = _validated_waveform(waveform)
    return {
        "gaussian_noise_snr20db": add_gaussian_noise(array, snr_db=20.0, seed=seed),
        "gain_0p9x": apply_gain(array, 0.9),
        "gain_1p1x": apply_gain(array, 1.1),
        "baseline_wander_0p05mv_0p33hz": add_baseline_wander(array),
        "time_shift_minus_100ms": temporal_shift(array, shift_samples=-50),
        "time_shift_plus_100ms": temporal_shift(array, shift_samples=50),
        "limb_lead_dropout": dropout_leads(array, LIMB_LEAD_INDICES),
        "precordial_lead_dropout": dropout_leads(array, PRECORDIAL_LEAD_INDICES),
    }
