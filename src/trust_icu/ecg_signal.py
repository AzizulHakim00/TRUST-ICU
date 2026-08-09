"""Deterministic, source-agnostic waveform standardization for TRUST-ECG."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from trust_icu.ecg_data import EXPECTED_LEADS

_GAIN_RE = re.compile(
    r"^(?P<gain>[+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\((?P<baseline>[+-]?\d+)\))?/(?P<unit>[^\s]+)$"
)
_AUGMENTED_LEAD_CANONICAL = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}


@dataclass(frozen=True)
class SignalSpec:
    lead: str
    gain_per_unit: float
    baseline_digital: float
    physical_unit: str


@dataclass(frozen=True)
class StandardizedSignal:
    waveform_mv: np.ndarray
    valid_mask: np.ndarray
    source_sampling_rate_hz: float
    target_sampling_rate_hz: int
    source_sample_count: int
    target_sample_count: int
    crop_start_after_resampling: int | None
    left_padding: int
    right_padding: int


@dataclass(frozen=True)
class NormalizationStats:
    means_mv: tuple[float, ...]
    stds_mv: tuple[float, ...]
    valid_sample_counts: tuple[int, ...]
    leads: tuple[str, ...]
    fit_folds: tuple[int, ...]
    stats_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_payload(payload: dict[str, Any], key: str) -> str:
    material = dict(payload)
    material[key] = ""
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_lead_name(name: str) -> str:
    """Normalize only the documented augmented-lead capitalization difference."""

    return _AUGMENTED_LEAD_CANONICAL.get(str(name), str(name))


def _parse_gain(token: str, adc_zero: str) -> tuple[float, float, str]:
    match = _GAIN_RE.match(token)
    if not match:
        raise ValueError(f"Unsupported WFDB gain specification: {token!r}")
    gain = float(match.group("gain"))
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError(f"WFDB gain must be positive: {token!r}")
    baseline_text = match.group("baseline")
    baseline = float(baseline_text) if baseline_text is not None else float(adc_zero)
    unit = match.group("unit")
    if unit != "mV":
        raise ValueError(f"Primary TRUST-ECG signal path requires mV units, found {unit!r}")
    return gain, baseline, unit


def parse_signal_header(path: str | Path) -> tuple[str, float, int, tuple[SignalSpec, ...]]:
    """Parse physical scaling information from a Challenge or original PTB-XL WFDB header."""

    header_path = Path(path)
    lines = [
        line.strip()
        for line in header_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"Empty ECG header: {header_path}")
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f"Malformed WFDB first line: {header_path}")
    record_id = first[0]
    n_sig = int(first[1])
    sampling_rate = float(first[2].split("/", 1)[0])
    sample_count = int(first[3])
    if n_sig != len(EXPECTED_LEADS):
        raise ValueError(f"TRUST-ECG requires 12 leads, found {n_sig} in {header_path}")
    if len(lines) < 1 + n_sig:
        raise ValueError(f"Missing signal lines in {header_path}")

    specs: list[SignalSpec] = []
    for line in lines[1 : 1 + n_sig]:
        tokens = line.split()
        if len(tokens) < 9:
            raise ValueError(f"Malformed WFDB signal line in {header_path}: {line}")
        gain, baseline, unit = _parse_gain(tokens[2], tokens[4])
        specs.append(
            SignalSpec(
                lead=canonical_lead_name(tokens[-1]),
                gain_per_unit=gain,
                baseline_digital=baseline,
                physical_unit=unit,
            )
        )
    leads = tuple(spec.lead for spec in specs)
    if len(set(leads)) != len(EXPECTED_LEADS) or set(leads) != set(EXPECTED_LEADS):
        raise ValueError(f"Header does not contain each standard 12-lead name exactly once: {header_path}")
    if not math.isfinite(sampling_rate) or sampling_rate <= 0 or sample_count <= 0:
        raise ValueError(f"Invalid signal dimensions in {header_path}")
    return record_id, sampling_rate, sample_count, tuple(specs)


def load_mat_digital_signal(path: str | Path) -> np.ndarray:
    """Load a Challenge MATLAB-v4 `val` matrix as floating point digital samples."""

    mat_path = Path(path)
    payload = loadmat(mat_path)
    if "val" not in payload:
        raise ValueError(f"MAT file does not contain a 'val' matrix: {mat_path}")
    values = np.asarray(payload["val"], dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"ECG signal matrix must be two-dimensional: {mat_path}")
    if not np.isfinite(values).all():
        raise ValueError(f"ECG signal contains non-finite digital samples: {mat_path}")
    return values


def load_wfdb_physical_signal(record_stem: str | Path) -> tuple[np.ndarray, tuple[str, ...], float]:
    """Load one original PTB-XL WFDB record as physical mV using the official WFDB reader."""

    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover - exercised only without optional runtime
        raise RuntimeError(
            "Original PTB-XL waveform execution requires the optional 'ecg-waveform' dependency."
        ) from exc

    stem = Path(record_stem).expanduser().resolve()
    record = wfdb.rdrecord(str(stem), physical=True)
    signal = getattr(record, "p_signal", None)
    names = getattr(record, "sig_name", None)
    units = getattr(record, "units", None)
    fs = float(getattr(record, "fs", 0.0))
    if signal is None or names is None or units is None:
        raise ValueError(f"WFDB record is missing physical signal metadata: {stem}")
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(EXPECTED_LEADS):
        raise ValueError(f"Original PTB-XL WFDB record must contain 12 channels: {stem}")
    if not np.isfinite(values).all() or fs <= 0:
        raise ValueError(f"Original PTB-XL WFDB record contains invalid physical samples: {stem}")
    if any(str(unit) != "mV" for unit in units):
        raise ValueError(f"Original PTB-XL WFDB record must resolve to mV units: {stem}")
    leads = tuple(canonical_lead_name(str(name)) for name in names)
    return values.T, leads, fs


def digital_to_physical_mv(digital: np.ndarray, specs: tuple[SignalSpec, ...]) -> np.ndarray:
    """Convert digital samples to mV and reorder leads to the locked canonical order."""

    values = np.asarray(digital, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(specs):
        raise ValueError("Digital ECG shape does not match signal specifications.")
    if not np.isfinite(values).all():
        raise ValueError("Digital ECG contains non-finite values.")
    by_lead: dict[str, np.ndarray] = {}
    for index, spec in enumerate(specs):
        if spec.lead in by_lead:
            raise ValueError(f"Duplicate lead specification: {spec.lead}")
        if spec.physical_unit != "mV":
            raise ValueError(f"Unsupported primary physical unit: {spec.physical_unit}")
        by_lead[spec.lead] = (values[index] - spec.baseline_digital) / spec.gain_per_unit
    if set(by_lead) != set(EXPECTED_LEADS):
        raise ValueError("Signal specifications do not cover the standard 12 leads exactly once.")
    output = np.stack([by_lead[lead] for lead in EXPECTED_LEADS], axis=0)
    if not np.isfinite(output).all():
        raise ValueError("Physical ECG conversion produced non-finite values.")
    return output


def _resample_to_target(signal: np.ndarray, source_rate: float, target_rate: int) -> np.ndarray:
    if math.isclose(source_rate, target_rate, rel_tol=0.0, abs_tol=1e-9):
        return signal.copy()
    ratio = Fraction(target_rate / source_rate).limit_denominator(10000)
    return resample_poly(signal, up=ratio.numerator, down=ratio.denominator, axis=1)


def standardize_physical_signal(
    physical_mv: np.ndarray,
    lead_names: tuple[str, ...] | list[str],
    *,
    source_sampling_rate_hz: float,
    target_sampling_rate_hz: int = 500,
    target_duration_seconds: int = 10,
) -> StandardizedSignal:
    """Reorder a physical-mV ECG, resample, then deterministically crop/pad to 10 seconds."""

    values = np.asarray(physical_mv, dtype=np.float64)
    leads = tuple(canonical_lead_name(str(value)) for value in lead_names)
    if values.ndim != 2 or values.shape[0] != len(leads):
        raise ValueError("Physical ECG shape does not match supplied lead names.")
    if len(set(leads)) != len(EXPECTED_LEADS) or set(leads) != set(EXPECTED_LEADS):
        raise ValueError("Physical ECG must contain each standard 12-lead name exactly once.")
    if not np.isfinite(values).all():
        raise ValueError("Physical ECG contains non-finite values.")
    if not math.isfinite(source_sampling_rate_hz) or source_sampling_rate_hz <= 0:
        raise ValueError("Physical ECG sampling rate must be positive and finite.")
    by_lead = {lead: values[index] for index, lead in enumerate(leads)}
    ordered = np.stack([by_lead[lead] for lead in EXPECTED_LEADS], axis=0)
    source_sample_count = ordered.shape[1]
    resampled = _resample_to_target(ordered, source_sampling_rate_hz, target_sampling_rate_hz)
    target_samples = target_sampling_rate_hz * target_duration_seconds
    current = resampled.shape[1]
    crop_start: int | None = None
    left_padding = 0
    right_padding = 0

    if current > target_samples:
        crop_start = (current - target_samples) // 2
        standardized = resampled[:, crop_start : crop_start + target_samples]
        mask = np.ones(target_samples, dtype=bool)
    elif current < target_samples:
        missing = target_samples - current
        left_padding = missing // 2
        right_padding = missing - left_padding
        standardized = np.pad(
            resampled,
            ((0, 0), (left_padding, right_padding)),
            mode="constant",
            constant_values=0.0,
        )
        mask = np.zeros(target_samples, dtype=bool)
        mask[left_padding : left_padding + current] = True
    else:
        standardized = resampled
        mask = np.ones(target_samples, dtype=bool)

    if standardized.shape != (len(EXPECTED_LEADS), target_samples):
        raise RuntimeError("ECG standardization produced an unexpected shape.")
    if not np.isfinite(standardized).all():
        raise ValueError("Standardized ECG contains non-finite physical values.")
    return StandardizedSignal(
        waveform_mv=np.asarray(standardized, dtype=np.float32),
        valid_mask=mask,
        source_sampling_rate_hz=float(source_sampling_rate_hz),
        target_sampling_rate_hz=target_sampling_rate_hz,
        source_sample_count=source_sample_count,
        target_sample_count=target_samples,
        crop_start_after_resampling=crop_start,
        left_padding=left_padding,
        right_padding=right_padding,
    )


def standardize_signal(
    digital: np.ndarray,
    specs: tuple[SignalSpec, ...],
    *,
    source_sampling_rate_hz: float,
    target_sampling_rate_hz: int = 500,
    target_duration_seconds: int = 10,
) -> StandardizedSignal:
    """Convert a digital ECG to physical mV and apply the locked signal window contract."""

    physical = digital_to_physical_mv(digital, specs)
    return standardize_physical_signal(
        physical,
        tuple(spec.lead for spec in specs),
        source_sampling_rate_hz=source_sampling_rate_hz,
        target_sampling_rate_hz=target_sampling_rate_hz,
        target_duration_seconds=target_duration_seconds,
    )


class StreamingLeadStats:
    """Streaming per-lead mean/std estimator that excludes padded positions."""

    def __init__(self) -> None:
        self._count = np.zeros(len(EXPECTED_LEADS), dtype=np.int64)
        self._sum = np.zeros(len(EXPECTED_LEADS), dtype=np.float64)
        self._sum_sq = np.zeros(len(EXPECTED_LEADS), dtype=np.float64)

    def update(self, waveform_mv: np.ndarray, valid_mask: np.ndarray) -> None:
        waveform = np.asarray(waveform_mv, dtype=np.float64)
        mask = np.asarray(valid_mask, dtype=bool)
        if waveform.ndim != 2 or waveform.shape[0] != len(EXPECTED_LEADS):
            raise ValueError("Waveform must have shape (12, time).")
        if mask.ndim != 1 or mask.shape[0] != waveform.shape[1]:
            raise ValueError("Validity mask length must match waveform time dimension.")
        valid = waveform[:, mask]
        if valid.shape[1] == 0:
            raise ValueError("Normalization update contains no valid samples.")
        if not np.isfinite(valid).all():
            raise ValueError("Normalization update contains non-finite samples.")
        self._count += valid.shape[1]
        self._sum += valid.sum(axis=1)
        self._sum_sq += np.square(valid).sum(axis=1)

    def finalize(self, *, fit_folds: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)) -> NormalizationStats:
        if np.any(self._count <= 1):
            raise ValueError("Insufficient valid samples to estimate normalization statistics.")
        means = self._sum / self._count
        variance = self._sum_sq / self._count - np.square(means)
        variance = np.maximum(variance, 0.0)
        stds = np.sqrt(variance)
        if np.any(stds <= 1e-12) or not np.isfinite(means).all() or not np.isfinite(stds).all():
            raise ValueError("Degenerate or non-finite normalization statistics.")
        payload: dict[str, Any] = {
            "means_mv": [float(value) for value in means],
            "stds_mv": [float(value) for value in stds],
            "valid_sample_counts": [int(value) for value in self._count],
            "leads": list(EXPECTED_LEADS),
            "fit_folds": list(fit_folds),
            "stats_sha256": "",
        }
        payload["stats_sha256"] = _sha256_payload(payload, "stats_sha256")
        return NormalizationStats(
            means_mv=tuple(payload["means_mv"]),
            stds_mv=tuple(payload["stds_mv"]),
            valid_sample_counts=tuple(payload["valid_sample_counts"]),
            leads=tuple(EXPECTED_LEADS),
            fit_folds=fit_folds,
            stats_sha256=str(payload["stats_sha256"]),
        )


def normalize_signal(
    waveform_mv: np.ndarray,
    valid_mask: np.ndarray,
    stats: NormalizationStats,
) -> np.ndarray:
    """Apply training-only per-lead normalization and force padded positions back to zero."""

    waveform = np.asarray(waveform_mv, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if waveform.ndim != 2 or waveform.shape[0] != len(EXPECTED_LEADS):
        raise ValueError("Waveform must have shape (12, time).")
    if mask.ndim != 1 or mask.shape[0] != waveform.shape[1]:
        raise ValueError("Validity mask length must match waveform time dimension.")
    means = np.asarray(stats.means_mv, dtype=np.float64)[:, None]
    stds = np.asarray(stats.stds_mv, dtype=np.float64)[:, None]
    if means.shape[0] != waveform.shape[0] or np.any(stds <= 0):
        raise ValueError("Normalization statistics do not match the waveform contract.")
    normalized = (waveform - means) / stds
    normalized[:, ~mask] = 0.0
    if not np.isfinite(normalized).all():
        raise ValueError("Normalization produced non-finite values.")
    return np.asarray(normalized, dtype=np.float32)


def write_normalization_stats(stats: NormalizationStats, output: str | Path) -> None:
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = stats.to_dict()
    expected = _sha256_payload(payload, "stats_sha256")
    if expected != stats.stats_sha256:
        raise ValueError("Refusing to write normalization statistics with an invalid hash.")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
