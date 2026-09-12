"""Aggregate-only exact and near-duplicate screening for TRUST-ECG partitions."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np


def _validated_signal(waveform: np.ndarray) -> np.ndarray:
    array = np.asarray(waveform, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != 12 or array.shape[1] < 8:
        raise ValueError("Overlap screening requires a finite 12-lead ECG matrix.")
    if not np.isfinite(array).all():
        raise ValueError("Overlap screening requires finite ECG samples.")
    return array


def exact_signal_sha256(waveform: np.ndarray) -> str:
    """Hash signal shape and float64 values without persisting identifiers."""

    array = np.ascontiguousarray(_validated_signal(waveform), dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def compact_morphology_fingerprint(
    waveform: np.ndarray,
    *,
    bins_per_lead: int = 16,
) -> np.ndarray:
    """Return a low-dimensional, scale-normalized morphology fingerprint."""

    array = _validated_signal(waveform)
    if bins_per_lead < 4:
        raise ValueError("bins_per_lead must be at least 4.")
    centered = array - np.mean(array, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(scale, 1e-8)

    edges = np.linspace(0, normalized.shape[1], bins_per_lead + 1, dtype=np.int64)
    features: list[float] = []
    for lead in normalized:
        for start, stop in zip(edges[:-1], edges[1:], strict=True):
            segment = lead[start:stop]
            features.append(float(np.mean(segment)))
            features.append(float(np.std(segment)))
    fingerprint = np.asarray(features, dtype=np.float64)
    norm = float(np.linalg.norm(fingerprint))
    if norm > 0.0:
        fingerprint /= norm
    return fingerprint


def audit_cross_partition_overlap(
    certification_signals: Iterable[np.ndarray],
    recovery_signals: Iterable[np.ndarray],
    *,
    near_duplicate_threshold: float = 0.995,
    batch_size: int = 128,
) -> dict[str, object]:
    """Screen two partitions and return only aggregate overlap evidence."""

    if not 0.0 < near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must lie in (0, 1].")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    certification = [_validated_signal(item) for item in certification_signals]
    recovery = [_validated_signal(item) for item in recovery_signals]
    if not certification or not recovery:
        raise ValueError("Both partitions must contain at least one ECG.")

    certification_hashes = {exact_signal_sha256(item) for item in certification}
    recovery_hashes = [exact_signal_sha256(item) for item in recovery]
    exact_count = sum(item in certification_hashes for item in recovery_hashes)

    cert_fp = np.vstack([compact_morphology_fingerprint(item) for item in certification])
    rec_fp = np.vstack([compact_morphology_fingerprint(item) for item in recovery])
    maximum_by_recovery: list[float] = []
    for start in range(0, len(recovery), batch_size):
        similarity = rec_fp[start : start + batch_size] @ cert_fp.T
        maximum_by_recovery.extend(np.max(similarity, axis=1).tolist())
    maxima = np.asarray(maximum_by_recovery, dtype=np.float64)
    near_count = int(np.sum(maxima >= float(near_duplicate_threshold)))

    return {
        "certification_records": len(certification),
        "recovery_records": len(recovery),
        "exact_cross_partition_duplicates": int(exact_count),
        "near_duplicate_threshold": float(near_duplicate_threshold),
        "near_duplicate_candidates": near_count,
        "maximum_near_duplicate_similarity": float(np.max(maxima)),
        "q95_near_duplicate_similarity": float(np.quantile(maxima, 0.95)),
        "q99_near_duplicate_similarity": float(np.quantile(maxima, 0.99)),
        "patient_independence_claimed": False,
    }
