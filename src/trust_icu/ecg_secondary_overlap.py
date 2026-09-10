"""Aggregate-only exact and near-duplicate screening for TRUST-ECG.

The public API intentionally accepts waveform arrays without record identifiers and returns only
aggregate counts/similarities. It is secondary audit code and does not alter the frozen external
certification/recovery partition.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


def _validated_waveform(waveform: np.ndarray) -> np.ndarray:
    array = np.asarray(waveform, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != 12 or array.shape[1] <= 0:
        raise ValueError("ECG waveform must have shape (12, time) with a non-empty time axis.")
    if not np.isfinite(array).all():
        raise ValueError("ECG waveform fingerprinting requires finite samples.")
    return array


def _waveform_digest(waveform: np.ndarray) -> str:
    array = _validated_waveform(waveform)
    canonical = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def waveform_fingerprint(waveform: np.ndarray, *, bins_per_lead: int = 8) -> np.ndarray:
    """Create a deterministic compact morphology fingerprint for near-duplicate screening.

    Each lead is centered and scale-normalized independently, then represented by equal-width
    temporal-bin means. The final vector is L2-normalized. This intentionally removes simple
    offset/gain differences while preserving coarse waveform shape. It is an audit fingerprint,
    not a learned representation and not a patient-identity claim.
    """

    array = _validated_waveform(waveform)
    if not isinstance(bins_per_lead, int) or not 1 <= bins_per_lead <= 32:
        raise ValueError("bins_per_lead must be an integer in [1, 32].")

    pieces: list[np.ndarray] = []
    time_points = array.shape[1]
    edges = np.linspace(0, time_points, bins_per_lead + 1, dtype=np.int64)
    for lead in array:
        centered = lead - float(np.median(lead))
        scale = float(np.sqrt(np.mean(centered * centered)))
        if scale > 0.0:
            centered = centered / scale
        else:
            centered = np.zeros_like(centered)

        binned = np.zeros(bins_per_lead, dtype=np.float64)
        for index in range(bins_per_lead):
            start = int(edges[index])
            stop = int(edges[index + 1])
            if stop <= start:
                sample_index = min(start, time_points - 1)
                binned[index] = centered[sample_index]
            else:
                binned[index] = float(np.mean(centered[start:stop]))
        pieces.append(binned)

    fingerprint = np.concatenate(pieces)
    norm = float(np.linalg.norm(fingerprint))
    if norm > 0.0:
        fingerprint = fingerprint / norm
    return fingerprint


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity clipped to [0, 1] for same-shape finite vectors."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("Cosine similarity requires aligned one-dimensional vectors.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Cosine similarity requires finite vectors.")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    value = float(np.dot(a, b) / denominator)
    return float(np.clip(value, 0.0, 1.0))


def _fingerprint_matrix(waveforms: Sequence[np.ndarray]) -> np.ndarray:
    if not waveforms:
        return np.empty((0, 96), dtype=np.float64)
    matrix = np.vstack([waveform_fingerprint(waveform) for waveform in waveforms])
    if matrix.shape[1] != 96:
        raise RuntimeError("Unexpected TRUST-ECG overlap fingerprint width.")
    return matrix


def summarize_cross_partition_overlap(
    certification_waveforms: Sequence[np.ndarray],
    recovery_waveforms: Sequence[np.ndarray],
    *,
    similarity_threshold: float = 0.995,
    chunk_size: int = 256,
) -> dict[str, int | float | str]:
    """Return aggregate cross-partition exact/near-duplicate screening results.

    Near-duplicate counts include exact duplicate pairs because exact copies are also maximally
    similar. Similarity is evaluated in bounded chunks, avoiding construction of one full dense
    certification-by-recovery matrix. No record identifiers or fingerprints are returned.
    """

    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must lie in [0, 1].")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    certification = list(certification_waveforms)
    recovery = list(recovery_waveforms)
    for waveform in (*certification, *recovery):
        _validated_waveform(waveform)

    cert_hashes: dict[str, int] = {}
    recovery_hashes: dict[str, int] = {}
    for waveform in certification:
        digest = _waveform_digest(waveform)
        cert_hashes[digest] = cert_hashes.get(digest, 0) + 1
    for waveform in recovery:
        digest = _waveform_digest(waveform)
        recovery_hashes[digest] = recovery_hashes.get(digest, 0) + 1
    exact_pairs = sum(
        cert_count * recovery_hashes.get(digest, 0)
        for digest, cert_count in cert_hashes.items()
    )

    cert_fingerprints = _fingerprint_matrix(certification)
    recovery_fingerprints = _fingerprint_matrix(recovery)
    maximum_similarity = 0.0
    near_pairs = 0

    if cert_fingerprints.size and recovery_fingerprints.size:
        for cert_start in range(0, cert_fingerprints.shape[0], chunk_size):
            cert_chunk = cert_fingerprints[cert_start : cert_start + chunk_size]
            for recovery_start in range(0, recovery_fingerprints.shape[0], chunk_size):
                recovery_chunk = recovery_fingerprints[
                    recovery_start : recovery_start + chunk_size
                ]
                similarities = cert_chunk @ recovery_chunk.T
                np.clip(similarities, 0.0, 1.0, out=similarities)
                maximum_similarity = max(maximum_similarity, float(np.max(similarities)))
                near_pairs += int(np.count_nonzero(similarities >= similarity_threshold))

    overlap_detected = exact_pairs > 0 or near_pairs > 0
    return {
        "certification_waveforms": len(certification),
        "recovery_waveforms": len(recovery),
        "exact_duplicate_pairs": int(exact_pairs),
        "near_duplicate_pairs": int(near_pairs),
        "maximum_cross_partition_similarity": float(maximum_similarity),
        "similarity_threshold": float(similarity_threshold),
        "audit_status": (
            "cross_partition_overlap_detected"
            if overlap_detected
            else "no_overlap_detected_at_threshold"
        ),
    }
