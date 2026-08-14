"""Aggregate-only statistical helpers for the TRUST-ECG publication addendum."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

METRICS = ("pr_auc", "roc_auc", "brier")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_hash(payload: dict[str, Any], key: str) -> str:
    material = dict(payload)
    material[key] = ""
    encoded = json.dumps(
        json_ready(material),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, "mean": None, "median": None, "q025": None, "q975": None}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q025": float(np.quantile(array, 0.025)),
        "q975": float(np.quantile(array, 0.975)),
    }


def bootstrap_two_sided_p(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    non_positive = (np.sum(array <= 0.0) + 1.0) / (array.size + 1.0)
    non_negative = (np.sum(array >= 0.0) + 1.0) / (array.size + 1.0)
    return float(min(1.0, 2.0 * min(non_positive, non_negative)))


def benjamini_hochberg(p_values: Sequence[float | None]) -> list[float | None]:
    """Return BH-adjusted q-values while preserving missing entries."""

    valid = [(index, float(value)) for index, value in enumerate(p_values) if value is not None]
    output: list[float | None] = [None] * len(p_values)
    if not valid:
        return output
    ordered = sorted(valid, key=lambda item: item[1])
    m = len(ordered)
    running = 1.0
    adjusted: dict[int, float] = {}
    for rank_from_end, (index, value) in enumerate(reversed(ordered), start=1):
        rank = m - rank_from_end + 1
        running = min(running, value * m / rank)
        adjusted[index] = min(1.0, running)
    for index, _ in valid:
        output[index] = adjusted[index]
    return output


def stratified_binary_bootstrap_indices(
    y: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    targets = np.asarray(y, dtype=np.int64)
    positives = np.flatnonzero(targets == 1)
    negatives = np.flatnonzero(targets == 0)
    if positives.size == 0 or negatives.size == 0:
        raise ValueError("Stratified binary bootstrap requires both classes.")
    sampled = np.concatenate(
        [
            rng.choice(positives, size=positives.size, replace=True),
            rng.choice(negatives, size=negatives.size, replace=True),
        ]
    ).astype(np.int64, copy=False)
    rng.shuffle(sampled)
    return sampled


def binary_fast_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    targets = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if np.unique(targets).size != 2:
        raise ValueError("Binary metrics require both classes.")
    return {
        "pr_auc": float(average_precision_score(targets, probs)),
        "roc_auc": float(roc_auc_score(targets, probs)),
        "brier": float(brier_score_loss(targets, probs)),
    }


def macro_fast_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    targets = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    rows = [
        binary_fast_metrics(targets[:, index], probs[:, index])
        for index in range(targets.shape[1])
    ]
    return {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}


def paired_binary_bootstrap(
    y: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    targets = np.asarray(y, dtype=np.int64)
    candidate_probs = np.asarray(candidate, dtype=np.float64)
    reference_probs = np.asarray(reference, dtype=np.float64)
    point_candidate = binary_fast_metrics(targets, candidate_probs)
    point_reference = binary_fast_metrics(targets, reference_probs)
    deltas = {metric: [] for metric in METRICS}
    candidate_values = {metric: [] for metric in METRICS}
    reference_values = {metric: [] for metric in METRICS}
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        indices = stratified_binary_bootstrap_indices(targets, rng)
        candidate_metrics = binary_fast_metrics(targets[indices], candidate_probs[indices])
        reference_metrics = binary_fast_metrics(targets[indices], reference_probs[indices])
        for metric in METRICS:
            candidate_values[metric].append(candidate_metrics[metric])
            reference_values[metric].append(reference_metrics[metric])
            if metric == "brier":
                deltas[metric].append(reference_metrics[metric] - candidate_metrics[metric])
            else:
                deltas[metric].append(candidate_metrics[metric] - reference_metrics[metric])
    return {
        "point_candidate": point_candidate,
        "point_reference": point_reference,
        "candidate_intervals": {
            metric: quantile_summary(candidate_values[metric]) for metric in METRICS
        },
        "reference_intervals": {
            metric: quantile_summary(reference_values[metric]) for metric in METRICS
        },
        "paired_improvement": {
            metric: {
                **quantile_summary(deltas[metric]),
                "two_sided_bootstrap_p": bootstrap_two_sided_p(deltas[metric]),
                "direction": (
                    "positive_means_lower_candidate_brier"
                    if metric == "brier"
                    else "positive_means_higher_candidate_metric"
                ),
            }
            for metric in METRICS
        },
    }


def paired_macro_bootstrap(
    y: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    targets = np.asarray(y, dtype=np.int64)
    candidate_probs = np.asarray(candidate, dtype=np.float64)
    reference_probs = np.asarray(reference, dtype=np.float64)
    point_candidate = macro_fast_metrics(targets, candidate_probs)
    point_reference = macro_fast_metrics(targets, reference_probs)
    deltas = {metric: [] for metric in METRICS}
    candidate_values = {metric: [] for metric in METRICS}
    reference_values = {metric: [] for metric in METRICS}
    rng = np.random.default_rng(seed)
    accepted = 0
    attempts = 0
    while accepted < repeats and attempts < repeats * 20:
        attempts += 1
        indices = rng.integers(0, targets.shape[0], size=targets.shape[0], endpoint=False)
        if any(np.unique(targets[indices, column]).size != 2 for column in range(targets.shape[1])):
            continue
        candidate_metrics = macro_fast_metrics(targets[indices], candidate_probs[indices])
        reference_metrics = macro_fast_metrics(targets[indices], reference_probs[indices])
        for metric in METRICS:
            candidate_values[metric].append(candidate_metrics[metric])
            reference_values[metric].append(reference_metrics[metric])
            if metric == "brier":
                deltas[metric].append(reference_metrics[metric] - candidate_metrics[metric])
            else:
                deltas[metric].append(candidate_metrics[metric] - reference_metrics[metric])
        accepted += 1
    if accepted != repeats:
        raise RuntimeError("Unable to obtain the requested valid macro bootstrap replicates.")
    return {
        "point_candidate": point_candidate,
        "point_reference": point_reference,
        "candidate_intervals": {
            metric: quantile_summary(candidate_values[metric]) for metric in METRICS
        },
        "reference_intervals": {
            metric: quantile_summary(reference_values[metric]) for metric in METRICS
        },
        "paired_improvement": {
            metric: {
                **quantile_summary(deltas[metric]),
                "two_sided_bootstrap_p": bootstrap_two_sided_p(deltas[metric]),
                "direction": (
                    "positive_means_lower_candidate_brier"
                    if metric == "brier"
                    else "positive_means_higher_candidate_metric"
                ),
            }
            for metric in METRICS
        },
        "bootstrap_attempts": attempts,
    }


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def equal_frequency_calibration_bins(
    y: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int,
) -> list[dict[str, float | int]]:
    targets = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if targets.shape != probs.shape or targets.ndim != 1:
        raise ValueError("Calibration-bin inputs must be aligned one-dimensional arrays.")
    if bins < 2:
        raise ValueError("At least two calibration bins are required.")
    order = np.argsort(probs, kind="mergesort")
    output: list[dict[str, float | int]] = []
    for index, group in enumerate(np.array_split(order, min(bins, order.size)), start=1):
        if group.size == 0:
            continue
        positives = int(targets[group].sum())
        low, high = wilson_interval(positives, int(group.size))
        output.append(
            {
                "bin": index,
                "n": int(group.size),
                "mean_predicted_probability": float(np.mean(probs[group])),
                "observed_prevalence": float(np.mean(targets[group])),
                "observed_wilson_low": float(low),
                "observed_wilson_high": float(high),
                "minimum_probability": float(np.min(probs[group])),
                "maximum_probability": float(np.max(probs[group])),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty aggregate table: {path.name}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(output_root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name != "aggregate_sha256_manifest.json":
            manifest[path.name] = sha256_file(path)
    (output_root / "aggregate_sha256_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
