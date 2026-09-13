"""Bootstrap gate uncertainty and conservative subgroup metadata helpers for TRUST-ECG."""

from __future__ import annotations

import math
import re
from dataclasses import asdict
from typing import Any

import numpy as np

from trust_icu.ecg_baseline import evaluate_binary_probabilities

_AGE_RE = re.compile(r"^#\s*age\s*:\s*(.+?)\s*$", flags=re.IGNORECASE)
_SEX_RE = re.compile(r"^#\s*sex\s*:\s*(.+?)\s*$", flags=re.IGNORECASE)


def parse_header_demographics(header_text: str) -> dict[str, float | str | None]:
    """Parse only conservative age/sex fields from an ECG header comment block."""

    age: float | None = None
    sex: str | None = None
    for raw_line in str(header_text).splitlines():
        line = raw_line.strip()
        age_match = _AGE_RE.match(line)
        if age_match:
            raw_age = age_match.group(1).strip()
            try:
                candidate = float(raw_age)
            except ValueError:
                candidate = math.nan
            if math.isfinite(candidate) and 0.0 <= candidate <= 120.0:
                age = float(candidate)
            continue

        sex_match = _SEX_RE.match(line)
        if sex_match:
            normalized = sex_match.group(1).strip().lower()
            if normalized in {"m", "male"}:
                sex = "male"
            elif normalized in {"f", "female"}:
                sex = "female"

    age_band: str | None = None
    if age is not None:
        if age < 40.0:
            age_band = "under_40"
        elif age < 65.0:
            age_band = "40_to_64"
        else:
            age_band = "65_plus"
    return {"age": age, "sex": sex, "age_band": age_band}


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def bootstrap_gate_uncertainty(
    y: np.ndarray,
    probabilities: np.ndarray,
    *,
    repeats: int = 1000,
    seed: int = 20260808,
    minimum_positives: int = 50,
    minimum_negatives: int = 50,
    minimum_pr_auc_to_prevalence_ratio: float = 2.0,
    maximum_absolute_slope_deviation: float = 0.35,
    maximum_absolute_intercept: float = 0.75,
) -> dict[str, Any]:
    """Estimate aggregate uncertainty around the frozen Phase-0 gate.

    The point-estimate certification remains primary. Bootstrap resamples are
    secondary evidence only and are never used to redefine the official status.
    """

    targets = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 1 or probs.shape != targets.shape or targets.size == 0:
        raise ValueError("Targets and probabilities must be aligned one-dimensional arrays.")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("Targets must be binary.")
    if not np.isfinite(probs).all() or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("Probabilities must be finite and lie in [0, 1].")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    positives = int(targets.sum())
    negatives = int(targets.size - positives)
    base: dict[str, Any] = {
        "status": "estimable",
        "n": int(targets.size),
        "positives": positives,
        "negatives": negatives,
        "repeats_requested": int(repeats),
        "estimable_repeats": 0,
        "nonestimable_repeats": 0,
        "support_eligible_repeats": 0,
        "complete_gate_satisfaction_count": 0,
        "complete_gate_satisfaction_rate": None,
        "pr_auc_to_prevalence_ratio_q025": None,
        "pr_auc_to_prevalence_ratio_q50": None,
        "pr_auc_to_prevalence_ratio_q975": None,
        "calibration_slope_q025": None,
        "calibration_slope_q50": None,
        "calibration_slope_q975": None,
        "calibration_intercept_q025": None,
        "calibration_intercept_q50": None,
        "calibration_intercept_q975": None,
        "brier_skill_vs_prevalence_q025": None,
        "brier_skill_vs_prevalence_q50": None,
        "brier_skill_vs_prevalence_q975": None,
        "secondary_only": True,
        "point_estimate_status_replaced": False,
    }
    if positives < minimum_positives or negatives < minimum_negatives:
        base["status"] = "insufficient_support"
        return base

    rng = np.random.default_rng(int(seed))
    ratios: list[float] = []
    slopes: list[float] = []
    intercepts: list[float] = []
    brier_skills: list[float] = []
    gate_successes = 0
    support_eligible = 0
    estimable = 0

    for _ in range(int(repeats)):
        indices = rng.integers(0, targets.size, size=targets.size)
        sampled_y = targets[indices]
        sampled_p = probs[indices]
        sampled_pos = int(sampled_y.sum())
        sampled_neg = int(sampled_y.size - sampled_pos)
        if sampled_pos == 0 or sampled_neg == 0:
            continue
        metrics = asdict(evaluate_binary_probabilities(sampled_y, sampled_p))
        estimable += 1
        ratio = float(metrics["pr_auc_to_prevalence_ratio"])
        slope = float(metrics["calibration_slope"])
        intercept = float(metrics["calibration_intercept"])
        brier_skill = float(metrics["brier_skill_vs_prevalence"])
        ratios.append(ratio)
        slopes.append(slope)
        intercepts.append(intercept)
        brier_skills.append(brier_skill)

        has_support = sampled_pos >= minimum_positives and sampled_neg >= minimum_negatives
        if has_support:
            support_eligible += 1
        if (
            has_support
            and ratio >= minimum_pr_auc_to_prevalence_ratio
            and abs(slope - 1.0) <= maximum_absolute_slope_deviation
            and abs(intercept) <= maximum_absolute_intercept
            and brier_skill > 0.0
        ):
            gate_successes += 1

    base["estimable_repeats"] = estimable
    base["nonestimable_repeats"] = int(repeats) - estimable
    base["support_eligible_repeats"] = support_eligible
    base["complete_gate_satisfaction_count"] = gate_successes
    base["complete_gate_satisfaction_rate"] = (
        None if estimable == 0 else float(gate_successes / estimable)
    )
    for name, values in (
        ("pr_auc_to_prevalence_ratio", ratios),
        ("calibration_slope", slopes),
        ("calibration_intercept", intercepts),
        ("brier_skill_vs_prevalence", brier_skills),
    ):
        base[f"{name}_q025"] = _quantile(values, 0.025)
        base[f"{name}_q50"] = _quantile(values, 0.5)
        base[f"{name}_q975"] = _quantile(values, 0.975)
    return base
