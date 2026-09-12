"""Aggregate-only secondary statistics for TRUST-ECG v0.4."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

PRIMARY_SEED = 20260808
REPLICATION_SEEDS = (20260809, 20260810)


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    successes = int(successes)
    total = int(total)
    if total <= 0:
        raise ValueError("Wilson interval requires total > 0.")
    if successes < 0 or successes > total:
        raise ValueError("successes must lie in [0, total].")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive.")

    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def phase1_estimability_rows(phase1_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten aggregate Phase-1 estimability while preserving missing repeats."""

    plan = phase1_report.get("phase1_plan")
    pairs = phase1_report.get("pair_results")
    if not isinstance(plan, Mapping) or not isinstance(pairs, Mapping):
        raise ValueError("Phase-1 report lacks phase1_plan or pair_results.")
    default_repeats = int(plan.get("repeats", 0))
    if default_repeats <= 0:
        raise ValueError("Phase-1 repeat count must be positive.")

    rows: list[dict[str, Any]] = []
    for pair_key in sorted(pairs):
        pair = pairs[pair_key]
        if not isinstance(pair, Mapping):
            raise ValueError(f"Invalid pair result: {pair_key}")
        source = str(pair["source"])
        label_code = str(pair["label_code"])
        budgets = pair.get("budgets")
        if not isinstance(budgets, Mapping):
            raise ValueError(f"Pair {pair_key} lacks budget results.")
        for budget_key in sorted(budgets, key=lambda value: int(value)):
            budget_payload = budgets[budget_key]
            if not isinstance(budget_payload, Mapping):
                raise ValueError("Budget payload must be an object.")
            methods = budget_payload.get("methods")
            if not isinstance(methods, Mapping):
                raise ValueError("Budget payload lacks method summaries.")
            for method_name in sorted(methods):
                method = methods[method_name]
                if not isinstance(method, Mapping):
                    raise ValueError("Method payload must be an object.")
                requested = int(method.get("repeats_requested", default_repeats))
                estimable = int(method.get("estimable_repeats", 0))
                nonestimable = int(
                    method.get("nonestimable_repeats", requested - estimable)
                )
                if estimable < 0 or nonestimable < 0 or estimable + nonestimable != requested:
                    raise ValueError("Phase-1 estimability counts are internally inconsistent.")
                success_count = int(method.get("recovery_envelope_met_count", 0))
                rate = method.get("recovery_envelope_met_rate_among_estimable")
                if estimable == 0:
                    low = None
                    high = None
                    normalized_rate = None
                else:
                    expected_rate = success_count / estimable
                    normalized_rate = expected_rate if rate is None else float(rate)
                    if not np.isclose(normalized_rate, expected_rate, atol=1e-12, rtol=0.0):
                        raise ValueError("Phase-1 recovery count and rate disagree.")
                    low, high = wilson_interval(success_count, estimable)

                reasons = method.get("nonestimable_reasons", {})
                if not isinstance(reasons, Mapping):
                    raise ValueError("nonestimable_reasons must be an aggregate mapping.")
                reason_counts = {str(key): int(value) for key, value in reasons.items()}
                if reason_counts and sum(reason_counts.values()) != nonestimable:
                    raise ValueError("Non-estimability reason counts do not match total.")

                rows.append(
                    {
                        "source": source,
                        "label_code": label_code,
                        "budget": int(budget_key),
                        "method": str(method_name),
                        "repeats_requested": requested,
                        "estimable_repeats": estimable,
                        "nonestimable_repeats": nonestimable,
                        "nonestimable_reasons": reason_counts,
                        "recovery_success_count": success_count,
                        "recovery_success_rate": normalized_rate,
                        "recovery_rate_wilson_low": low,
                        "recovery_rate_wilson_high": high,
                    }
                )
    return rows


def _flatten_statuses(report: Mapping[str, Any]) -> dict[str, str]:
    external = report.get("external_certification")
    if not isinstance(external, Mapping):
        raise ValueError("Seed report lacks external_certification.")
    flattened: dict[str, str] = {}
    for source in sorted(external):
        pairs = external[source]
        if not isinstance(pairs, Mapping):
            raise ValueError("Seed external source payload must be an object.")
        for code in sorted(pairs):
            payload = pairs[code]
            if not isinstance(payload, Mapping) or "status" not in payload:
                raise ValueError("Seed pair payload lacks status.")
            flattened[f"{source}/{code}"] = str(payload["status"])
    return flattened


def _internal_metric_summary(reports: Mapping[int, Mapping[str, Any]], metric: str) -> dict[str, float]:
    values: list[float] = []
    for seed in (PRIMARY_SEED, *REPLICATION_SEEDS):
        internal = reports[seed].get("internal_test")
        if not isinstance(internal, Mapping):
            raise ValueError("Seed report lacks internal_test metrics.")
        values.append(float(internal[metric]))
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Seed-stability metrics must be finite.")
    return {
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def aggregate_seed_stability(
    reports: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize exactly three seeds while preserving 20260808 as primary."""

    required = {PRIMARY_SEED, *REPLICATION_SEEDS}
    observed = {int(seed) for seed in reports}
    if observed != required:
        raise ValueError(
            "Seed stability requires exactly 20260808, 20260809, and 20260810."
        )
    normalized = {int(seed): report for seed, report in reports.items()}
    primary_statuses = _flatten_statuses(normalized[PRIMARY_SEED])
    if not primary_statuses:
        raise ValueError("Primary seed report contains no external pair statuses.")

    agreement: dict[str, float] = {}
    for seed in REPLICATION_SEEDS:
        statuses = _flatten_statuses(normalized[seed])
        if set(statuses) != set(primary_statuses):
            raise ValueError("Seed reports do not contain the same external pairs.")
        matched = sum(statuses[key] == value for key, value in primary_statuses.items())
        agreement[str(seed)] = matched / len(primary_statuses)

    return {
        "primary_seed": PRIMARY_SEED,
        "replication_seeds": list(REPLICATION_SEEDS),
        "primary_seed_replaced": False,
        "pair_status_agreement_with_primary": agreement,
        "internal_macro_pr_auc": _internal_metric_summary(
            normalized, "macro_pr_auc"
        ),
        "internal_macro_roc_auc": _internal_metric_summary(
            normalized, "macro_roc_auc"
        ),
        "internal_macro_brier": _internal_metric_summary(normalized, "macro_brier"),
    }
