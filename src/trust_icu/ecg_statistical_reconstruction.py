"""Audited cross-host reconstruction checks for TRUST-ECG addendum inference."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit

from trust_icu.ecg_baseline import evaluate_binary_probabilities
from trust_icu.ecg_statistical_core import write_csv

LABEL_NAMES = {
    "59118001": "RBBB",
    "164889003": "AF",
    "164909002": "LBBB",
    "270492004": "IAVB",
    "284470004": "PAC",
    "426783006": "NSR",
    "427084000": "STach",
}
EXTERNAL_SOURCES = ("georgia", "cpsc_2018", "cpsc_2018_extra")

# The primary report was created from float32 PyTorch inference before checkpoint
# serialization. The addendum replays the exact checkpoint on a later hosted CPU.
# These thresholds are intentionally far below normal reporting precision while
# allowing harmless cross-host floating-point/kernel drift. Integer support counts
# remain exact and therefore use zero tolerance.
RECONSTRUCTION_TOLERANCES: dict[str, tuple[float, float]] = {
    "n": (0.0, 0.0),
    "positives": (0.0, 0.0),
    "negatives": (0.0, 0.0),
    "prevalence": (1e-15, 0.0),
    "pr_auc": (5e-4, 0.0),
    "pr_auc_to_prevalence_ratio": (5e-3, 0.0),
    "roc_auc": (5e-4, 0.0),
    "brier": (5e-4, 0.0),
    "brier_skill_vs_prevalence": (5e-3, 0.0),
    "calibration_slope": (5e-3, 0.0),
    "calibration_intercept": (5e-3, 0.0),
}


def apply_phase0_calibration_payload(
    raw_logits: np.ndarray,
    calibration_payload: dict[str, Any],
    label_codes: tuple[str, ...],
) -> np.ndarray:
    """Replay Phase-0 Platt scaling with scipy.expit, matching sklearn predict_proba."""

    logits = np.asarray(raw_logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != len(label_codes):
        raise ValueError("Raw Phase-0 logits do not align with the locked label set.")
    columns: list[np.ndarray] = []
    for index, code in enumerate(label_codes):
        params = calibration_payload["parameters"][code]
        coefficient = float(params["coefficient"])
        intercept = float(params["intercept"])
        columns.append(expit(coefficient * logits[:, index] + intercept))
    return np.column_stack(columns)


def _metric_match(metric: str, observed: float, expected: float) -> tuple[bool, float, float]:
    if metric not in RECONSTRUCTION_TOLERANCES:
        raise KeyError(f"No reconstruction tolerance declared for metric: {metric}")
    atol, rtol = RECONSTRUCTION_TOLERANCES[metric]
    absolute_delta = abs(float(observed) - float(expected))
    allowed_delta = float(atol + rtol * abs(float(expected)))
    return absolute_delta <= allowed_delta, absolute_delta, allowed_delta


def _audit_row(
    *,
    scope: str,
    source: str,
    code: str,
    metric: str,
    observed: float,
    expected: float,
) -> dict[str, Any]:
    within, absolute_delta, allowed_delta = _metric_match(metric, observed, expected)
    denominator = max(abs(float(expected)), np.finfo(np.float64).tiny)
    return {
        "scope": scope,
        "source": source,
        "label_code": code,
        "label_name": LABEL_NAMES[code],
        "metric": metric,
        "expected": float(expected),
        "observed": float(observed),
        "absolute_delta": float(absolute_delta),
        "relative_delta": float(absolute_delta / denominator),
        "allowed_absolute_delta": float(allowed_delta),
        "within_tolerance": bool(within),
        "tolerance_basis": "same_hashed_model_cross_host_float32_cpu_reinference",
    }


def validate_resnet_report_metrics_audited(
    *,
    report: dict[str, Any],
    internal_targets: np.ndarray,
    internal_probabilities: np.ndarray,
    external_rows: list[Any],
    external_targets: np.ndarray,
    external_probabilities: np.ndarray,
    label_codes: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Audit report reconstruction, persist exact deltas, and fail on material drift."""

    audit_rows: list[dict[str, Any]] = []
    for index, code in enumerate(label_codes):
        observed = asdict(
            evaluate_binary_probabilities(
                internal_targets[:, index],
                internal_probabilities[:, index],
            )
        )
        expected = report["internal_test"]["per_label"][code]
        for metric, value in observed.items():
            audit_rows.append(
                _audit_row(
                    scope="internal_fold10",
                    source="ptb-xl",
                    code=code,
                    metric=metric,
                    observed=float(value),
                    expected=float(expected[metric]),
                )
            )

    for source in EXTERNAL_SOURCES:
        indices = [index for index, row in enumerate(external_rows) if row.source == source]
        for label_index, code in enumerate(label_codes):
            expected_pair = report["external_certification"][source][code]
            if expected_pair["metrics"] is None:
                continue
            observed = asdict(
                evaluate_binary_probabilities(
                    external_targets[indices, label_index],
                    external_probabilities[indices, label_index],
                )
            )
            for metric, value in observed.items():
                audit_rows.append(
                    _audit_row(
                        scope="external_certification",
                        source=source,
                        code=code,
                        metric=metric,
                        observed=float(value),
                        expected=float(expected_pair["metrics"][metric]),
                    )
                )

    output_root = Path(os.environ.get("ADDENDUM_ROOT", ".")).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    audit_path = output_root / "resnet_metric_reconstruction_audit.csv"
    write_csv(audit_path, audit_rows)

    failures = [row for row in audit_rows if not bool(row["within_tolerance"])]
    maxima: dict[str, float] = {}
    for row in audit_rows:
        metric = str(row["metric"])
        maxima[metric] = max(maxima.get(metric, 0.0), float(row["absolute_delta"]))
    print(
        json.dumps(
            {
                "resnet_metric_reconstruction": {
                    "rows": len(audit_rows),
                    "failures": len(failures),
                    "audit_path": str(audit_path),
                    "max_absolute_delta_by_metric": dict(sorted(maxima.items())),
                }
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    if failures:
        details = "; ".join(
            (
                f"{row['scope']}/{row['source']}/{row['label_code']}/{row['metric']}: "
                f"expected={row['expected']:.17g}, observed={row['observed']:.17g}, "
                f"abs_delta={row['absolute_delta']:.6g}, "
                f"allowed={row['allowed_absolute_delta']:.6g}"
            )
            for row in failures[:8]
        )
        raise RuntimeError(
            "ResNet metric reconstruction exceeded the declared numerical tolerance. "
            + details
        )
    return audit_rows
