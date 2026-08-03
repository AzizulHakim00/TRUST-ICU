"""Run a privacy-safe synthetic Phase 0 pipeline from cohort construction to metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from trust_icu.baseline import fit_catboost_baseline, fit_logistic_baseline
from trust_icu.cohort import assign_task_labels, build_landmark_cohort
from trust_icu.features import build_feature_matrix


def generate_synthetic_inputs(n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    admit = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(np.arange(n), unit="h")
    age = np.clip(rng.normal(64, 15, n), 18, 95)
    sex = rng.choice(["female", "male"], n)
    stays = pd.DataFrame(
        {
            "patient_id": np.arange(n),
            "hospital_admission_id": np.arange(n),
            "stay_id": np.arange(n),
            "age": age,
            "sex": sex,
            "icu_admit_time": admit,
            "icu_discharge_time": admit + pd.to_timedelta(rng.uniform(20, 72, n), unit="h"),
            "death_time": pd.NaT,
        }
    )

    observation_rows: list[dict[str, object]] = []
    last_lactate = np.empty(n)
    last_heart_rate = np.empty(n)
    for stay_id in range(n):
        lactate_base = max(0.4, rng.lognormal(0.25, 0.5))
        heart_base = np.clip(rng.normal(88, 18), 35, 180)
        last_lactate[stay_id] = lactate_base
        last_heart_rate[stay_id] = heart_base
        for hour in (1, 3, 5):
            if rng.random() > 0.18:
                observation_rows.append(
                    {
                        "stay_id": stay_id,
                        "variable": "heart_rate",
                        "event_time": admit[stay_id] + pd.Timedelta(hours=hour),
                        "value": heart_base + rng.normal(0, 5),
                    }
                )
            if rng.random() > 0.40:
                value = max(0.2, lactate_base + 0.10 * hour + rng.normal(0, 0.15))
                last_lactate[stay_id] = value
                observation_rows.append(
                    {
                        "stay_id": stay_id,
                        "variable": "lactate",
                        "event_time": admit[stay_id] + pd.Timedelta(hours=hour),
                        "value": value,
                    }
                )
    observations = pd.DataFrame(observation_rows)

    logit = -6.2 + 0.025 * age + 0.025 * last_heart_rate + 0.65 * last_lactate
    risk = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, risk)
    preexisting = rng.random(n) < 0.05
    event_rows: list[dict[str, object]] = []
    for stay_id in range(n):
        if preexisting[stay_id]:
            start_hour = 4.0
        elif target[stay_id]:
            start_hour = float(rng.uniform(6.0, 17.5))
        else:
            continue
        event_rows.append(
            {
                "stay_id": stay_id,
                "task": "vasopressor_initiation",
                "start_time": admit[stay_id] + pd.Timedelta(hours=start_hour),
                "end_time": admit[stay_id] + pd.Timedelta(hours=start_hour + 2),
            }
        )
    events = pd.DataFrame(event_rows)
    return stays, observations, events


def run(n: int, seed: int) -> dict[str, object]:
    stays, observations, events = generate_synthetic_inputs(n, seed)
    cohort = build_landmark_cohort(stays)
    labeled = assign_task_labels(cohort, events, "vasopressor_initiation")
    labeled = labeled.loc[
        labeled["vasopressor_initiation__primary_analysis_eligible"]
    ].copy()
    features, feature_audit = build_feature_matrix(
        labeled,
        observations,
        ["heart_rate", "lactate"],
    )
    static = labeled[["stay_id", "age", "sex", "vasopressor_initiation__target"]]
    matrix = features.merge(static, on="stay_id", validate="one_to_one")
    matrix = matrix.rename(columns={"vasopressor_initiation__target": "y"})
    matrix = matrix.drop(columns="stay_id")

    split = int(len(matrix) * 0.70)
    train = matrix.iloc[:split].copy()
    test = matrix.iloc[split:].copy()
    _, _, logistic_metrics = fit_logistic_baseline(
        train,
        test,
        target_column="y",
        data_classification="synthetic",
        random_state=seed,
    )
    report: dict[str, object] = {
        "data_classification": "synthetic",
        "n_generated": n,
        "n_primary_analysis": len(matrix),
        "feature_audit": asdict(feature_audit),
        "logistic_regression": asdict(logistic_metrics),
    }
    try:
        _, _, catboost_metrics = fit_catboost_baseline(
            train,
            test,
            target_column="y",
            data_classification="synthetic",
            random_state=seed,
        )
        report["catboost"] = asdict(catboost_metrics)
    except RuntimeError as exc:
        report["catboost"] = {"status": "not_run", "reason": str(exc)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.n < 500:
        raise ValueError("Synthetic dry-run requires at least 500 stays.")
    report = run(args.n, args.seed)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
