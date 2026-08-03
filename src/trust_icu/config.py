"""Typed loading and validation for the preregistered study configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FeasibilityGates:
    minimum_positive_events_per_primary_task: int
    minimum_external_positive_events_per_primary_task: int
    minimum_event_rate: float
    maximum_event_rate: float
    minimum_external_pr_auc_prevalence_ratio: float
    maximum_external_calibration_slope_deviation: float
    maximum_absolute_external_calibration_intercept: float
    minimum_number_of_external_hospitals: int
    require_no_post_index_leakage: bool
    require_outcome_definition_equivalence: bool
    require_all_checks_to_continue: bool

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FeasibilityGates":
        gates = cls(**values)
        if gates.minimum_positive_events_per_primary_task <= 0:
            raise ValueError("Development positive-event minimum must be greater than zero.")
        if gates.minimum_external_positive_events_per_primary_task <= 0:
            raise ValueError("External positive-event minimum must be greater than zero.")
        if not 0 < gates.minimum_event_rate < gates.maximum_event_rate < 1:
            raise ValueError("Event-rate limits must satisfy 0 < minimum < maximum < 1.")
        if gates.minimum_external_pr_auc_prevalence_ratio <= 1:
            raise ValueError("PR-AUC/prevalence gate must be greater than one.")
        if gates.minimum_number_of_external_hospitals < 2:
            raise ValueError("External validation must include at least two hospitals.")
        return gates


@dataclass(frozen=True)
class StudyConfig:
    name: str
    phase: str
    seed: int
    observation_window_hours: int
    prediction_horizon_hours: int
    minimum_age: int
    gates: FeasibilityGates

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StudyConfig":
        study = raw["study"]
        config = cls(
            name=str(study["name"]),
            phase=str(study["phase"]),
            seed=int(study["seed"]),
            observation_window_hours=int(study["observation_window_hours"]),
            prediction_horizon_hours=int(study["prediction_horizon_hours"]),
            minimum_age=int(study["minimum_age"]),
            gates=FeasibilityGates.from_dict(raw["feasibility_gates"]),
        )
        if config.observation_window_hours <= 0:
            raise ValueError("Observation window must be positive.")
        if config.prediction_horizon_hours <= 0:
            raise ValueError("Prediction horizon must be positive.")
        if config.minimum_age < 18:
            raise ValueError("Phase 0 is preregistered for an adult population.")
        return config


def load_config(path: str | Path) -> StudyConfig:
    """Load and validate a TRUST-ICU YAML configuration."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping.")
    return StudyConfig.from_dict(raw)
