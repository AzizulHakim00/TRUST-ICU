"""Validation and aggregate audit of canonical source-adapter extracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from trust_icu.features import FeatureContract


@dataclass(frozen=True)
class CanonicalExtractAudit:
    dataset: str
    source_contract_sha256: str
    feature_contract_sha256: str
    stay_rows: int
    event_rows: int
    observation_rows: int
    unique_stays: int
    unknown_tasks: tuple[str, ...]
    unknown_variables: tuple[str, ...]
    unit_mismatches: tuple[str, ...]
    duplicate_stay_ids: int
    duplicate_observation_rows: int
    unlinked_event_rows: int
    unlinked_observation_rows: int
    observations_before_icu: int
    observations_at_or_after_landmark: int
    invalid_observation_values: int
    invalid_icu_intervals: int
    invalid_event_intervals: int
    missing_stay_core_values: int
    missing_event_core_values: int
    missing_observation_core_values: int
    missing_event_provenance: int
    missing_observation_provenance: int
    critical_failures: tuple[str, ...]
    warnings: tuple[str, ...]
    ready_for_cohort_build: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(raw: dict[str, Any]) -> str:
    return json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def mapping_sha256(raw: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(raw).encode("utf-8")).hexdigest()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Contract not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Contract root must be a mapping: {source}")
    return raw


def load_source_adapter_contract(path: str | Path) -> dict[str, Any]:
    raw = _load_yaml(path)
    validate_source_adapter_contract(raw)
    return raw


def validate_source_adapter_contract(raw: dict[str, Any]) -> None:
    datasets = raw.get("allowed_datasets")
    tasks = raw.get("allowed_tasks")
    tables = raw.get("tables")
    window = raw.get("observation_window")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("allowed_datasets must be a non-empty list.")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("allowed_tasks must be a non-empty list.")
    if not isinstance(tables, dict):
        raise ValueError("tables must be a mapping.")
    for table in ("stays", "events", "observations"):
        spec = tables.get(table)
        if not isinstance(spec, dict):
            raise ValueError(f"Missing table contract: {table}")
        required = spec.get("required_columns")
        if not isinstance(required, list) or not required:
            raise ValueError(f"{table}.required_columns must be a non-empty list.")
    if not isinstance(window, dict):
        raise ValueError("observation_window must be a mapping.")
    start = int(window["start_minutes"])
    end = int(window["end_minutes"])
    if start != 0 or end <= start:
        raise ValueError("Observation window must start at zero and have a positive end.")
    if window.get("interval") != "left_closed_right_open":
        raise ValueError("Observation window must be left-closed and right-open.")


def _require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _parse_utc(series: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().sum() > series.isna().sum():
        raise ValueError(f"{label} contains unparseable timestamps.")
    return parsed


def _blank_count(frame: pd.DataFrame, columns: tuple[str, ...]) -> int:
    mask = pd.Series(False, index=frame.index)
    for column in columns:
        values = frame[column]
        mask |= values.isna() | values.astype(str).str.strip().eq("")
    return int(mask.sum())


def audit_canonical_extract(
    stays: pd.DataFrame,
    events: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    dataset: str,
    source_contract: dict[str, Any],
    feature_contract: FeatureContract,
) -> CanonicalExtractAudit:
    """Audit canonical adapter outputs without returning row-level examples."""

    validate_source_adapter_contract(source_contract)
    if dataset not in source_contract["allowed_datasets"]:
        raise ValueError(f"Unsupported dataset {dataset!r}.")

    table_specs = source_contract["tables"]
    _require_columns(stays, table_specs["stays"]["required_columns"], "stays")
    _require_columns(events, table_specs["events"]["required_columns"], "events")
    _require_columns(
        observations,
        table_specs["observations"]["required_columns"],
        "observations",
    )

    stays_work = stays.copy()
    events_work = events.copy()
    obs_work = observations.copy()

    dataset_values = set(stays_work["dataset_id"].dropna().astype(str))
    if dataset_values != {dataset}:
        raise ValueError(
            f"stays.dataset_id must contain only {dataset!r}; got {sorted(dataset_values)}."
        )

    stays_work["icu_admit_time"] = _parse_utc(
        stays_work["icu_admit_time"], "stays.icu_admit_time"
    )
    stays_work["icu_discharge_time"] = _parse_utc(
        stays_work["icu_discharge_time"], "stays.icu_discharge_time"
    )
    if "death_time" in stays_work:
        stays_work["death_time"] = _parse_utc(
            stays_work["death_time"], "stays.death_time"
        )

    events_work["start_time"] = _parse_utc(
        events_work["start_time"], "events.start_time"
    )
    events_work["end_time"] = _parse_utc(events_work["end_time"], "events.end_time")
    obs_work["event_time"] = _parse_utc(
        obs_work["event_time"], "observations.event_time"
    )
    numeric_values = pd.to_numeric(obs_work["value"], errors="coerce")
    invalid_observation_values = int(numeric_values.isna().sum())

    duplicate_stays = int(stays_work["stay_id"].duplicated(keep=False).sum())
    duplicate_observations = int(
        obs_work.duplicated(
            [
                "stay_id",
                "variable",
                "event_time",
                "value",
                "unit",
                "source_table",
                "source_code",
            ],
            keep=False,
        ).sum()
    )
    invalid_icu = int(
        stays_work["icu_discharge_time"].le(stays_work["icu_admit_time"]).sum()
    )
    invalid_event = int(
        (
            events_work["end_time"].notna()
            & events_work["end_time"].le(events_work["start_time"])
        ).sum()
    )

    valid_stays = set(stays_work["stay_id"])
    unlinked_events = int((~events_work["stay_id"].isin(valid_stays)).sum())
    unlinked_observations = int((~obs_work["stay_id"].isin(valid_stays)).sum())

    allowed_tasks = set(source_contract["allowed_tasks"])
    unknown_tasks = tuple(
        sorted(set(events_work["task"].dropna().astype(str)) - allowed_tasks)
    )
    variables = {variable.name: variable for variable in feature_contract.variables}
    unknown_variables = tuple(
        sorted(set(obs_work["variable"].dropna().astype(str)) - set(variables))
    )

    expected_units = {
        name: variable.canonical_unit for name, variable in variables.items()
    }
    mismatches: set[str] = set()
    distinct_pairs = obs_work[["variable", "unit"]].dropna().astype(str).drop_duplicates()
    for variable, unit in distinct_pairs.itertuples(index=False):
        expected = expected_units.get(variable)
        if expected is not None and unit != expected:
            mismatches.add(f"{variable}:{unit}->{expected}")
    unit_mismatches = tuple(sorted(mismatches))

    windows = (
        stays_work[["stay_id", "icu_admit_time"]]
        .drop_duplicates("stay_id", keep="first")
        .copy()
    )
    window_end = int(source_contract["observation_window"]["end_minutes"])
    windows["landmark_time"] = windows["icu_admit_time"] + pd.to_timedelta(
        window_end, unit="m"
    )
    linked_obs = obs_work.merge(windows, on="stay_id", how="left", validate="many_to_one")
    before_icu = int(
        (
            linked_obs["icu_admit_time"].notna()
            & linked_obs["event_time"].lt(linked_obs["icu_admit_time"])
        ).sum()
    )
    at_or_after = int(
        (
            linked_obs["landmark_time"].notna()
            & linked_obs["event_time"].ge(linked_obs["landmark_time"])
        ).sum()
    )

    missing_stay_core_values = int(
        stays_work[
            [
                "dataset_id",
                "patient_id",
                "hospital_admission_id",
                "stay_id",
                "icu_admit_time",
                "icu_discharge_time",
            ]
        ]
        .isna()
        .any(axis=1)
        .sum()
    )
    missing_event_core_values = int(
        events_work[["stay_id", "task", "start_time"]].isna().any(axis=1).sum()
    )
    missing_observation_core_values = int(
        obs_work[["stay_id", "variable", "event_time", "value", "unit"]]
        .isna()
        .any(axis=1)
        .sum()
    )
    missing_event_provenance = _blank_count(
        events_work, ("source_table", "source_code")
    )
    missing_observation_provenance = _blank_count(
        obs_work, ("source_table", "source_code")
    )

    failures: list[str] = []
    warnings: list[str] = []
    checks = (
        ("duplicate_stay_ids", duplicate_stays),
        ("invalid_icu_intervals", invalid_icu),
        ("invalid_event_intervals", invalid_event),
        ("unlinked_event_rows", unlinked_events),
        ("unlinked_observation_rows", unlinked_observations),
        ("observations_before_icu", before_icu),
        ("observations_at_or_after_landmark", at_or_after),
        ("invalid_observation_values", invalid_observation_values),
        ("missing_stay_core_values", missing_stay_core_values),
        ("missing_event_core_values", missing_event_core_values),
        ("missing_observation_core_values", missing_observation_core_values),
        ("duplicate_observation_rows", duplicate_observations),
        ("missing_event_provenance", missing_event_provenance),
        ("missing_observation_provenance", missing_observation_provenance),
    )
    failures.extend(name for name, count in checks if count > 0)
    if unknown_tasks:
        failures.append("unknown_tasks")
    if unknown_variables:
        failures.append("unknown_variables")
    if unit_mismatches:
        failures.append("unit_mismatches")

    if stays_work["hospital_id"].isna().any():
        warnings.append("missing_hospital_id")
    if stays_work["sex"].isna().any():
        warnings.append("missing_sex")
    if pd.to_numeric(stays_work["age"], errors="coerce").isna().any():
        warnings.append("missing_or_nonnumeric_age")
    if events_work.empty:
        warnings.append("no_event_rows")
    if observations.empty:
        failures.append("no_observation_rows")

    critical_failures = tuple(dict.fromkeys(failures))
    return CanonicalExtractAudit(
        dataset=dataset,
        source_contract_sha256=mapping_sha256(source_contract),
        feature_contract_sha256=mapping_sha256(asdict(feature_contract)),
        stay_rows=len(stays_work),
        event_rows=len(events_work),
        observation_rows=len(obs_work),
        unique_stays=stays_work["stay_id"].nunique(dropna=True),
        unknown_tasks=unknown_tasks,
        unknown_variables=unknown_variables,
        unit_mismatches=unit_mismatches,
        duplicate_stay_ids=duplicate_stays,
        duplicate_observation_rows=duplicate_observations,
        unlinked_event_rows=unlinked_events,
        unlinked_observation_rows=unlinked_observations,
        observations_before_icu=before_icu,
        observations_at_or_after_landmark=at_or_after,
        invalid_observation_values=invalid_observation_values,
        invalid_icu_intervals=invalid_icu,
        invalid_event_intervals=invalid_event,
        missing_stay_core_values=missing_stay_core_values,
        missing_event_core_values=missing_event_core_values,
        missing_observation_core_values=missing_observation_core_values,
        missing_event_provenance=missing_event_provenance,
        missing_observation_provenance=missing_observation_provenance,
        critical_failures=critical_failures,
        warnings=tuple(dict.fromkeys(warnings)),
        ready_for_cohort_build=not critical_failures,
    )
