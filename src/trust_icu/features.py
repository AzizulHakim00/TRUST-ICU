"""Leakage-safe aggregation of canonical observations into six-hour feature matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

_REQUIRED_OBSERVATION_COLUMNS: Final = {"stay_id", "variable", "event_time", "value"}
_SUMMARIES: Final = ("first", "last", "min", "max", "mean", "slope", "count")


@dataclass(frozen=True)
class FeatureMatrixAudit:
    cohort_rows: int
    observation_rows_received: int
    observation_rows_used: int
    rows_before_icu: int
    rows_at_or_after_landmark: int
    unknown_variables: tuple[str, ...]
    duplicate_stay_variable_time_rows: int


def _slope_per_hour(times: pd.Series, values: pd.Series) -> float:
    if len(values) < 2:
        return float("nan")
    elapsed = (times - times.iloc[0]).dt.total_seconds().to_numpy() / 3600.0
    numeric = values.to_numpy(dtype=float)
    if np.ptp(elapsed) == 0:
        return float("nan")
    return float(np.polyfit(elapsed, numeric, 1)[0])


def build_feature_matrix(
    cohort: pd.DataFrame,
    observations: pd.DataFrame,
    variables: list[str] | tuple[str, ...],
) -> tuple[pd.DataFrame, FeatureMatrixAudit]:
    """Aggregate observations strictly before each stay's landmark.

    For every canonical variable the matrix contains first, last, minimum, maximum, mean,
    linear slope per hour, count, missingness and hours since the last measurement. Rows outside
    ``[ICU admission, landmark)`` are audited and excluded, never clipped into the window.
    """

    required_cohort = {"stay_id", "icu_admit_time", "landmark_time"}
    missing_cohort = sorted(required_cohort - set(cohort.columns))
    if missing_cohort:
        raise ValueError(f"cohort is missing required columns: {missing_cohort}")
    missing_obs = sorted(_REQUIRED_OBSERVATION_COLUMNS - set(observations.columns))
    if missing_obs:
        raise ValueError(f"observations is missing required columns: {missing_obs}")
    canonical = tuple(dict.fromkeys(str(variable) for variable in variables))
    if not canonical:
        raise ValueError("At least one canonical variable is required.")

    obs = observations.copy()
    obs["event_time"] = pd.to_datetime(obs["event_time"], errors="coerce", utc=True)
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    if obs["event_time"].isna().any():
        raise ValueError("Observation event times must be parseable.")
    unknown = tuple(sorted(set(obs["variable"].astype(str)) - set(canonical)))
    obs = obs.loc[obs["variable"].isin(canonical)].copy()

    windows = cohort[["stay_id", "icu_admit_time", "landmark_time"]].copy()
    windows["icu_admit_time"] = pd.to_datetime(
        windows["icu_admit_time"], errors="coerce", utc=True
    )
    windows["landmark_time"] = pd.to_datetime(
        windows["landmark_time"], errors="coerce", utc=True
    )
    if windows[["icu_admit_time", "landmark_time"]].isna().any().any():
        raise ValueError("Cohort window times must be parseable.")
    if windows["stay_id"].duplicated().any():
        raise ValueError("Cohort must contain one row per stay_id.")

    merged = obs.merge(windows, on="stay_id", how="inner", validate="many_to_one")
    before_icu = merged["event_time"].lt(merged["icu_admit_time"])
    at_or_after = merged["event_time"].ge(merged["landmark_time"])
    usable = merged.loc[~before_icu & ~at_or_after & merged["value"].notna()].copy()
    duplicate_count = int(
        usable.duplicated(["stay_id", "variable", "event_time"], keep=False).sum()
    )
    usable = (
        usable.groupby(["stay_id", "variable", "event_time"], as_index=False, sort=False)
        .agg(value=("value", "mean"), landmark_time=("landmark_time", "first"))
        .sort_values(["stay_id", "variable", "event_time"], kind="mergesort")
    )

    rows: list[dict[str, float | int]] = []
    for (stay_id, variable), group in usable.groupby(
        ["stay_id", "variable"], sort=False
    ):
        values = group["value"]
        times = group["event_time"]
        rows.append(
            {
                "stay_id": stay_id,
                "variable": variable,
                "first": float(values.iloc[0]),
                "last": float(values.iloc[-1]),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "slope": _slope_per_hour(times, values),
                "count": int(len(values)),
                "hours_since_last": float(
                    (group["landmark_time"].iloc[0] - times.iloc[-1]).total_seconds()
                    / 3600.0
                ),
            }
        )
    summary = pd.DataFrame(rows)
    matrix = cohort[["stay_id"]].drop_duplicates().set_index("stay_id")
    for variable in canonical:
        subset = summary.loc[summary.get("variable", pd.Series(dtype=str)).eq(variable)]
        indexed = subset.set_index("stay_id") if not subset.empty else None
        for statistic in _SUMMARIES:
            column = f"{variable}__{statistic}"
            if indexed is None:
                matrix[column] = 0 if statistic == "count" else np.nan
            else:
                matrix[column] = indexed[statistic].reindex(matrix.index)
        since_column = f"{variable}__hours_since_last"
        matrix[since_column] = (
            indexed["hours_since_last"].reindex(matrix.index)
            if indexed is not None
            else np.nan
        )
        matrix[f"{variable}__count"] = matrix[f"{variable}__count"].fillna(0).astype("int32")
        matrix[f"{variable}__missing"] = matrix[f"{variable}__count"].eq(0).astype("int8")

    matrix = matrix.reset_index()
    audit = FeatureMatrixAudit(
        cohort_rows=len(matrix),
        observation_rows_received=len(observations),
        observation_rows_used=len(usable),
        rows_before_icu=int(before_icu.sum()),
        rows_at_or_after_landmark=int(at_or_after.sum()),
        unknown_variables=unknown,
        duplicate_stay_variable_time_rows=duplicate_count,
    )
    return matrix, audit
