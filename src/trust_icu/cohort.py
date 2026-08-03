"""Landmark-cohort construction and task-label assignment.

The functions are database-agnostic. Source-specific SQL must emit the canonical columns
used here. Restricted patient-level data stay outside the public repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

_REQUIRED_STAY_COLUMNS: Final = {
    "patient_id",
    "hospital_admission_id",
    "stay_id",
    "icu_admit_time",
    "icu_discharge_time",
    "age",
}
_REQUIRED_EVENT_COLUMNS: Final = {"stay_id", "task", "start_time"}


@dataclass(frozen=True)
class LandmarkSpec:
    """Time rules for the Phase 0 landmark cohort."""

    observation_end_minutes: int = 360
    prediction_end_minutes: int = 1080
    minimum_age: int = 18

    def validate(self) -> None:
        if self.observation_end_minutes <= 0:
            raise ValueError("observation_end_minutes must be positive.")
        if self.prediction_end_minutes <= self.observation_end_minutes:
            raise ValueError("prediction_end_minutes must exceed observation_end_minutes.")
        if self.minimum_age < 18:
            raise ValueError("Phase 0 is restricted to adults.")


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def build_landmark_cohort(
    stays: pd.DataFrame,
    spec: LandmarkSpec | None = None,
) -> pd.DataFrame:
    """Build one first eligible ICU unit stay per hospital admission.

    Eligibility is assessed at the six-hour landmark. A stay must represent an adult who is
    alive and still in the ICU at the landmark. Follow-up ends at the earlier of ICU discharge
    or the administrative 18-hour boundary. Cross-admission chronology is never inferred.
    """

    resolved_spec = spec or LandmarkSpec()
    resolved_spec.validate()
    _require_columns(stays, _REQUIRED_STAY_COLUMNS, "stays")
    frame = stays.copy()
    for column in ("icu_admit_time", "icu_discharge_time", "death_time"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")

    if frame[["icu_admit_time", "icu_discharge_time"]].isna().any().any():
        raise ValueError("ICU admission and discharge times must be present and parseable.")
    if (frame["icu_discharge_time"] <= frame["icu_admit_time"]).any():
        raise ValueError("Every ICU discharge time must be after ICU admission time.")

    frame["landmark_time"] = frame["icu_admit_time"] + pd.to_timedelta(
        resolved_spec.observation_end_minutes, unit="m"
    )
    frame["administrative_end_time"] = frame["icu_admit_time"] + pd.to_timedelta(
        resolved_spec.prediction_end_minutes, unit="m"
    )

    alive_at_landmark = pd.Series(True, index=frame.index)
    if "death_time" in frame:
        alive_at_landmark = frame["death_time"].isna() | (
            frame["death_time"] > frame["landmark_time"]
        )

    eligible = (
        frame["age"].ge(resolved_spec.minimum_age)
        & frame["icu_discharge_time"].gt(frame["landmark_time"])
        & alive_at_landmark
    )
    frame = frame.loc[eligible].copy()
    frame = frame.sort_values(
        ["hospital_admission_id", "icu_admit_time", "stay_id"], kind="mergesort"
    )
    frame = frame.drop_duplicates("hospital_admission_id", keep="first")

    frame["followup_end_time"] = frame[
        ["icu_discharge_time", "administrative_end_time"]
    ].min(axis=1)
    if "death_time" in frame:
        frame["death_before_administrative_end"] = (
            frame["death_time"].notna()
            & frame["death_time"].gt(frame["landmark_time"])
            & frame["death_time"].lt(frame["administrative_end_time"])
        )
        frame["followup_end_time"] = pd.concat(
            [frame["followup_end_time"], frame["death_time"]], axis=1
        ).min(axis=1)
    else:
        frame["death_before_administrative_end"] = False

    ordered = [
        "patient_id",
        "hospital_admission_id",
        "stay_id",
        "age",
        "icu_admit_time",
        "landmark_time",
        "followup_end_time",
        "administrative_end_time",
        "icu_discharge_time",
        "death_before_administrative_end",
    ]
    remainder = [column for column in frame.columns if column not in ordered]
    return frame[ordered + remainder].reset_index(drop=True)


def assign_task_labels(
    cohort: pd.DataFrame,
    events: pd.DataFrame,
    task: str,
) -> pd.DataFrame:
    """Assign an outcome while excluding support active during the observation window.

    Intervals overlap the observation window when ``start < landmark`` and ``end > ICU admit``.
    A missing event end is treated as ongoing. New starts are positive only inside the locked
    ``[landmark, followup_end)`` interval. Death before the horizon without an event is flagged
    as a competing event and excluded from the primary binary analysis.
    """

    _require_columns(
        cohort,
        {"stay_id", "icu_admit_time", "landmark_time", "followup_end_time"},
        "cohort",
    )
    _require_columns(events, _REQUIRED_EVENT_COLUMNS, "events")
    labeled = cohort.copy()
    task_events = events.loc[events["task"].eq(task)].copy()
    task_events["start_time"] = pd.to_datetime(
        task_events["start_time"], errors="coerce", utc=True
    )
    if "end_time" not in task_events:
        task_events["end_time"] = pd.NaT
    else:
        task_events["end_time"] = pd.to_datetime(
            task_events["end_time"], errors="coerce", utc=True
        )
    if task_events["start_time"].isna().any():
        raise ValueError("All task-event start times must be parseable.")
    invalid_end = task_events["end_time"].notna() & (
        task_events["end_time"] <= task_events["start_time"]
    )
    if invalid_end.any():
        raise ValueError("Event end times must be after start times.")

    joined = task_events.merge(
        labeled[
            ["stay_id", "icu_admit_time", "landmark_time", "followup_end_time"]
        ],
        on="stay_id",
        how="inner",
        validate="many_to_one",
    )
    observation_overlap = joined["start_time"].lt(joined["landmark_time"]) & (
        joined["end_time"].isna()
        | joined["end_time"].gt(joined["icu_admit_time"])
    )
    active_stays = set(joined.loc[observation_overlap, "stay_id"])

    prediction_start = joined["start_time"].ge(joined["landmark_time"])
    before_followup_end = joined["start_time"].lt(joined["followup_end_time"])
    incident = joined.loc[prediction_start & before_followup_end]
    first_start = incident.groupby("stay_id", sort=False)["start_time"].min()

    labeled[f"{task}__observation_active"] = labeled["stay_id"].isin(active_stays)
    labeled[f"{task}__eligible"] = ~labeled[f"{task}__observation_active"]
    labeled[f"{task}__first_start_time"] = labeled["stay_id"].map(first_start)
    labeled[f"{task}__target"] = (
        labeled[f"{task}__eligible"]
        & labeled[f"{task}__first_start_time"].notna()
    ).astype("int8")
    labeled[f"{task}__competing_death"] = (
        labeled[f"{task}__eligible"]
        & labeled["death_before_administrative_end"].astype(bool)
        & labeled[f"{task}__first_start_time"].isna()
    )
    labeled[f"{task}__primary_analysis_eligible"] = (
        labeled[f"{task}__eligible"]
        & ~labeled[f"{task}__competing_death"]
    )
    return labeled
