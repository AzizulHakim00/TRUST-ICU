import pandas as pd

from trust_icu.cohort import LandmarkSpec, assign_task_labels, build_landmark_cohort


def _stays() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 3],
            "hospital_admission_id": [10, 10, 20, 30],
            "stay_id": [100, 101, 200, 300],
            "age": [65, 65, 17, 55],
            "icu_admit_time": pd.to_datetime(
                [
                    "2026-01-01 00:00Z",
                    "2026-01-02 00:00Z",
                    "2026-01-01 00:00Z",
                    "2026-01-01 00:00Z",
                ]
            ),
            "icu_discharge_time": pd.to_datetime(
                [
                    "2026-01-02 00:00Z",
                    "2026-01-03 00:00Z",
                    "2026-01-02 00:00Z",
                    "2026-01-02 00:00Z",
                ]
            ),
            "death_time": pd.to_datetime([None, None, None, "2026-01-01 12:00Z"], utc=True),
        }
    )


def test_landmark_selects_first_eligible_unit_stay_per_admission() -> None:
    cohort = build_landmark_cohort(_stays())
    assert cohort["stay_id"].tolist() == [100, 300]
    assert cohort.loc[0, "landmark_time"] == pd.Timestamp("2026-01-01 06:00Z")


def test_outcome_active_before_landmark_is_excluded() -> None:
    cohort = build_landmark_cohort(_stays())
    events = pd.DataFrame(
        {
            "stay_id": [100, 300],
            "task": ["vent", "vent"],
            "start_time": pd.to_datetime(["2026-01-01 05:00Z", "2026-01-01 08:00Z"]),
            "end_time": pd.to_datetime(["2026-01-01 07:00Z", "2026-01-01 09:00Z"]),
        }
    )
    labeled = assign_task_labels(cohort, events, "vent").set_index("stay_id")
    assert not labeled.loc[100, "vent__eligible"]
    assert labeled.loc[300, "vent__target"] == 1


def test_event_at_six_hours_is_incident_positive() -> None:
    cohort = build_landmark_cohort(_stays())
    events = pd.DataFrame(
        {
            "stay_id": [100],
            "task": ["vaso"],
            "start_time": pd.to_datetime(["2026-01-01 06:00Z"]),
            "end_time": pd.to_datetime(["2026-01-01 07:00Z"]),
        }
    )
    labeled = assign_task_labels(cohort, events, "vaso").set_index("stay_id")
    assert labeled.loc[100, "vaso__target"] == 1


def test_landmark_spec_rejects_invalid_horizon() -> None:
    spec = LandmarkSpec(observation_end_minutes=360, prediction_end_minutes=360)
    try:
        build_landmark_cohort(_stays(), spec)
    except ValueError as exc:
        assert "exceed" in str(exc)
    else:
        raise AssertionError("Invalid landmark specification was accepted.")
