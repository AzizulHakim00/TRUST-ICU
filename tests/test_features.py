import numpy as np
import pandas as pd

from trust_icu.features import build_feature_matrix


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [1, 2],
            "icu_admit_time": pd.to_datetime(["2026-01-01 00:00Z", "2026-01-01 00:00Z"]),
            "landmark_time": pd.to_datetime(["2026-01-01 06:00Z", "2026-01-01 06:00Z"]),
        }
    )


def test_feature_builder_excludes_future_rows_and_adds_missingness() -> None:
    observations = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1],
            "variable": ["heart_rate", "heart_rate", "heart_rate", "unknown"],
            "event_time": pd.to_datetime(
                [
                    "2026-01-01 01:00Z",
                    "2026-01-01 05:00Z",
                    "2026-01-01 06:00Z",
                    "2026-01-01 02:00Z",
                ]
            ),
            "value": [80, 100, 200, 1],
        }
    )
    matrix, audit = build_feature_matrix(_cohort(), observations, ["heart_rate"])
    row1 = matrix.set_index("stay_id").loc[1]
    row2 = matrix.set_index("stay_id").loc[2]
    assert row1["heart_rate__last"] == 100
    assert row1["heart_rate__mean"] == 90
    assert row1["heart_rate__count"] == 2
    assert row1["heart_rate__hours_since_last"] == 1
    assert row2["heart_rate__missing"] == 1
    assert np.isnan(row2["heart_rate__last"])
    assert audit.rows_at_or_after_landmark == 1
    assert audit.unknown_variables == ("unknown",)


def test_duplicate_timestamp_values_are_mean_collapsed() -> None:
    observations = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "variable": ["lactate", "lactate"],
            "event_time": pd.to_datetime(["2026-01-01 03:00Z", "2026-01-01 03:00Z"]),
            "value": [2.0, 4.0],
        }
    )
    matrix, audit = build_feature_matrix(_cohort(), observations, ["lactate"])
    assert matrix.set_index("stay_id").loc[1, "lactate__last"] == 3.0
    assert audit.duplicate_stay_variable_time_rows == 2
