from pathlib import Path

import pandas as pd

from trust_icu.features import load_feature_contract
from trust_icu.source_validation import (
    audit_canonical_extract,
    load_source_adapter_contract,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "schemas" / "source_adapter_contract.yaml"
FEATURES = ROOT / "schemas" / "phase0_features.yaml"


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    admit = pd.to_datetime(["2026-01-01T00:00Z", "2026-01-02T00:00Z"])
    stays = pd.DataFrame(
        {
            "dataset_id": ["mimic_iv_3_1", "mimic_iv_3_1"],
            "patient_id": [1, 2],
            "hospital_admission_id": [11, 22],
            "stay_id": [111, 222],
            "hospital_id": [1, 1],
            "icu_admit_time": admit,
            "icu_discharge_time": admit + pd.Timedelta(hours=24),
            "age": [60, 70],
            "sex": ["F", "M"],
        }
    )
    events = pd.DataFrame(
        {
            "stay_id": [111],
            "task": ["vasopressor_initiation"],
            "start_time": [admit[0] + pd.Timedelta(hours=8)],
            "end_time": [admit[0] + pd.Timedelta(hours=10)],
            "source_table": ["mimiciv_derived.norepinephrine"],
            "source_code": ["norepinephrine"],
        }
    )
    observations = pd.DataFrame(
        {
            "stay_id": [111, 222],
            "variable": ["heart_rate", "lactate"],
            "event_time": [
                admit[0] + pd.Timedelta(hours=2),
                admit[1] + pd.Timedelta(hours=5),
            ],
            "value": [90.0, 2.2],
            "unit": ["beats/min", "mmol/L"],
            "source_table": [
                "mimiciv_derived.vitalsign",
                "mimiciv_hosp.labevents",
            ],
            "source_code": ["heart_rate", "50813"],
        }
    )
    return stays, events, observations


def _audit(
    stays: pd.DataFrame,
    events: pd.DataFrame,
    observations: pd.DataFrame,
):
    return audit_canonical_extract(
        stays,
        events,
        observations,
        dataset="mimic_iv_3_1",
        source_contract=load_source_adapter_contract(SOURCE),
        feature_contract=load_feature_contract(FEATURES),
    )


def test_good_extract_is_ready() -> None:
    audit = _audit(*_inputs())
    assert audit.ready_for_cohort_build is True
    assert audit.critical_failures == ()
    assert audit.unique_stays == 2


def test_future_observation_blocks_extract() -> None:
    stays, events, observations = _inputs()
    observations.loc[0, "event_time"] = pd.Timestamp("2026-01-01T06:00Z")
    audit = _audit(stays, events, observations)
    assert audit.ready_for_cohort_build is False
    assert "observations_at_or_after_landmark" in audit.critical_failures


def test_unit_mismatch_blocks_extract() -> None:
    stays, events, observations = _inputs()
    observations.loc[1, "unit"] = "mg/dL"
    audit = _audit(stays, events, observations)
    assert audit.ready_for_cohort_build is False
    assert "unit_mismatches" in audit.critical_failures


def test_unlinked_and_unknown_rows_are_reported() -> None:
    stays, events, observations = _inputs()
    events.loc[0, "stay_id"] = 999
    events.loc[0, "task"] = "unknown_task"
    observations.loc[0, "variable"] = "future_score"
    audit = _audit(stays, events, observations)
    assert "unlinked_event_rows" in audit.critical_failures
    assert "unknown_tasks" in audit.critical_failures
    assert "unknown_variables" in audit.critical_failures


def test_missing_provenance_blocks_extract() -> None:
    stays, events, observations = _inputs()
    events.loc[0, "source_code"] = ""
    observations.loc[0, "source_table"] = None
    audit = _audit(stays, events, observations)
    assert "missing_event_provenance" in audit.critical_failures
    assert "missing_observation_provenance" in audit.critical_failures


def test_duplicate_stays_and_missing_values_fail_closed() -> None:
    stays, events, observations = _inputs()
    stays = pd.concat([stays, stays.iloc[[0]]], ignore_index=True)
    observations.loc[0, "value"] = None
    audit = _audit(stays, events, observations)
    assert "duplicate_stay_ids" in audit.critical_failures
    assert "invalid_observation_values" in audit.critical_failures
    assert "missing_observation_core_values" in audit.critical_failures
