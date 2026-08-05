import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trust_icu.config import FeasibilityGates, StudyConfig
from trust_icu.features import FeatureContract, VariableSpec
from trust_icu.phase0_runner import _canonical_hash, _sha256_file
from trust_icu.phase0_runtime import (
    build_feature_matrix_from_extract,
    run_task_phase0,
    temporal_patient_purged_split,
    verify_credentialed_run,
)


def _study_config() -> StudyConfig:
    return StudyConfig(
        name="TRUST-ICU",
        phase="phase_0_feasibility",
        seed=20260804,
        observation_window_hours=6,
        prediction_horizon_hours=12,
        minimum_age=18,
        gates=FeasibilityGates(
            minimum_positive_events_per_primary_task=2000,
            minimum_external_positive_events_per_primary_task=500,
            minimum_event_rate=0.01,
            maximum_event_rate=0.50,
            minimum_external_pr_auc_prevalence_ratio=2.0,
            maximum_external_calibration_slope_deviation=0.35,
            maximum_absolute_external_calibration_intercept=0.75,
            minimum_number_of_external_hospitals=20,
            require_no_post_index_leakage=True,
            require_outcome_definition_equivalence=True,
            require_all_checks_to_continue=True,
        ),
    )


def _raw_config() -> dict[str, object]:
    return {
        "validation": {
            "temporal_split": {
                "train_fraction": 0.70,
                "calibration_fraction": 0.15,
                "test_fraction": 0.15,
                "patient_overlap_policy": "purge_later_splits",
            },
            "bootstrap_iterations": 30,
            "confidence_level": 0.95,
            "hospital_metric_minimum_rows": 20,
            "hospital_metric_minimum_events": 2,
        }
    }


def _task_data(n: int, task: str, *, external: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42 if external else 7)
    age = rng.normal(63, 11, n)
    lactate = rng.lognormal(0.2, 0.45, n)
    target = (np.arange(n) % 4 == 0).astype("int8")
    frame = pd.DataFrame(
        {
            "age": age,
            "sex": np.where(np.arange(n) % 2 == 0, "female", "male"),
            "lactate__last": lactate + target * 0.8,
            "lactate__count": np.where(np.arange(n) % 7 == 0, 0, 2),
            "lactate__missing": np.where(np.arange(n) % 7 == 0, 1, 0),
            f"{task}__target": target,
        }
    )
    metadata = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(n)],
            "icu_admit_time": pd.date_range(
                "2018-01-01" if external else "2012-01-01",
                periods=n,
                freq="h",
                tz="UTC",
            ),
            f"{task}__target": target,
        }
    )
    if external:
        metadata["hospital_id"] = [f"h{i // 20}" for i in range(n)]
    return frame, metadata


def test_temporal_split_purges_patients_from_later_splits() -> None:
    metadata = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(100)],
            "icu_admit_time": pd.date_range("2020-01-01", periods=100, freq="h", tz="UTC"),
        }
    )
    metadata.loc[75, "patient_id"] = metadata.loc[5, "patient_id"]
    metadata.loc[95, "patient_id"] = metadata.loc[65, "patient_id"]
    splits, report = temporal_patient_purged_split(metadata, (0.70, 0.15, 0.15))
    assert report.purged_calibration_rows == 1
    assert report.purged_temporal_test_rows == 1
    assert report.patient_overlap_after_purge is False
    assert not set(metadata.loc[splits["train"], "patient_id"]) & set(
        metadata.loc[splits["temporal_test"], "patient_id"]
    )


def test_locked_task_modeling_runs_without_saving_predictions() -> None:
    task = "vasopressor_initiation"
    mimic_matrix, mimic_metadata = _task_data(800, task, external=False)
    eicu_matrix, eicu_metadata = _task_data(600, task, external=True)
    report = run_task_phase0(
        task=task,
        mimic_matrix=mimic_matrix,
        mimic_metadata=mimic_metadata,
        eicu_matrix=eicu_matrix,
        eicu_metadata=eicu_metadata,
        config=_study_config(),
        raw_config=_raw_config(),
        models=("logistic_regression", "catboost"),
    )
    assert report.selected_model in {"logistic_regression", "catboost"}
    assert len(report.models) == 2
    assert len(report.external_cluster_bootstrap) == 3
    assert report.hospital_robustness.total_hospitals == 30
    assert report.split.patient_overlap_after_purge is False
    assert all(model.external_metrics_calibrated.n == 600 for model in report.models)
    assert report.feasibility_decision.continue_to_architecture_development is False


def _write_gzip(path: Path, text: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(text)


def _fake_credentialed_run(tmp_path: Path) -> Path:
    dataset = "mimic_iv_3_1"
    run_dir = tmp_path / dataset
    run_dir.mkdir()
    contents = {
        "stays": "stay_id,patient_id\n1,p1\n",
        "events": "stay_id,task,start_time\n",
        "observations": "stay_id,variable,event_time,value\n",
    }
    exports = []
    for kind, text in contents.items():
        path = run_dir / f"{dataset}_{kind}.csv.gz"
        _write_gzip(path, text)
        exports.append(
            {
                "name": f"mimic_{kind}",
                "path": str(path),
                "row_count": 1 if kind == "stays" else 0,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    audit = {
        "dataset": dataset,
        "ready_for_cohort_build": True,
        "critical_failures": [],
    }
    (run_dir / "canonical_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "dataset": dataset,
        "audit": audit,
        "exports": exports,
        "report_sha256": "",
    }
    report["report_sha256"] = _canonical_hash(report, "report_sha256")
    (run_dir / "credentialed_run_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_dir


def test_credentialed_run_hashes_are_verified(tmp_path: Path) -> None:
    run_dir = _fake_credentialed_run(tmp_path)
    verified = verify_credentialed_run(run_dir, "mimic_iv_3_1")
    assert verified.dataset == "mimic_iv_3_1"
    _write_gzip(Path(verified.events_path), "tampered\n")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_credentialed_run(run_dir, "mimic_iv_3_1")


def test_duckdb_feature_builder_uses_locked_window_and_bounds(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    stays = tmp_path / "stays.csv.gz"
    observations = tmp_path / "observations.csv.gz"
    _write_gzip(
        stays,
        "stay_id,age,sex,icu_admit_time\n1,60,female,2020-01-01T00:00:00Z\n",
    )
    _write_gzip(
        observations,
        "stay_id,variable,event_time,value\n"
        "1,heart_rate,2019-12-31T23:00:00Z,70\n"
        "1,heart_rate,2020-01-01T01:00:00Z,80\n"
        "1,heart_rate,2020-01-01T02:00:00Z,100\n"
        "1,heart_rate,2020-01-01T03:00:00Z,500\n"
        "1,heart_rate,2020-01-01T06:00:00Z,90\n",
    )
    contract = FeatureContract(
        version="test",
        observation_start_minutes=0,
        observation_end_minutes=360,
        variables=(VariableSpec("heart_rate", "beats/min", 20, 300),),
    )
    matrix, audit = build_feature_matrix_from_extract(stays, observations, contract)
    assert matrix.loc[0, "heart_rate__count"] == 2
    assert matrix.loc[0, "heart_rate__first"] == 80
    assert matrix.loc[0, "heart_rate__last"] == 100
    assert matrix.loc[0, "heart_rate__mean"] == 90
    assert matrix.loc[0, "heart_rate__missing"] == 0
    assert audit.rows_before_icu == 1
    assert audit.rows_at_or_after_landmark == 1
    assert audit.out_of_range_rows == 1
