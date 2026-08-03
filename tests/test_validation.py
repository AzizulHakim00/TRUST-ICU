from pathlib import Path

import pytest

from trust_icu.config import load_config
from trust_icu.validation import DatasetAudit, evaluate_feasibility


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "feasibility.yaml"


def passing_audit() -> DatasetAudit:
    return DatasetAudit(
        task="vasopressor_initiation",
        development_n=40_000,
        development_events=4_000,
        external_n=60_000,
        external_events=5_000,
        external_hospitals=120,
        external_pr_auc=0.20,
        external_brier=0.07,
        external_calibration_slope=0.90,
        external_calibration_intercept=0.10,
        no_post_index_leakage=True,
        outcome_definition_equivalent=True,
    )


def test_valid_config_loads() -> None:
    config = load_config(CONFIG)
    assert config.name == "TRUST-ICU"
    assert config.observation_window_hours == 6
    assert config.prediction_horizon_hours == 12


def test_all_gates_pass_for_prespecified_synthetic_audit() -> None:
    decision = evaluate_feasibility(passing_audit(), load_config(CONFIG))
    assert decision.continue_to_architecture_development is True
    assert decision.passed_checks == decision.total_checks


def test_leakage_forces_stop() -> None:
    values = passing_audit().__dict__ | {"no_post_index_leakage": False}
    decision = evaluate_feasibility(DatasetAudit(**values), load_config(CONFIG))
    assert decision.continue_to_architecture_development is False
    leakage_check = next(check for check in decision.checks if check.name == "no_post_index_leakage")
    assert leakage_check.passed is False


def test_weak_external_pr_auc_forces_stop() -> None:
    values = passing_audit().__dict__ | {"external_pr_auc": 0.10}
    decision = evaluate_feasibility(DatasetAudit(**values), load_config(CONFIG))
    assert decision.continue_to_architecture_development is False


def test_invalid_event_count_is_rejected() -> None:
    values = passing_audit().__dict__ | {"external_events": 60_001}
    with pytest.raises(ValueError, match="external_events"):
        evaluate_feasibility(DatasetAudit(**values), load_config(CONFIG))
