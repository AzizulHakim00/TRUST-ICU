from pathlib import Path

import pytest

from trust_icu.outcomes import (
    assert_task_training_allowed,
    classify_event_offset_minutes,
    contract_sha256,
    evaluate_outcome_locks,
    load_outcome_contracts,
)

CONTRACTS = Path(__file__).resolve().parents[1] / "schemas" / "outcome_contracts.yaml"


def test_public_contracts_load_and_have_stable_hash() -> None:
    raw = load_outcome_contracts(CONTRACTS)
    digest = contract_sha256(raw)
    assert len(digest) == 64
    assert digest == contract_sha256(raw)


def test_current_contracts_fail_closed_before_data_access() -> None:
    reports = evaluate_outcome_locks(load_outcome_contracts(CONTRACTS))
    assert len(reports) == 3
    assert all(report.ready_for_model_training is False for report in reports)
    assert all(report.synthetic_timeline_tests_passed is True for report in reports)
    assert all("synthetic_timeline_tests" not in report.missing_requirements for report in reports)
    assert all("eicu_local_validation" in report.missing_requirements for report in reports)


def test_training_guard_blocks_unlocked_task() -> None:
    raw = load_outcome_contracts(CONTRACTS)
    with pytest.raises(RuntimeError, match="Model training is prohibited"):
        assert_task_training_allowed(raw, "vasopressor_initiation")


def test_training_guard_allows_only_fully_locked_task() -> None:
    raw = load_outcome_contracts(CONTRACTS)
    task = raw["outcomes"]["vasopressor_initiation"]
    task["mimic"]["status"] = "locally_validated"
    task["eicu"]["status"] = "locally_validated"
    task["clinical_equivalence_review"] = "approved"
    task["synthetic_timeline_tests"] = "passed"

    report = assert_task_training_allowed(raw, "vasopressor_initiation")
    assert report.ready_for_model_training is True
    assert report.missing_requirements == ()


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (-1, "pre_icu"),
        (0, "observation"),
        (359, "observation"),
        (360, "prediction"),
        (1079, "prediction"),
        (1080, "post_prediction"),
    ],
)
def test_locked_time_boundaries(offset: int, expected: str) -> None:
    assert classify_event_offset_minutes(offset) == expected


def test_invalid_time_boundaries_are_rejected() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        classify_event_offset_minutes(10, observation_end_minutes=360, prediction_end_minutes=360)
