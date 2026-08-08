from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from trust_icu.ecg_protocol import validate_open_ecg_protocol

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "schemas/open_ecg_protocol.yaml"


def _mutated_protocol(tmp_path: Path, mutate) -> Path:
    payload = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(payload)
    mutate(mutated)
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(mutated, sort_keys=False), encoding="utf-8")
    return path


def test_open_ecg_protocol_is_valid() -> None:
    report = validate_open_ecg_protocol(PROTOCOL)
    assert report["valid"] is True
    assert report["version"] == "0.2.0"
    assert report["development_source"] == "ptb_xl"
    assert report["development_records"] == 21837
    assert report["model_fit_folds"] == [1, 2, 3, 4, 5, 6, 7]
    assert report["optimization_validation_fold"] == 8
    assert report["calibration_fold"] == 9
    assert report["internal_test_fold"] == 10
    assert report["external_sources"] == {
        "georgia": 10344,
        "cpsc_2018": 6877,
        "cpsc_2018_extra": 3453,
    }
    assert report["public_primary_records"] == 42511
    assert report["architecture_search_allowed"] is False


def test_hidden_test_set_cannot_be_enabled(tmp_path: Path) -> None:
    path = _mutated_protocol(
        tmp_path,
        lambda payload: payload["resource"]["hidden_challenge_test_set"].update({"use": True}),
    )
    with pytest.raises(ValueError, match="test set must not be used"):
        validate_open_ecg_protocol(path)


def test_external_source_cannot_be_removed(tmp_path: Path) -> None:
    def mutate(payload):
        del payload["sources"]["external_primary"]["georgia"]

    path = _mutated_protocol(tmp_path, mutate)
    with pytest.raises(ValueError, match="External source set"):
        validate_open_ecg_protocol(path)


def test_external_tuning_cannot_be_enabled(tmp_path: Path) -> None:
    path = _mutated_protocol(
        tmp_path,
        lambda payload: payload["internal_validation"].update(
            {"external_data_for_model_selection": "allowed"}
        ),
    )
    with pytest.raises(ValueError, match="External data must not be used"):
        validate_open_ecg_protocol(path)


def test_architecture_search_cannot_be_enabled(tmp_path: Path) -> None:
    path = _mutated_protocol(
        tmp_path,
        lambda payload: payload["phase0_models"]["architecture_search"].update({"allowed": True}),
    )
    with pytest.raises(ValueError, match="Architecture search must remain disabled"):
        validate_open_ecg_protocol(path)


def test_label_budgets_cannot_drift(tmp_path: Path) -> None:
    path = _mutated_protocol(
        tmp_path,
        lambda payload: payload["phase1_if_phase0_passes"].update(
            {"target_label_budgets": [0, 100, 500]}
        ),
    )
    with pytest.raises(ValueError, match="label budgets must remain prespecified"):
        validate_open_ecg_protocol(path)


def test_crosswalk_requirement_cannot_be_disabled(tmp_path: Path) -> None:
    path = _mutated_protocol(
        tmp_path,
        lambda payload: payload["sources"]["development"]["ptb_xl"][
            "challenge_to_original_crosswalk"
        ].update({"required": False}),
    )
    with pytest.raises(ValueError, match="Every Challenge/PTB-XL pair"):
        validate_open_ecg_protocol(path)


def test_optimization_and_calibration_folds_cannot_be_merged(tmp_path: Path) -> None:
    path = _mutated_protocol(
        tmp_path,
        lambda payload: payload["internal_validation"].update(
            {"optimization_validation_fold": 9}
        ),
    )
    with pytest.raises(ValueError, match="fold 8 must remain optimization-only"):
        validate_open_ecg_protocol(path)


def test_primary_filtering_cannot_be_enabled_without_amendment(tmp_path: Path) -> None:
    path = _mutated_protocol(
        tmp_path,
        lambda payload: payload["signal_contract"].update(
            {"primary_filtering": "bandpass_0.5_40_hz"}
        ),
    )
    with pytest.raises(ValueError, match="filtering must remain disabled"):
        validate_open_ecg_protocol(path)
