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
    assert report["development_source"] == "ptb_xl"
    assert report["development_records"] == 21837
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
