from pathlib import Path

import pytest
import yaml

from trust_icu.adapter_manifest import (
    load_and_validate_adapter_manifest,
    validate_adapter_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "schemas" / "source_adapter_manifest.yaml"


def test_repository_manifest_is_complete() -> None:
    report = load_and_validate_adapter_manifest(MANIFEST, ROOT)
    assert report.valid is True
    reports = {item.dataset: item for item in report.datasets}
    assert reports["mimic_iv_3_1"].ready_for_credentialed_execution is True
    assert reports["eicu_crd_2_0"].ready_for_credentialed_execution is False
    assert reports["eicu_crd_2_0"].local_mappings_required is True


def test_dataset_identifier_mismatch_is_rejected() -> None:
    raw = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    raw["datasets"]["mimic_iv_3_1"]["output_dataset_id"] = "MIMIC-IV-3.1"
    with pytest.raises(ValueError, match="output_dataset_id"):
        validate_adapter_manifest(raw, ROOT)


def test_unsafe_relative_path_is_rejected() -> None:
    raw = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    raw["datasets"]["mimic_iv_3_1"]["execution_order"]["events"] = "../events.sql"
    with pytest.raises(ValueError, match="safe path"):
        validate_adapter_manifest(raw, ROOT)


def test_missing_declared_contract_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    with pytest.raises(FileNotFoundError, match="Declared contract"):
        validate_adapter_manifest(raw, tmp_path)


def test_missing_adapter_files_return_structured_invalid_report(tmp_path: Path) -> None:
    raw = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for key in ("canonical_contract", "feature_contract", "outcome_contract"):
        source = ROOT / raw[key]
        target = tmp_path / raw[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    report = validate_adapter_manifest(raw, tmp_path)
    assert report.valid is False
    assert all(dataset.files_present is False for dataset in report.datasets)
    assert any(dataset.missing_paths for dataset in report.datasets)
