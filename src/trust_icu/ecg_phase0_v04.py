"""Protocol-v0.4 execution adapter for TRUST-ECG Phase 0.

The mature Phase-0 statistical implementation is reused, but all record-index parsing and waveform
loading are replaced by source-aware v0.4 functions. Original PTB-XL is read from WFDB ``.dat``
records; Georgia/CPSC external records remain Challenge ``.mat`` inputs. This avoids duplicating
calibration, metrics, certification, and report-hashing logic while removing the obsolete
Challenge-PTB-XL assumption from model execution.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from trust_icu import ecg_phase0 as _legacy
from trust_icu.ecg_index import EcgIndexRow, load_and_verify_index_audit, model_index_sha256
from trust_icu.ecg_signal import (
    load_mat_digital_signal,
    load_wfdb_physical_signal,
    parse_signal_header,
    standardize_physical_signal,
    standardize_signal,
)

_ALLOWED_ROLES = {
    "model_fit",
    "optimization_validation",
    "calibration",
    "internal_test",
    "external_certification",
    "external_recovery_pool",
}
_ALLOWED_SOURCES = {"ptb-xl", "georgia", "cpsc_2018", "cpsc_2018_extra"}
_EXTERNAL_SOURCES = {"georgia", "cpsc_2018", "cpsc_2018_extra"}

EcgPhase0Report = _legacy.EcgPhase0Report
MultiLabelMetrics = _legacy.MultiLabelMetrics
load_and_verify_normalization_stats = _legacy.load_and_verify_normalization_stats
write_phase0_report = _legacy.write_phase0_report


def _safe_relative_path(value: str) -> Path:
    if "\\" in value:
        raise ValueError(f"ECG model-index paths must use POSIX separators: {value!r}")
    path = Path(value)
    if path.is_absolute() or not value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe relative path in ECG model index: {value!r}")
    return path


def _expected_ptb_role(fold: int) -> str:
    if fold in range(1, 8):
        return "model_fit"
    if fold == 8:
        return "optimization_validation"
    if fold == 9:
        return "calibration"
    if fold == 10:
        return "internal_test"
    raise ValueError(f"PTB-XL fold must lie in 1..10, found {fold}")


def load_and_verify_model_index(
    *,
    index_csv: str | Path,
    index_audit_path: str | Path,
) -> tuple[list[EcgIndexRow], dict[str, Any]]:
    """Load the v0.4 source-aware local index and verify all aggregate invariants."""

    audit = load_and_verify_index_audit(index_audit_path)
    if str(audit.get("audit_version")) != "0.2.0":
        raise ValueError("TRUST-ECG v0.4 requires model-index audit version 0.2.0.")
    if audit.get("development_source") != "original_ptbxl_v1_0_1":
        raise ValueError("Model-index development source is not original PTB-XL v1.0.1.")
    if audit.get("challenge_ptbxl_model_input") is not False:
        raise ValueError("Model index illegally enables Challenge PTB-XL model input.")

    label_codes = tuple(str(code) for code in audit.get("label_codes", []))
    if not label_codes:
        raise ValueError("ECG model-index audit contains no locked labels.")
    label_columns = tuple(f"label_{code}" for code in label_codes)

    path = Path(index_csv).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ECG model index not found: {path}")
    rows: list[EcgIndexRow] = []
    seen: set[tuple[str, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {
            "source",
            "record_id",
            "relative_header_path",
            "relative_waveform_path",
            "waveform_format",
            "role",
            "strat_fold",
            *label_columns,
        }
        if set(reader.fieldnames or []) != expected:
            raise ValueError("ECG v0.4 model-index columns do not match the locked audit labels.")
        for raw in reader:
            source = str(raw["source"])
            record_id = str(raw["record_id"])
            role = str(raw["role"])
            waveform_format = str(raw["waveform_format"])
            if source not in _ALLOWED_SOURCES:
                raise ValueError(f"Unexpected ECG source: {source}")
            if role not in _ALLOWED_ROLES:
                raise ValueError(f"Unexpected ECG statistical role: {role}")
            key = (source, record_id)
            if key in seen:
                raise ValueError(f"Duplicate source/record pair in ECG model index: {key}")
            seen.add(key)
            header_path = str(raw["relative_header_path"])
            waveform_path = str(raw["relative_waveform_path"])
            _safe_relative_path(header_path)
            _safe_relative_path(waveform_path)
            fold_text = str(raw["strat_fold"]).strip()
            fold = None if fold_text == "" else int(fold_text)
            if source == "ptb-xl":
                if fold is None or role != _expected_ptb_role(fold):
                    raise ValueError(f"PTB-XL fold/role mismatch for {record_id}")
                if waveform_format != "wfdb_dat_original_ptbxl_v1_0_1":
                    raise ValueError("Original PTB-XL index row has an unexpected waveform format.")
                if not waveform_path.endswith(".dat"):
                    raise ValueError("Original PTB-XL waveform path must end in .dat.")
            else:
                if fold is not None or role not in {
                    "external_certification",
                    "external_recovery_pool",
                }:
                    raise ValueError(f"External source has invalid fold/role assignment: {key}")
                if waveform_format != "challenge_mat_v4" or not waveform_path.endswith(".mat"):
                    raise ValueError("Challenge external index row has an unexpected waveform format.")
            labels = tuple(int(raw[column]) for column in label_columns)
            if any(value not in (0, 1) for value in labels):
                raise ValueError(f"Non-binary locked label vector for {key}")
            rows.append(
                EcgIndexRow(
                    source=source,
                    record_id=record_id,
                    relative_header_path=header_path,
                    relative_waveform_path=waveform_path,
                    waveform_format=waveform_format,
                    role=role,  # type: ignore[arg-type]
                    strat_fold=fold,
                    labels=labels,
                )
            )

    if len(rows) != int(audit["total_rows"]):
        raise ValueError("ECG model-index row count does not match its audit.")
    if model_index_sha256(rows, label_codes) != str(audit["index_sha256"]):
        raise ValueError("ECG model-index SHA-256 verification failed.")
    observed_sources = dict(sorted(Counter(row.source for row in rows).items()))
    observed_roles = dict(sorted(Counter(row.role for row in rows).items()))
    expected_sources = {str(key): int(value) for key, value in dict(audit["source_rows"]).items()}
    expected_roles = {str(key): int(value) for key, value in dict(audit["role_rows"]).items()}
    if observed_sources != dict(sorted(expected_sources.items())):
        raise ValueError("ECG model-index source counts do not match its audit.")
    if observed_roles != dict(sorted(expected_roles.items())):
        raise ValueError("ECG model-index role counts do not match its audit.")
    if not any(row.role == "external_recovery_pool" for row in rows):
        raise RuntimeError("Independent external recovery pool is required.")
    if not any(row.role == "external_certification" for row in rows):
        raise RuntimeError("Independent external certification partition is required.")
    return rows, audit


def load_standardized_record(data_root: Path, row: EcgIndexRow):
    """Load one record according to its frozen source-specific waveform format."""

    root = data_root.expanduser().resolve()
    header_path = (root / _safe_relative_path(row.relative_header_path)).resolve()
    waveform_path = (root / _safe_relative_path(row.relative_waveform_path)).resolve()
    if root not in header_path.parents or root not in waveform_path.parents:
        raise ValueError("ECG model index escaped the declared primary data root.")
    if not header_path.is_file() or not waveform_path.is_file():
        raise FileNotFoundError(f"Indexed ECG record is incomplete: {row.source}/{row.record_id}")

    if row.source == "ptb-xl":
        if row.waveform_format != "wfdb_dat_original_ptbxl_v1_0_1":
            raise ValueError("PTB-XL row does not use the locked WFDB waveform format.")
        header_record_id, header_rate, header_samples, _ = parse_signal_header(header_path)
        if header_record_id != row.record_id:
            raise ValueError(f"Header record ID mismatch for {row.source}/{row.record_id}")
        physical, leads, wfdb_rate = load_wfdb_physical_signal(waveform_path.with_suffix(""))
        if abs(wfdb_rate - header_rate) > 1e-9 or physical.shape[1] != header_samples:
            raise ValueError(f"WFDB physical signal disagrees with header for {row.record_id}")
        return standardize_physical_signal(
            physical,
            leads,
            source_sampling_rate_hz=wfdb_rate,
            target_sampling_rate_hz=500,
            target_duration_seconds=10,
        )

    if row.source not in _EXTERNAL_SOURCES or row.waveform_format != "challenge_mat_v4":
        raise ValueError(f"Unsupported ECG source/format pair: {row.source}/{row.waveform_format}")
    record_id, sampling_rate, sample_count, specs = parse_signal_header(header_path)
    if record_id != row.record_id:
        raise ValueError(f"Header record ID mismatch for {row.source}/{row.record_id}")
    digital = load_mat_digital_signal(waveform_path)
    if digital.shape != (12, sample_count):
        raise ValueError(f"Waveform dimensions disagree with header for {row.source}/{row.record_id}")
    return standardize_signal(
        digital,
        specs,
        source_sampling_rate_hz=sampling_rate,
        target_sampling_rate_hz=500,
        target_duration_seconds=10,
    )


def _install_legacy_adapter() -> None:
    """Inject only the v0.4 index/loader boundary into the locked statistical implementation."""

    _legacy.load_and_verify_model_index = load_and_verify_model_index
    _legacy._load_standardized_record = load_standardized_record


def execute_logistic_reference_phase0(
    *,
    primary_data_root: str | Path,
    index_csv: str | Path,
    index_audit_path: str | Path,
    label_manifest_path: str | Path,
    protocol_path: str | Path,
) -> EcgPhase0Report:
    """Run the fixed Logistic reference using the v0.4 source-aware data boundary."""

    _install_legacy_adapter()
    return _legacy.execute_logistic_reference_phase0(
        challenge_training_root=primary_data_root,
        index_csv=index_csv,
        index_audit_path=index_audit_path,
        label_manifest_path=label_manifest_path,
        protocol_path=protocol_path,
    )


def build_phase0_dry_run_plan(protocol_path: str | Path) -> dict[str, Any]:
    plan = _legacy.build_phase0_dry_run_plan(protocol_path)
    plan.update(
        {
            "protocol_adapter": "v0.4_source_aware",
            "development_source": "original_ptbxl_v1_0_1",
            "challenge_ptbxl_model_input": False,
            "development_waveform_format": "wfdb_dat_original_ptbxl_v1_0_1",
            "external_waveform_format": "challenge_mat_v4",
        }
    )
    return plan
