"""Locked Phase 0 execution utilities for the open-data TRUST-ECG study.

This module executes only the prespecified statistical roles. It never uses the external recovery
pool in Phase 0 and never selects a model from external performance. The Logistic Regression path
is a low-capacity reference; only the fixed ResNet path is eligible for the primary Phase 0 gate.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trust_icu.ecg_baseline import (
    BinaryProbabilityMetrics,
    PairCertification,
    apply_platt_calibrators,
    certify_label_domain_pair,
    evaluate_binary_probabilities,
    extract_handcrafted_features,
    fit_logistic_reference,
    fit_platt_calibrators,
    logistic_decision_scores,
)
from trust_icu.ecg_data import EXPECTED_LEADS
from trust_icu.ecg_index import EcgIndexRow, load_and_verify_index_audit, model_index_sha256
from trust_icu.ecg_manifest import load_and_verify_label_manifest
from trust_icu.ecg_protocol import load_open_ecg_protocol, validate_open_ecg_protocol
from trust_icu.ecg_signal import (
    NormalizationStats,
    load_mat_digital_signal,
    parse_signal_header,
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
_EXTERNAL_SOURCES = ("georgia", "cpsc_2018", "cpsc_2018_extra")


@dataclass(frozen=True)
class MultiLabelMetrics:
    n: int
    label_count: int
    macro_pr_auc: float
    macro_roc_auc: float
    macro_brier: float
    per_label: dict[str, BinaryProbabilityMetrics]


@dataclass(frozen=True)
class EcgPhase0Report:
    report_version: str
    study: str
    model_name: str
    model_role: str
    primary_gate_eligible: bool
    protocol_version: str
    protocol_sha256: str
    model_index_audit_sha256: str
    model_index_sha256: str
    label_manifest_sha256: str
    model_sha256: str
    label_codes: tuple[str, ...]
    role_rows: dict[str, int]
    internal_test: MultiLabelMetrics
    external_certification: dict[str, dict[str, PairCertification]]
    external_recovery_pool_used: bool
    calibration_fit_role: str
    optimization_role_used: bool
    report_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_hash(payload: dict[str, Any], key: str) -> str:
    material = dict(payload)
    material[key] = ""
    encoded = json.dumps(
        _json_ready(material),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    """Load the local record index and verify it against its aggregate audit."""

    audit = load_and_verify_index_audit(index_audit_path)
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
            "relative_mat_path",
            "role",
            "strat_fold",
            *label_columns,
        }
        if set(reader.fieldnames or []) != expected:
            raise ValueError("ECG model-index columns do not match the locked audit labels.")
        for raw in reader:
            source = str(raw["source"])
            record_id = str(raw["record_id"])
            role = str(raw["role"])
            if source not in _ALLOWED_SOURCES:
                raise ValueError(f"Unexpected ECG source: {source}")
            if role not in _ALLOWED_ROLES:
                raise ValueError(f"Unexpected ECG statistical role: {role}")
            key = (source, record_id)
            if key in seen:
                raise ValueError(f"Duplicate source/record pair in ECG model index: {key}")
            seen.add(key)
            header_path = str(raw["relative_header_path"])
            mat_path = str(raw["relative_mat_path"])
            _safe_relative_path(header_path)
            _safe_relative_path(mat_path)
            fold_text = str(raw["strat_fold"]).strip()
            fold = None if fold_text == "" else int(fold_text)
            if source == "ptb-xl":
                if fold is None or role != _expected_ptb_role(fold):
                    raise ValueError(f"PTB-XL fold/role mismatch for {record_id}")
            elif fold is not None or role not in {"external_certification", "external_recovery_pool"}:
                raise ValueError(f"External source has invalid fold/role assignment: {key}")
            labels = tuple(int(raw[column]) for column in label_columns)
            if any(value not in (0, 1) for value in labels):
                raise ValueError(f"Non-binary locked label vector for {key}")
            rows.append(
                EcgIndexRow(
                    source=source,
                    record_id=record_id,
                    relative_header_path=header_path,
                    relative_mat_path=mat_path,
                    role=role,  # type: ignore[arg-type]
                    strat_fold=fold,
                    labels=labels,
                )
            )

    if len(rows) != int(audit["total_rows"]):
        raise ValueError("ECG model-index row count does not match its audit.")
    observed_hash = model_index_sha256(rows, label_codes)
    if observed_hash != str(audit["index_sha256"]):
        raise ValueError("ECG model-index SHA-256 verification failed.")

    observed_sources = dict(sorted(Counter(row.source for row in rows).items()))
    observed_roles = dict(sorted(Counter(row.role for row in rows).items()))
    expected_sources = {
        str(key): int(value) for key, value in dict(audit["source_rows"]).items()
    }
    expected_roles = {str(key): int(value) for key, value in dict(audit["role_rows"]).items()}
    if observed_sources != dict(sorted(expected_sources.items())):
        raise ValueError("ECG model-index source counts do not match its audit.")
    if observed_roles != dict(sorted(expected_roles.items())):
        raise ValueError("ECG model-index role counts do not match its audit.")

    recovery_rows = [row for row in rows if row.role == "external_recovery_pool"]
    certification_rows = [row for row in rows if row.role == "external_certification"]
    if not recovery_rows or not certification_rows:
        raise RuntimeError("Independent external certification/recovery partitions are required.")
    return rows, audit


def load_and_verify_normalization_stats(path: str | Path) -> NormalizationStats:
    """Verify the training-only normalization artifact used by the fixed ResNet."""

    stats_path = Path(path).expanduser().resolve()
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ECG normalization statistics must be a JSON object.")
    observed = str(payload.get("stats_sha256", ""))
    expected = _canonical_hash(payload, "stats_sha256")
    if not observed or observed != expected:
        raise ValueError("ECG normalization-statistics SHA-256 verification failed.")
    leads = tuple(str(value) for value in payload.get("leads", []))
    fit_folds = tuple(int(value) for value in payload.get("fit_folds", []))
    if leads != tuple(EXPECTED_LEADS):
        raise ValueError("ECG normalization lead order does not match the locked signal contract.")
    if fit_folds != (1, 2, 3, 4, 5, 6, 7):
        raise ValueError("ECG normalization must be fit only on PTB-XL folds 1-7.")
    means = tuple(float(value) for value in payload.get("means_mv", []))
    stds = tuple(float(value) for value in payload.get("stds_mv", []))
    counts = tuple(int(value) for value in payload.get("valid_sample_counts", []))
    if not (len(means) == len(stds) == len(counts) == 12):
        raise ValueError("ECG normalization statistics must contain exactly 12 leads.")
    if any(not np.isfinite(value) for value in (*means, *stds)) or any(value <= 0 for value in stds):
        raise ValueError("ECG normalization statistics contain invalid values.")
    return NormalizationStats(
        means_mv=means,
        stds_mv=stds,
        valid_sample_counts=counts,
        leads=leads,
        fit_folds=fit_folds,
        stats_sha256=observed,
    )


def _load_standardized_record(data_root: Path, row: EcgIndexRow):
    header_path = (data_root / _safe_relative_path(row.relative_header_path)).resolve()
    mat_path = (data_root / _safe_relative_path(row.relative_mat_path)).resolve()
    if data_root not in header_path.parents or data_root not in mat_path.parents:
        raise ValueError("ECG model index escaped the declared training root.")
    record_id, sampling_rate, sample_count, specs = parse_signal_header(header_path)
    if record_id != row.record_id:
        raise ValueError(f"Header record ID mismatch for {row.source}/{row.record_id}")
    digital = load_mat_digital_signal(mat_path)
    if digital.shape != (12, sample_count):
        raise ValueError(f"Waveform dimensions disagree with header for {row.source}/{row.record_id}")
    return standardize_signal(
        digital,
        specs,
        source_sampling_rate_hz=sampling_rate,
        target_sampling_rate_hz=500,
        target_duration_seconds=10,
    )


def _metrics_for_multilabel(
    y: np.ndarray,
    probabilities: np.ndarray,
    label_codes: tuple[str, ...],
) -> MultiLabelMetrics:
    targets = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 2 or probs.shape != targets.shape or targets.shape[1] != len(label_codes):
        raise ValueError("Multi-label metric inputs do not align with locked labels.")
    per_label: dict[str, BinaryProbabilityMetrics] = {}
    for index, code in enumerate(label_codes):
        per_label[code] = evaluate_binary_probabilities(targets[:, index], probs[:, index])
    return MultiLabelMetrics(
        n=int(targets.shape[0]),
        label_count=len(label_codes),
        macro_pr_auc=float(np.mean([value.pr_auc for value in per_label.values()])),
        macro_roc_auc=float(np.mean([value.roc_auc for value in per_label.values()])),
        macro_brier=float(np.mean([value.brier for value in per_label.values()])),
        per_label=per_label,
    )


def _external_certification(
    *,
    rows: list[EcgIndexRow],
    probabilities: np.ndarray,
    label_codes: tuple[str, ...],
) -> dict[str, dict[str, PairCertification]]:
    if probabilities.shape != (len(rows), len(label_codes)):
        raise ValueError("External certification probabilities do not align with index rows.")
    output: dict[str, dict[str, PairCertification]] = {}
    for source in _EXTERNAL_SOURCES:
        indices = [index for index, row in enumerate(rows) if row.source == source]
        if not indices:
            raise RuntimeError(f"External certification source is empty: {source}")
        y_source = np.asarray([rows[index].labels for index in indices], dtype=np.int64)
        p_source = probabilities[indices]
        output[source] = {
            code: certify_label_domain_pair(y_source[:, label_index], p_source[:, label_index])
            for label_index, code in enumerate(label_codes)
        }
    return output


def _logistic_model_hash(model: Any, calibrator: Any) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(model.scaler.mean_, dtype=np.float64).tobytes())
    digest.update(np.asarray(model.scaler.scale_, dtype=np.float64).tobytes())
    for estimator in model.estimators:
        digest.update(np.asarray(estimator.coef_, dtype=np.float64).tobytes())
        digest.update(np.asarray(estimator.intercept_, dtype=np.float64).tobytes())
    for estimator in calibrator.estimators:
        digest.update(np.asarray(estimator.coef_, dtype=np.float64).tobytes())
        digest.update(np.asarray(estimator.intercept_, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _finalize_report(payload: dict[str, Any]) -> EcgPhase0Report:
    payload["report_sha256"] = _canonical_hash(payload, "report_sha256")
    return EcgPhase0Report(
        report_version=str(payload["report_version"]),
        study=str(payload["study"]),
        model_name=str(payload["model_name"]),
        model_role=str(payload["model_role"]),
        primary_gate_eligible=bool(payload["primary_gate_eligible"]),
        protocol_version=str(payload["protocol_version"]),
        protocol_sha256=str(payload["protocol_sha256"]),
        model_index_audit_sha256=str(payload["model_index_audit_sha256"]),
        model_index_sha256=str(payload["model_index_sha256"]),
        label_manifest_sha256=str(payload["label_manifest_sha256"]),
        model_sha256=str(payload["model_sha256"]),
        label_codes=tuple(payload["label_codes"]),
        role_rows={str(key): int(value) for key, value in payload["role_rows"].items()},
        internal_test=payload["internal_test"],
        external_certification=payload["external_certification"],
        external_recovery_pool_used=bool(payload["external_recovery_pool_used"]),
        calibration_fit_role=str(payload["calibration_fit_role"]),
        optimization_role_used=bool(payload["optimization_role_used"]),
        report_sha256=str(payload["report_sha256"]),
    )


def execute_logistic_reference_phase0(
    *,
    challenge_training_root: str | Path,
    index_csv: str | Path,
    index_audit_path: str | Path,
    label_manifest_path: str | Path,
    protocol_path: str | Path,
) -> EcgPhase0Report:
    """Execute the fixed handcrafted Logistic Regression reference without touching recovery data."""

    protocol_file = Path(protocol_path).expanduser().resolve()
    protocol_summary = validate_open_ecg_protocol(protocol_file)
    protocol = load_open_ecg_protocol(protocol_file)
    manifest = load_and_verify_label_manifest(label_manifest_path)
    rows, audit = load_and_verify_model_index(index_csv=index_csv, index_audit_path=index_audit_path)
    if str(audit["label_manifest_sha256"]) != str(manifest["manifest_sha256"]):
        raise ValueError("ECG model-index audit and label manifest are from different study states.")
    label_codes = tuple(str(code) for code in audit["label_codes"])
    if tuple(str(item["canonical_code"]) for item in manifest["labels"]) != label_codes:
        raise ValueError("Locked label order differs between model index and label manifest.")

    root = Path(challenge_training_root).expanduser().resolve()
    allowed_execution_roles = {"model_fit", "calibration", "internal_test", "external_certification"}
    selected_rows = [row for row in rows if row.role in allowed_execution_roles]
    if any(row.role == "external_recovery_pool" for row in selected_rows):
        raise RuntimeError("External recovery records are prohibited in Phase 0 execution.")

    features_by_role: dict[str, list[np.ndarray]] = {role: [] for role in allowed_execution_roles}
    labels_by_role: dict[str, list[tuple[int, ...]]] = {role: [] for role in allowed_execution_roles}
    external_rows: list[EcgIndexRow] = []
    for row in selected_rows:
        standardized = _load_standardized_record(root, row)
        features = extract_handcrafted_features(standardized.waveform_mv, standardized.valid_mask)
        features_by_role[row.role].append(features)
        labels_by_role[row.role].append(row.labels)
        if row.role == "external_certification":
            external_rows.append(row)

    def matrix(role: str) -> tuple[np.ndarray, np.ndarray]:
        if not features_by_role[role]:
            raise RuntimeError(f"ECG execution role is empty: {role}")
        return (
            np.vstack(features_by_role[role]).astype(np.float64, copy=False),
            np.asarray(labels_by_role[role], dtype=np.int64),
        )

    X_fit, y_fit = matrix("model_fit")
    X_cal, y_cal = matrix("calibration")
    X_test, y_test = matrix("internal_test")
    X_external, _ = matrix("external_certification")

    model = fit_logistic_reference(X_fit, y_fit, label_codes=label_codes)
    cal_scores = logistic_decision_scores(model, X_cal)
    calibrator = fit_platt_calibrators(cal_scores, y_cal, label_codes=label_codes)
    test_probabilities = apply_platt_calibrators(calibrator, logistic_decision_scores(model, X_test))
    external_probabilities = apply_platt_calibrators(
        calibrator,
        logistic_decision_scores(model, X_external),
    )

    internal_metrics = _metrics_for_multilabel(y_test, test_probabilities, label_codes)
    external = _external_certification(
        rows=external_rows,
        probabilities=external_probabilities,
        label_codes=label_codes,
    )
    role_counts = Counter(row.role for row in rows)
    payload: dict[str, Any] = {
        "report_version": "0.1.0",
        "study": "TRUST-ECG",
        "model_name": "logistic_regression_handcrafted",
        "model_role": "low_capacity_reference_only",
        "primary_gate_eligible": False,
        "protocol_version": str(protocol_summary["version"]),
        "protocol_sha256": _sha256_file(protocol_file),
        "model_index_audit_sha256": str(audit["audit_sha256"]),
        "model_index_sha256": str(audit["index_sha256"]),
        "label_manifest_sha256": str(manifest["manifest_sha256"]),
        "model_sha256": _logistic_model_hash(model, calibrator),
        "label_codes": label_codes,
        "role_rows": dict(sorted(role_counts.items())),
        "internal_test": internal_metrics,
        "external_certification": external,
        "external_recovery_pool_used": False,
        "calibration_fit_role": "ptb_xl_fold_9_only",
        "optimization_role_used": False,
        "report_sha256": "",
    }
    if protocol["phase0_models"]["primary_model"] == "logistic_regression_handcrafted":
        raise RuntimeError("Protocol drift: Logistic Regression must remain reference-only.")
    return _finalize_report(payload)


def write_phase0_report(report: EcgPhase0Report, output: str | Path) -> None:
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    expected = _canonical_hash(payload, "report_sha256")
    if expected != report.report_sha256:
        raise ValueError("Refusing to write a Phase 0 ECG report with an invalid embedded hash.")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_phase0_dry_run_plan(protocol_path: str | Path) -> dict[str, Any]:
    summary = validate_open_ecg_protocol(protocol_path)
    return {
        "study": "TRUST-ECG",
        "stage": "locked_phase0_model_execution",
        "protocol_version": summary["version"],
        "primary_model": summary["primary_model"],
        "reference_model": "logistic_regression_handcrafted",
        "development_roles": {
            "model_fit": "ptb_xl_folds_1_to_7",
            "optimization": "ptb_xl_fold_8_primary_resnet_only",
            "calibration": "ptb_xl_fold_9_only",
            "internal_test": "ptb_xl_fold_10_only",
        },
        "external_phase0_role": "external_certification_60_percent_only",
        "external_recovery_pool_access": "prohibited",
        "primary_gate_model": "resnet1d_fixed_only",
        "outputs": [
            "aggregate model execution report with SHA-256",
            "no record-level predictions",
            "no external recovery records",
        ],
    }
