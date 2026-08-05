"""Locked Phase 0 temporal and external baseline orchestration.

The module consumes only credentialed canonical exports that passed the source-adapter audit.
It never writes patient-level predictions, trained models, or identifiers to public artifacts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

from trust_icu.baseline import (
    BaselineMetrics,
    assert_matrix_is_leakage_safe,
    evaluate_probabilities,
    fit_catboost_baseline,
    fit_logistic_baseline,
)
from trust_icu.cohort import assign_task_labels, build_landmark_cohort
from trust_icu.config import StudyConfig, load_config
from trust_icu.features import FeatureContract, FeatureMatrixAudit, load_feature_contract
from trust_icu.outcomes import assert_task_training_allowed, load_outcome_contracts
from trust_icu.validation import DatasetAudit, GateDecision, evaluate_feasibility

_DATASETS = {"mimic_iv_3_1", "eicu_crd_2_0"}
_TASKS = (
    "invasive_mechanical_ventilation",
    "vasopressor_initiation",
    "renal_replacement_therapy",
)
_MODELS = ("logistic_regression", "catboost")
_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class VerifiedCanonicalRun:
    dataset: str
    run_dir: str
    report_sha256: str
    audit_sha256: str
    stays_path: str
    events_path: str
    observations_path: str
    stays_sha256: str
    events_sha256: str
    observations_sha256: str


@dataclass(frozen=True)
class TemporalSplitReport:
    train_rows: int
    calibration_rows: int
    temporal_test_rows: int
    purged_calibration_rows: int
    purged_temporal_test_rows: int
    train_end_time_utc: str
    calibration_end_time_utc: str
    patient_overlap_after_purge: bool


@dataclass(frozen=True)
class BootstrapInterval:
    metric: str
    estimate: float
    lower: float
    upper: float
    successful_iterations: int
    requested_iterations: int


@dataclass(frozen=True)
class HospitalRobustness:
    total_hospitals: int
    eligible_metric_hospitals: int
    minimum_rows: int
    minimum_events: int
    median_pr_auc: float | None
    tenth_percentile_pr_auc: float | None
    worst_pr_auc: float | None
    median_brier: float | None
    worst_brier: float | None


@dataclass(frozen=True)
class MissingnessShift:
    feature_count: int
    median_absolute_difference: float
    maximum_absolute_difference: float
    features_over_0_10: int
    features_over_0_20: int
    largest_differences: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ModelEvaluation:
    model: str
    calibration_metrics: BaselineMetrics
    temporal_metrics_raw: BaselineMetrics
    temporal_metrics_calibrated: BaselineMetrics
    external_metrics_raw: BaselineMetrics
    external_metrics_calibrated: BaselineMetrics
    selected_from_development_only: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskPhase0Report:
    task: str
    development_rows: int
    development_events: int
    external_rows: int
    external_events: int
    split: TemporalSplitReport
    selected_model: str
    selection_rule: str
    models: tuple[ModelEvaluation, ...]
    external_cluster_bootstrap: tuple[BootstrapInterval, ...]
    hospital_robustness: HospitalRobustness
    missingness_shift: MissingnessShift
    feasibility_decision: GateDecision

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feasibility_decision"] = self.feasibility_decision.to_dict()
        return payload


@dataclass(frozen=True)
class Phase0BaselineReport:
    study: str
    phase: str
    config_sha256: str
    feature_contract_sha256: str
    outcome_contract_sha256: str
    mimic_input: VerifiedCanonicalRun
    eicu_input: VerifiedCanonicalRun
    feature_audits: dict[str, dict[str, Any]]
    tasks: tuple[TaskPhase0Report, ...]
    all_tasks_continue: bool
    report_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: dict[str, Any], hash_field: str) -> str:
    material = copy.deepcopy(payload)
    material[hash_field] = ""
    return _sha256_bytes(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _json_mapping(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return raw


def _yaml_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return raw


def verify_credentialed_run(run_dir: str | Path, dataset: str) -> VerifiedCanonicalRun:
    """Verify the hashed credentialed report and every canonical export before loading rows."""

    if dataset not in _DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    root = Path(run_dir).expanduser().resolve()
    report_path = root / "credentialed_run_report.json"
    audit_path = root / "canonical_audit.json"
    if not report_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError(f"Credentialed report or audit is missing from {root}.")

    report = _json_mapping(report_path)
    audit = _json_mapping(audit_path)
    if report.get("dataset") != dataset:
        raise ValueError(f"Credentialed report dataset must equal {dataset}.")
    observed_report_hash = str(report.get("report_sha256", ""))
    expected_report_hash = _canonical_hash(report, "report_sha256")
    if observed_report_hash != expected_report_hash:
        raise ValueError("Credentialed report SHA-256 verification failed.")
    if report.get("audit") != audit:
        raise ValueError("Credentialed report audit does not match canonical_audit.json.")
    if audit.get("dataset") != dataset or audit.get("ready_for_cohort_build") is not True:
        raise RuntimeError("Canonical audit is not approved for cohort construction.")
    if audit.get("critical_failures"):
        raise RuntimeError("Canonical audit contains critical failures.")

    standard_paths = {
        "stays": root / f"{dataset}_stays.csv.gz",
        "events": root / f"{dataset}_events.csv.gz",
        "observations": root / f"{dataset}_observations.csv.gz",
    }
    exports = report.get("exports")
    if not isinstance(exports, list) or len(exports) != 3:
        raise ValueError("Credentialed report must contain three export artifacts.")
    by_kind: dict[str, dict[str, Any]] = {}
    for item in exports:
        if not isinstance(item, dict):
            raise ValueError("Each export artifact must be an object.")
        name = str(item.get("name", ""))
        matches = [kind for kind in standard_paths if name.endswith(f"_{kind}")]
        if len(matches) != 1:
            raise ValueError(f"Unrecognized export relation name: {name!r}.")
        by_kind[matches[0]] = item
    if set(by_kind) != set(standard_paths):
        raise ValueError("Credentialed report does not define stays, events, and observations.")

    verified_hashes: dict[str, str] = {}
    for kind, path in standard_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Canonical {kind} export not found: {path}")
        expected = str(by_kind[kind].get("sha256", ""))
        observed = _sha256_file(path)
        if observed != expected:
            raise ValueError(f"Canonical {kind} export SHA-256 verification failed.")
        if path.stat().st_size != int(by_kind[kind].get("bytes", -1)):
            raise ValueError(f"Canonical {kind} export byte count does not match report.")
        verified_hashes[kind] = observed

    return VerifiedCanonicalRun(
        dataset=dataset,
        run_dir=str(root),
        report_sha256=observed_report_hash,
        audit_sha256=_sha256_file(audit_path),
        stays_path=str(standard_paths["stays"]),
        events_path=str(standard_paths["events"]),
        observations_path=str(standard_paths["observations"]),
        stays_sha256=verified_hashes["stays"],
        events_sha256=verified_hashes["events"],
        observations_sha256=verified_hashes["observations"],
    )


def _sql_literal(path: str | Path) -> str:
    return "'" + str(Path(path).resolve()).replace("'", "''") + "'"


def _safe_feature_name(value: str) -> str:
    if not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"Unsafe canonical feature name: {value!r}")
    return value


def build_feature_matrix_from_extract(
    stays_path: str | Path,
    observations_path: str | Path,
    contract: FeatureContract,
) -> tuple[pd.DataFrame, FeatureMatrixAudit]:
    """Build a wide feature matrix in DuckDB without loading raw observations into pandas."""

    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "Scalable feature extraction requires DuckDB. Install trust-icu with the analytics extra."
        ) from exc

    stays_literal = _sql_literal(stays_path)
    observations_literal = _sql_literal(observations_path)
    variables = tuple(_safe_feature_name(variable.name) for variable in contract.variables)
    bounds = contract.plausible_bounds
    contract_frame = pd.DataFrame(
        {
            "variable": variables,
            "plausible_min": [bounds[name][0] for name in variables],
            "plausible_max": [bounds[name][1] for name in variables],
        }
    )

    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("feature_contract", contract_frame)
        connection.execute(
            f"""
            CREATE VIEW canonical_stays AS
            SELECT
                CAST(stay_id AS VARCHAR) AS stay_id,
                TRY_CAST(age AS DOUBLE) AS age,
                CAST(sex AS VARCHAR) AS sex,
                TRY_CAST(icu_admit_time AS TIMESTAMPTZ) AS icu_admit_time
            FROM read_csv_auto({stays_literal}, header=true, all_varchar=true, sample_size=-1)
            """
        )
        connection.execute(
            f"""
            CREATE VIEW canonical_observations AS
            SELECT
                CAST(stay_id AS VARCHAR) AS stay_id,
                CAST(variable AS VARCHAR) AS variable,
                TRY_CAST(event_time AS TIMESTAMPTZ) AS event_time,
                TRY_CAST(value AS DOUBLE) AS value
            FROM read_csv_auto(
                {observations_literal}, header=true, all_varchar=true, sample_size=-1
            )
            """
        )

        audit_row = connection.execute(
            """
            WITH linked AS (
                SELECT
                    o.*,
                    s.icu_admit_time,
                    fc.variable AS known_variable,
                    fc.plausible_min,
                    fc.plausible_max
                FROM canonical_observations AS o
                LEFT JOIN canonical_stays AS s USING (stay_id)
                LEFT JOIN feature_contract AS fc USING (variable)
            ),
            duplicate_rows AS (
                SELECT COALESCE(SUM(row_count), 0) AS duplicate_count
                FROM (
                    SELECT COUNT(*) AS row_count
                    FROM canonical_observations
                    GROUP BY stay_id, variable, event_time
                    HAVING COUNT(*) > 1
                )
            )
            SELECT
                COUNT(*) AS received,
                COUNT(*) FILTER (
                    WHERE known_variable IS NOT NULL
                      AND event_time >= icu_admit_time
                      AND event_time < icu_admit_time + INTERVAL '6 hours'
                      AND value BETWEEN plausible_min AND plausible_max
                ) AS used,
                COUNT(*) FILTER (WHERE event_time < icu_admit_time) AS before_icu,
                COUNT(*) FILTER (
                    WHERE event_time >= icu_admit_time + INTERVAL '6 hours'
                ) AS at_or_after,
                COUNT(*) FILTER (
                    WHERE known_variable IS NOT NULL
                      AND value IS NOT NULL
                      AND NOT (value BETWEEN plausible_min AND plausible_max)
                ) AS out_of_range,
                (SELECT duplicate_count FROM duplicate_rows) AS duplicate_count
            FROM linked
            """
        ).fetchone()
        unknown_rows = connection.execute(
            """
            SELECT DISTINCT o.variable
            FROM canonical_observations AS o
            LEFT JOIN feature_contract AS fc USING (variable)
            WHERE fc.variable IS NULL
            ORDER BY o.variable
            """
        ).fetchall()
        unknown_variables = tuple(str(row[0]) for row in unknown_rows)

        expressions: list[str] = []
        for variable in variables:
            lower, upper = bounds[variable]
            condition = (
                f"o.variable = '{variable}' "
                "AND o.event_time >= s.icu_admit_time "
                "AND o.event_time < s.icu_admit_time + INTERVAL '6 hours' "
                f"AND o.value BETWEEN {float(lower)} AND {float(upper)}"
            )
            expressions.extend(
                [
                    f'arg_min(o.value, o.event_time) FILTER (WHERE {condition}) AS "{variable}__first"',
                    f'arg_max(o.value, o.event_time) FILTER (WHERE {condition}) AS "{variable}__last"',
                    f'min(o.value) FILTER (WHERE {condition}) AS "{variable}__min"',
                    f'max(o.value) FILTER (WHERE {condition}) AS "{variable}__max"',
                    f'avg(o.value) FILTER (WHERE {condition}) AS "{variable}__mean"',
                    (
                        "regr_slope(o.value, epoch(o.event_time - s.icu_admit_time) / 3600.0) "
                        f'FILTER (WHERE {condition}) AS "{variable}__slope_per_hour"'
                    ),
                    f'count(o.value) FILTER (WHERE {condition}) AS "{variable}__count"',
                    (
                        "epoch((s.icu_admit_time + INTERVAL '6 hours') - "
                        f'max(o.event_time) FILTER (WHERE {condition})) / 3600.0 '
                        f'AS "{variable}__hours_since_last"'
                    ),
                ]
            )
        select_features = ",\n                ".join(expressions)
        matrix = connection.execute(
            f"""
            SELECT
                s.stay_id,
                s.age,
                s.sex,
                {select_features}
            FROM canonical_stays AS s
            LEFT JOIN canonical_observations AS o USING (stay_id)
            GROUP BY s.stay_id, s.age, s.sex, s.icu_admit_time
            ORDER BY s.stay_id
            """
        ).fetch_df()
    finally:
        connection.close()

    for variable in variables:
        count_column = f"{variable}__count"
        matrix[count_column] = matrix[count_column].fillna(0).astype("int32")
        matrix[f"{variable}__missing"] = matrix[count_column].eq(0).astype("int8")

    audit = FeatureMatrixAudit(
        cohort_rows=len(matrix),
        observation_rows_received=int(audit_row[0]),
        observation_rows_used=int(audit_row[1]),
        rows_before_icu=int(audit_row[2]),
        rows_at_or_after_landmark=int(audit_row[3]),
        out_of_range_rows=int(audit_row[4]),
        unknown_variables=unknown_variables,
        duplicate_stay_variable_time_rows=int(audit_row[5]),
    )
    return matrix, audit


def _read_stays(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    for column in ("patient_id", "hospital_admission_id", "stay_id", "hospital_id"):
        if column in frame:
            frame[column] = frame[column].astype("string")
    return frame


def _read_events(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if "stay_id" in frame:
        frame["stay_id"] = frame["stay_id"].astype("string")
    return frame


def _validate_fractions(raw_config: dict[str, Any]) -> tuple[float, float, float]:
    validation = raw_config.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("validation configuration must be a mapping.")
    temporal = validation.get("temporal_split")
    if not isinstance(temporal, dict):
        raise ValueError("validation.temporal_split must be a mapping.")
    fractions = (
        float(temporal.get("train_fraction", 0)),
        float(temporal.get("calibration_fraction", 0)),
        float(temporal.get("test_fraction", 0)),
    )
    if any(value <= 0 for value in fractions) or not math.isclose(sum(fractions), 1.0):
        raise ValueError("Temporal split fractions must be positive and sum to one.")
    if temporal.get("patient_overlap_policy") != "purge_later_splits":
        raise ValueError("Temporal split must use purge_later_splits patient isolation.")
    return fractions


def temporal_patient_purged_split(
    metadata: pd.DataFrame,
    fractions: tuple[float, float, float],
) -> tuple[dict[str, pd.Index], TemporalSplitReport]:
    """Create ordered train/calibration/test splits and purge repeated patients downstream."""

    required = {"patient_id", "icu_admit_time"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Temporal split metadata is missing: {missing}")
    ordered = metadata.copy()
    ordered["icu_admit_time"] = pd.to_datetime(
        ordered["icu_admit_time"], errors="coerce", utc=True
    )
    if ordered[["patient_id", "icu_admit_time"]].isna().any().any():
        raise ValueError("Temporal split metadata must have patient IDs and parseable times.")
    ordered = ordered.sort_values(["icu_admit_time", "patient_id"], kind="mergesort")
    n = len(ordered)
    if n < 30:
        raise ValueError("At least 30 eligible rows are required for the temporal split.")
    train_end = max(1, int(math.floor(n * fractions[0])))
    calibration_end = max(train_end + 1, int(math.floor(n * (fractions[0] + fractions[1]))))
    calibration_end = min(calibration_end, n - 1)

    train = ordered.iloc[:train_end]
    calibration_raw = ordered.iloc[train_end:calibration_end]
    temporal_raw = ordered.iloc[calibration_end:]
    train_patients = set(train["patient_id"])
    calibration = calibration_raw.loc[~calibration_raw["patient_id"].isin(train_patients)]
    earlier_patients = train_patients | set(calibration["patient_id"])
    temporal = temporal_raw.loc[~temporal_raw["patient_id"].isin(earlier_patients)]

    splits = {
        "train": train.index,
        "calibration": calibration.index,
        "temporal_test": temporal.index,
    }
    for name, index in splits.items():
        if len(index) == 0:
            raise ValueError(f"Temporal split {name} is empty after patient purging.")

    patient_sets = {
        name: set(metadata.loc[index, "patient_id"].astype(str)) for name, index in splits.items()
    }
    overlap = bool(
        patient_sets["train"] & patient_sets["calibration"]
        or patient_sets["train"] & patient_sets["temporal_test"]
        or patient_sets["calibration"] & patient_sets["temporal_test"]
    )
    return splits, TemporalSplitReport(
        train_rows=len(train),
        calibration_rows=len(calibration),
        temporal_test_rows=len(temporal),
        purged_calibration_rows=len(calibration_raw) - len(calibration),
        purged_temporal_test_rows=len(temporal_raw) - len(temporal),
        train_end_time_utc=train["icu_admit_time"].max().isoformat(),
        calibration_end_time_utc=calibration["icu_admit_time"].max().isoformat(),
        patient_overlap_after_purge=overlap,
    )


def _fit_platt(y_true: pd.Series, probabilities: np.ndarray) -> LogisticRegression:
    y = pd.to_numeric(y_true, errors="raise").astype(int)
    if y.nunique() != 2:
        raise ValueError("Calibration target must contain two classes.")
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logits, y)
    return model


def _apply_platt(model: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def _predict_model(
    model_name: str,
    model: Any,
    frame: pd.DataFrame,
    predictors: tuple[str, ...],
    categorical: tuple[str, ...],
) -> np.ndarray:
    if model_name == "logistic_regression":
        return np.asarray(model.predict_proba(frame[list(predictors)])[:, 1], dtype=float)
    if model_name == "catboost":
        prepared = frame[list(predictors)].copy()
        for column in categorical:
            prepared[column] = prepared[column].astype("string").fillna("__MISSING__")
        return np.asarray(model.predict_proba(prepared)[:, 1], dtype=float)
    raise ValueError(f"Unknown baseline model: {model_name}")


def _fit_model_evaluation(
    model_name: str,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    temporal_test: pd.DataFrame,
    external: pd.DataFrame,
    target_column: str,
    seed: int,
) -> tuple[ModelEvaluation, np.ndarray]:
    predictors = tuple(assert_matrix_is_leakage_safe(train, target_column=target_column))
    categorical = tuple(
        column
        for column in predictors
        if not pd.api.types.is_numeric_dtype(train[column])
    )
    if model_name == "logistic_regression":
        model, calibration_raw, calibration_metrics = fit_logistic_baseline(
            train,
            calibration,
            target_column=target_column,
            data_classification="credentialed_locked",
            random_state=seed,
        )
    elif model_name == "catboost":
        model, calibration_raw, calibration_metrics = fit_catboost_baseline(
            train,
            calibration,
            target_column=target_column,
            data_classification="credentialed_locked",
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")

    calibrator = _fit_platt(calibration[target_column], calibration_raw)
    temporal_raw = _predict_model(model_name, model, temporal_test, predictors, categorical)
    external_raw = _predict_model(model_name, model, external, predictors, categorical)
    temporal_calibrated = _apply_platt(calibrator, temporal_raw)
    external_calibrated = _apply_platt(calibrator, external_raw)
    evaluation = ModelEvaluation(
        model=model_name,
        calibration_metrics=calibration_metrics,
        temporal_metrics_raw=evaluate_probabilities(temporal_test[target_column], temporal_raw),
        temporal_metrics_calibrated=evaluate_probabilities(
            temporal_test[target_column], temporal_calibrated
        ),
        external_metrics_raw=evaluate_probabilities(external[target_column], external_raw),
        external_metrics_calibrated=evaluate_probabilities(
            external[target_column], external_calibrated
        ),
        selected_from_development_only=False,
    )
    return evaluation, external_calibrated


def _choose_model(evaluations: tuple[ModelEvaluation, ...]) -> str:
    ordered = sorted(
        evaluations,
        key=lambda item: (
            -item.temporal_metrics_calibrated.pr_auc,
            item.temporal_metrics_calibrated.brier,
            0 if item.model == "logistic_regression" else 1,
        ),
    )
    return ordered[0].model


def _cluster_bootstrap(
    y_true: pd.Series,
    probabilities: np.ndarray,
    hospitals: pd.Series,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[BootstrapInterval, ...]:
    if iterations <= 0:
        return ()
    frame = pd.DataFrame(
        {
            "y": pd.to_numeric(y_true, errors="raise").astype(int).to_numpy(),
            "p": np.asarray(probabilities, dtype=float),
            "hospital": hospitals.astype("string").fillna("__MISSING__").to_numpy(),
        }
    )
    grouped = {name: group.index.to_numpy() for name, group in frame.groupby("hospital")}
    names = np.asarray(list(grouped), dtype=object)
    if len(names) < 2:
        raise ValueError("Cluster bootstrap requires at least two external hospitals.")
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"pr_auc": [], "roc_auc": [], "brier": []}
    for _ in range(iterations):
        sampled = rng.choice(names, size=len(names), replace=True)
        indices = np.concatenate([grouped[str(name)] for name in sampled])
        y_sample = frame.loc[indices, "y"]
        if y_sample.nunique() != 2:
            continue
        metrics = evaluate_probabilities(y_sample, frame.loc[indices, "p"].to_numpy())
        values["pr_auc"].append(metrics.pr_auc)
        values["roc_auc"].append(metrics.roc_auc)
        values["brier"].append(metrics.brier)

    alpha = (1.0 - confidence_level) / 2.0
    point = evaluate_probabilities(frame["y"], frame["p"].to_numpy())
    estimates = {"pr_auc": point.pr_auc, "roc_auc": point.roc_auc, "brier": point.brier}
    intervals: list[BootstrapInterval] = []
    for metric, samples in values.items():
        if not samples:
            raise RuntimeError(f"No valid cluster-bootstrap samples for {metric}.")
        intervals.append(
            BootstrapInterval(
                metric=metric,
                estimate=estimates[metric],
                lower=float(np.quantile(samples, alpha)),
                upper=float(np.quantile(samples, 1.0 - alpha)),
                successful_iterations=len(samples),
                requested_iterations=iterations,
            )
        )
    return tuple(intervals)


def _hospital_robustness(
    y_true: pd.Series,
    probabilities: np.ndarray,
    hospitals: pd.Series,
    *,
    minimum_rows: int,
    minimum_events: int,
) -> HospitalRobustness:
    frame = pd.DataFrame(
        {
            "y": pd.to_numeric(y_true, errors="raise").astype(int).to_numpy(),
            "p": np.asarray(probabilities, dtype=float),
            "hospital": hospitals.astype("string").fillna("__MISSING__").to_numpy(),
        }
    )
    metrics: list[BaselineMetrics] = []
    for _, group in frame.groupby("hospital"):
        events = int(group["y"].sum())
        negatives = len(group) - events
        if len(group) < minimum_rows or events < minimum_events or negatives < minimum_events:
            continue
        metrics.append(evaluate_probabilities(group["y"], group["p"].to_numpy()))
    if metrics:
        pr_values = np.asarray([metric.pr_auc for metric in metrics])
        brier_values = np.asarray([metric.brier for metric in metrics])
        median_pr = float(np.median(pr_values))
        p10_pr = float(np.quantile(pr_values, 0.10))
        worst_pr = float(np.min(pr_values))
        median_brier = float(np.median(brier_values))
        worst_brier = float(np.max(brier_values))
    else:
        median_pr = p10_pr = worst_pr = median_brier = worst_brier = None
    return HospitalRobustness(
        total_hospitals=int(frame["hospital"].nunique()),
        eligible_metric_hospitals=len(metrics),
        minimum_rows=minimum_rows,
        minimum_events=minimum_events,
        median_pr_auc=median_pr,
        tenth_percentile_pr_auc=p10_pr,
        worst_pr_auc=worst_pr,
        median_brier=median_brier,
        worst_brier=worst_brier,
    )


def _missingness_shift(development: pd.DataFrame, external: pd.DataFrame) -> MissingnessShift:
    columns = sorted(column for column in development if column.endswith("__missing"))
    if not columns:
        return MissingnessShift(0, 0.0, 0.0, 0, 0, ())
    differences = {
        column: abs(float(development[column].mean()) - float(external[column].mean()))
        for column in columns
    }
    ordered = sorted(differences.items(), key=lambda item: (-item[1], item[0]))
    values = np.asarray(list(differences.values()), dtype=float)
    return MissingnessShift(
        feature_count=len(columns),
        median_absolute_difference=float(np.median(values)),
        maximum_absolute_difference=float(np.max(values)),
        features_over_0_10=int((values > 0.10).sum()),
        features_over_0_20=int((values > 0.20).sum()),
        largest_differences=tuple((name, float(value)) for name, value in ordered[:10]),
    )


def _task_matrices(
    cohort: pd.DataFrame,
    events: pd.DataFrame,
    features: pd.DataFrame,
    task: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    labeled = assign_task_labels(cohort, events, task)
    eligibility = f"{task}__primary_analysis_eligible"
    target = f"{task}__target"
    labeled = labeled.loc[labeled[eligibility].astype(bool)].copy()
    metadata_columns = ["stay_id", "patient_id", "icu_admit_time", target]
    if "hospital_id" in labeled:
        metadata_columns.append("hospital_id")
    metadata = labeled[metadata_columns].copy()
    matrix = features.merge(labeled[["stay_id", target]], on="stay_id", how="inner")
    matrix = matrix.drop(columns=["stay_id"])
    if len(matrix) != len(metadata):
        raise RuntimeError("Feature and label rows are not one-to-one after task filtering.")
    matrix.index = metadata.index
    return matrix, metadata, target


def _require_two_classes(frame: pd.DataFrame, target: str, name: str) -> None:
    if frame[target].nunique() != 2:
        raise ValueError(f"{name} target must contain two classes.")


def run_task_phase0(
    *,
    task: str,
    mimic_matrix: pd.DataFrame,
    mimic_metadata: pd.DataFrame,
    eicu_matrix: pd.DataFrame,
    eicu_metadata: pd.DataFrame,
    config: StudyConfig,
    raw_config: dict[str, Any],
    models: tuple[str, ...] = _MODELS,
) -> TaskPhase0Report:
    """Fit prespecified baselines and apply the unchanged external feasibility gate."""

    if task not in _TASKS:
        raise ValueError(f"Unknown Phase 0 task: {task}")
    if not models or any(model not in _MODELS for model in models):
        raise ValueError(f"Models must be a non-empty subset of {_MODELS}.")
    fractions = _validate_fractions(raw_config)
    splits, split_report = temporal_patient_purged_split(mimic_metadata, fractions)

    target = f"{task}__target"
    train = mimic_matrix.loc[splits["train"]].copy()
    calibration = mimic_matrix.loc[splits["calibration"]].copy()
    temporal_test = mimic_matrix.loc[splits["temporal_test"]].copy()
    external = eicu_matrix.copy()
    for name, frame in (
        ("train", train),
        ("calibration", calibration),
        ("temporal_test", temporal_test),
        ("external", external),
    ):
        _require_two_classes(frame, target, name)

    evaluations: list[ModelEvaluation] = []
    external_probabilities: dict[str, np.ndarray] = {}
    for model_name in models:
        evaluation, probabilities = _fit_model_evaluation(
            model_name,
            train,
            calibration,
            temporal_test,
            external,
            target,
            config.seed,
        )
        evaluations.append(evaluation)
        external_probabilities[model_name] = probabilities

    selected_model = _choose_model(tuple(evaluations))
    selected_evaluations = tuple(
        ModelEvaluation(
            **{
                **evaluation.__dict__,
                "selected_from_development_only": evaluation.model == selected_model,
            }
        )
        for evaluation in evaluations
    )
    selected = next(item for item in selected_evaluations if item.model == selected_model)
    selected_probabilities = external_probabilities[selected_model]

    validation = raw_config["validation"]
    iterations = int(validation.get("bootstrap_iterations", 2000))
    confidence_level = float(validation.get("confidence_level", 0.95))
    minimum_rows = int(validation.get("hospital_metric_minimum_rows", 100))
    minimum_events = int(validation.get("hospital_metric_minimum_events", 10))
    if "hospital_id" not in eicu_metadata or eicu_metadata["hospital_id"].isna().any():
        raise ValueError("External eICU metadata requires a non-missing hospital_id for every row.")
    hospitals = eicu_metadata["hospital_id"]
    bootstrap = _cluster_bootstrap(
        external[target],
        selected_probabilities,
        hospitals,
        iterations=iterations,
        confidence_level=confidence_level,
        seed=config.seed,
    )
    robustness = _hospital_robustness(
        external[target],
        selected_probabilities,
        hospitals,
        minimum_rows=minimum_rows,
        minimum_events=minimum_events,
    )
    missingness = _missingness_shift(mimic_matrix, eicu_matrix)

    audit = DatasetAudit(
        task=task,
        development_n=len(mimic_matrix),
        development_events=int(mimic_matrix[target].sum()),
        external_n=len(eicu_matrix),
        external_events=int(eicu_matrix[target].sum()),
        external_hospitals=int(hospitals.astype("string").nunique()),
        external_pr_auc=selected.external_metrics_calibrated.pr_auc,
        external_brier=selected.external_metrics_calibrated.brier,
        external_calibration_slope=selected.external_metrics_calibrated.calibration_slope,
        external_calibration_intercept=selected.external_metrics_calibrated.calibration_intercept,
        no_post_index_leakage=True,
        outcome_definition_equivalent=True,
    )
    decision = evaluate_feasibility(audit, config)
    return TaskPhase0Report(
        task=task,
        development_rows=len(mimic_matrix),
        development_events=int(mimic_matrix[target].sum()),
        external_rows=len(eicu_matrix),
        external_events=int(eicu_matrix[target].sum()),
        split=split_report,
        selected_model=selected_model,
        selection_rule="mimic_temporal_calibrated_pr_auc_then_brier_then_logistic",
        models=selected_evaluations,
        external_cluster_bootstrap=bootstrap,
        hospital_robustness=robustness,
        missingness_shift=missingness,
        feasibility_decision=decision,
    )


def build_phase0_dry_run_plan(repo_root: str | Path) -> dict[str, Any]:
    """Return a credential-free execution plan and current outcome-lock blockers."""

    root = Path(repo_root).resolve()
    raw_config = _yaml_mapping(root / "configs/feasibility.yaml")
    _validate_fractions(raw_config)
    contracts = load_outcome_contracts(root / "schemas/outcome_contracts.yaml")
    lock_reports = []
    for task in _TASKS:
        try:
            assert_task_training_allowed(contracts, task)
            lock_reports.append({"task": task, "ready": True, "missing": []})
        except RuntimeError as exc:
            missing = str(exc).partition("missing: ")[2].rstrip(".")
            lock_reports.append(
                {"task": task, "ready": False, "missing": missing.split(", ") if missing else []}
            )
    return {
        "status": "credential_free_plan_only",
        "input_datasets": ["mimic_iv_3_1", "eicu_crd_2_0"],
        "models": list(_MODELS),
        "selection_rule": "mimic_temporal_calibrated_pr_auc_then_brier_then_logistic",
        "external_recalibration": "prohibited",
        "patient_overlap_policy": "purge_later_splits",
        "bootstrap_iterations": int(raw_config["validation"]["bootstrap_iterations"]),
        "outcome_locks": lock_reports,
    }


def execute_phase0_baselines(
    *,
    repo_root: str | Path,
    mimic_run_dir: str | Path,
    eicu_run_dir: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
    models: tuple[str, ...] = _MODELS,
) -> Phase0BaselineReport:
    """Execute the full locked Phase 0 baseline study and write aggregate-only artifacts."""

    root = Path(repo_root).resolve()
    output = Path(output_root).expanduser().resolve()
    if output == root or root in output.parents:
        raise ValueError("Phase 0 output must be outside the public repository.")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        output.chmod(0o700)
    except OSError:
        pass

    config_path = root / "configs/feasibility.yaml"
    feature_path = root / "schemas/phase0_features.yaml"
    outcome_path = root / "schemas/outcome_contracts.yaml"
    raw_config = _yaml_mapping(config_path)
    config = load_config(config_path)
    feature_contract = load_feature_contract(feature_path)
    outcome_contracts = load_outcome_contracts(outcome_path)
    for task in _TASKS:
        assert_task_training_allowed(outcome_contracts, task)

    mimic_verified = verify_credentialed_run(mimic_run_dir, "mimic_iv_3_1")
    eicu_verified = verify_credentialed_run(eicu_run_dir, "eicu_crd_2_0")
    mimic_stays = _read_stays(mimic_verified.stays_path)
    mimic_events = _read_events(mimic_verified.events_path)
    eicu_stays = _read_stays(eicu_verified.stays_path)
    eicu_events = _read_events(eicu_verified.events_path)
    mimic_cohort = build_landmark_cohort(mimic_stays)
    eicu_cohort = build_landmark_cohort(eicu_stays)

    mimic_features, mimic_feature_audit = build_feature_matrix_from_extract(
        mimic_verified.stays_path,
        mimic_verified.observations_path,
        feature_contract,
    )
    eicu_features, eicu_feature_audit = build_feature_matrix_from_extract(
        eicu_verified.stays_path,
        eicu_verified.observations_path,
        feature_contract,
    )
    for dataset, audit in (
        ("mimic_iv_3_1", mimic_feature_audit),
        ("eicu_crd_2_0", eicu_feature_audit),
    ):
        if audit.rows_before_icu or audit.rows_at_or_after_landmark or audit.unknown_variables:
            raise RuntimeError(f"Feature audit failed for {dataset}: {asdict(audit)}")

    reports: list[TaskPhase0Report] = []
    for task in _TASKS:
        mimic_matrix, mimic_metadata, _ = _task_matrices(
            mimic_cohort, mimic_events, mimic_features, task
        )
        eicu_matrix, eicu_metadata, _ = _task_matrices(
            eicu_cohort, eicu_events, eicu_features, task
        )
        if list(mimic_matrix.columns) != list(eicu_matrix.columns):
            raise RuntimeError(f"Cross-database model columns differ for {task}.")
        reports.append(
            run_task_phase0(
                task=task,
                mimic_matrix=mimic_matrix,
                mimic_metadata=mimic_metadata,
                eicu_matrix=eicu_matrix,
                eicu_metadata=eicu_metadata,
                config=config,
                raw_config=raw_config,
                models=models,
            )
        )

    provisional = Phase0BaselineReport(
        study=config.name,
        phase=config.phase,
        config_sha256=_sha256_file(config_path),
        feature_contract_sha256=_sha256_file(feature_path),
        outcome_contract_sha256=_sha256_file(outcome_path),
        mimic_input=mimic_verified,
        eicu_input=eicu_verified,
        feature_audits={
            "mimic_iv_3_1": asdict(mimic_feature_audit),
            "eicu_crd_2_0": asdict(eicu_feature_audit),
        },
        tasks=tuple(reports),
        all_tasks_continue=all(
            report.feasibility_decision.continue_to_architecture_development
            for report in reports
        ),
    )
    payload = provisional.to_dict()
    payload["report_sha256"] = ""
    report_hash = _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )
    report = Phase0BaselineReport(
        **{**provisional.__dict__, "report_sha256": report_hash}
    )
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False)
    (output / "phase0_go_no_go.json").write_text(rendered + "\n", encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    for task_report in report.tasks:
        for model_report in task_report.models:
            metrics = model_report.external_metrics_calibrated
            summary_rows.append(
                {
                    "task": task_report.task,
                    "model": model_report.model,
                    "selected_from_development_only": model_report.selected_from_development_only,
                    "external_n": metrics.n,
                    "external_events": metrics.events,
                    "external_prevalence": metrics.prevalence,
                    "external_pr_auc": metrics.pr_auc,
                    "external_pr_auc_prevalence_ratio": metrics.pr_auc_prevalence_ratio,
                    "external_roc_auc": metrics.roc_auc,
                    "external_brier": metrics.brier,
                    "external_calibration_slope": metrics.calibration_slope,
                    "external_calibration_intercept": metrics.calibration_intercept,
                    "continue_to_architecture_development": task_report.feasibility_decision.continue_to_architecture_development,
                }
            )
    pd.DataFrame(summary_rows).to_csv(output / "phase0_aggregate_summary.csv", index=False)
    try:
        os.chmod(output / "phase0_go_no_go.json", 0o600)
        os.chmod(output / "phase0_aggregate_summary.csv", 0o600)
    except OSError:
        pass
    return report
