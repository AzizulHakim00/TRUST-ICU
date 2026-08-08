"""Aggregate-only publication reporting for locked TRUST-ICU Phase 0 results.

The reporting layer never consumes patient-level predictions. It verifies the hashed
Phase 0 report, creates deterministic manuscript tables, records reporting traceability,
and renders aggregate figures outside the public repository.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

_TASKS = (
    "invasive_mechanical_ventilation",
    "vasopressor_initiation",
    "renal_replacement_therapy",
)
_BANNED_KEYS = {
    "patient_id",
    "stay_id",
    "hospital_id",
    "hospital_admission_id",
    "predictions",
    "row_predictions",
}


@dataclass(frozen=True)
class ReportingBundle:
    output_root: str
    source_report_sha256: str
    reporting_contract_sha256: str
    manifest_sha256: str
    files: tuple[str, ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_report_hash(payload: dict[str, Any]) -> str:
    material = copy.deepcopy(payload)
    material["report_sha256"] = ""
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


def load_and_verify_phase0_report(path: str | Path) -> dict[str, Any]:
    """Load a Phase 0 aggregate report and verify its integrity and privacy shape."""

    report_path = Path(path).expanduser().resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"Phase 0 report not found: {report_path}")
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Phase 0 report root must be a JSON object.")
    observed = str(raw.get("report_sha256", ""))
    expected = _canonical_report_hash(raw)
    if not observed or observed != expected:
        raise ValueError("Phase 0 report SHA-256 verification failed.")
    keys = _walk_keys(raw)
    leaked = sorted(keys & _BANNED_KEYS)
    if leaked:
        raise ValueError(f"Patient/site identifier keys are prohibited in reporting input: {leaked}")
    tasks = raw.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Phase 0 report must contain a tasks list.")
    observed_tasks = tuple(str(task.get("task", "")) for task in tasks if isinstance(task, dict))
    if set(observed_tasks) != set(_TASKS) or len(observed_tasks) != len(_TASKS):
        raise ValueError("Phase 0 report must contain exactly the three primary tasks.")
    return raw


def _load_contract(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Reporting contract must be a mapping.")
    expected_outputs = raw.get("outputs")
    if not isinstance(expected_outputs, list) or len(expected_outputs) < 6:
        raise ValueError("Reporting contract must define the publication bundle outputs.")
    return raw


def _task_index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["task"]): item for item in report["tasks"]}


def _selected_model(task: dict[str, Any]) -> dict[str, Any]:
    selected = str(task["selected_model"])
    matches = [model for model in task["models"] if str(model.get("model")) == selected]
    if len(matches) != 1:
        raise ValueError(f"Selected model {selected!r} is not uniquely represented.")
    if matches[0].get("selected_from_development_only") is not True:
        raise ValueError("Selected model must be explicitly marked development-only selection.")
    return matches[0]


def _cohort_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_name in _TASKS:
        task = _task_index(report)[task_name]
        dev_n = int(task["development_rows"])
        dev_events = int(task["development_events"])
        ext_n = int(task["external_rows"])
        ext_events = int(task["external_events"])
        decision = task["feasibility_decision"]
        rows.append(
            {
                "task": task_name,
                "development_n": dev_n,
                "development_events": dev_events,
                "development_prevalence": dev_events / dev_n,
                "external_n": ext_n,
                "external_events": ext_events,
                "external_prevalence": ext_events / ext_n,
                "selected_model": task["selected_model"],
                "feasibility_checks_passed": int(decision["passed_checks"]),
                "feasibility_checks_total": int(decision["total_checks"]),
                "continue_to_architecture_development": bool(
                    decision["continue_to_architecture_development"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _performance_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_name in _TASKS:
        task = _task_index(report)[task_name]
        for model in task["models"]:
            for evaluation_set, key in (
                ("mimic_temporal", "temporal_metrics_raw"),
                ("mimic_temporal_calibrated", "temporal_metrics_calibrated"),
                ("eicu_external", "external_metrics_raw"),
                ("eicu_external_calibrated", "external_metrics_calibrated"),
            ):
                metrics = model[key]
                rows.append(
                    {
                        "task": task_name,
                        "model": model["model"],
                        "evaluation_set": evaluation_set,
                        "selected_from_development_only": bool(
                            model["selected_from_development_only"]
                        ),
                        "n": int(metrics["n"]),
                        "events": int(metrics["events"]),
                        "prevalence": float(metrics["prevalence"]),
                        "pr_auc": float(metrics["pr_auc"]),
                        "pr_auc_prevalence_ratio": float(
                            metrics["pr_auc_prevalence_ratio"]
                        ),
                        "roc_auc": float(metrics["roc_auc"]),
                        "brier": float(metrics["brier"]),
                        "calibration_slope": float(metrics["calibration_slope"]),
                        "calibration_intercept": float(
                            metrics["calibration_intercept"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _robustness_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_name in _TASKS:
        task = _task_index(report)[task_name]
        robustness = task["hospital_robustness"]
        bootstrap = {item["metric"]: item for item in task["external_cluster_bootstrap"]}
        row: dict[str, Any] = {
            "task": task_name,
            "selected_model": task["selected_model"],
            "total_external_hospitals": int(robustness["total_hospitals"]),
            "eligible_metric_hospitals": int(robustness["eligible_metric_hospitals"]),
            "median_hospital_pr_auc": robustness["median_pr_auc"],
            "tenth_percentile_hospital_pr_auc": robustness["tenth_percentile_pr_auc"],
            "worst_hospital_pr_auc": robustness["worst_pr_auc"],
            "median_hospital_brier": robustness["median_brier"],
            "worst_hospital_brier": robustness["worst_brier"],
        }
        for metric in ("pr_auc", "roc_auc", "brier"):
            item = bootstrap.get(metric)
            row[f"{metric}_bootstrap_estimate"] = None if item is None else item["estimate"]
            row[f"{metric}_bootstrap_lower"] = None if item is None else item["lower"]
            row[f"{metric}_bootstrap_upper"] = None if item is None else item["upper"]
            row[f"{metric}_bootstrap_successful_iterations"] = (
                None if item is None else item["successful_iterations"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _tripod_traceability() -> pd.DataFrame:
    rows = [
        ("title_and_abstract", "README + manuscript metadata", "manual_completion_required"),
        ("background_and_objectives", "docs/phase0_protocol.md", "documented"),
        ("data_sources_and_setting", "configs/feasibility.yaml + hashed input reports", "documented"),
        ("participants_and_eligibility", "docs/phase0_protocol.md + cohort code", "documented"),
        ("outcome_definition", "private outcome lock report + schemas/outcome_contracts.yaml", "documented_after_lock"),
        ("predictors", "schemas/phase0_features.yaml", "documented"),
        ("sample_size_and_missing_data", "table_1 + missingness shift in Phase 0 report", "generated"),
        ("model_development", "baseline.py + configs/feasibility.yaml", "documented"),
        ("model_evaluation", "table_2 + table_3 + Phase 0 report", "generated"),
        ("model_specification_and_availability", "locked code/version hashes", "documented"),
        ("results_and_model_performance", "tables 1-3 + figures", "generated"),
        ("limitations_and_interpretation", "requires manuscript author judgment", "manual_completion_required"),
        ("open_science_and_reproducibility", "reproducibility_manifest.json + repository", "generated"),
        ("patient_public_involvement_and_fairness", "requires study-team documentation", "manual_completion_required"),
    ]
    return pd.DataFrame(rows, columns=["reporting_area", "evidence", "status"])


def _probast_self_audit(report: dict[str, Any]) -> pd.DataFrame:
    outcome_hash = bool(report.get("outcome_contract_sha256"))
    feature_hash = bool(report.get("feature_contract_sha256"))
    config_hash = bool(report.get("config_sha256"))
    rows = [
        {
            "domain": "participants_and_data_sources",
            "phase": "development_and_evaluation",
            "evidence_status": "available",
            "evidence": "hashed MIMIC-IV development and eICU external input reports",
            "formal_risk_of_bias_judgment": "not_assigned",
        },
        {
            "domain": "predictors",
            "phase": "development_and_evaluation",
            "evidence_status": "available" if feature_hash else "missing",
            "evidence": "locked six-hour feature contract and leakage guards",
            "formal_risk_of_bias_judgment": "not_assigned",
        },
        {
            "domain": "outcome",
            "phase": "development_and_evaluation",
            "evidence_status": "available" if outcome_hash else "missing",
            "evidence": "locked cross-database outcome contract hash",
            "formal_risk_of_bias_judgment": "not_assigned",
        },
        {
            "domain": "analysis",
            "phase": "development_and_evaluation",
            "evidence_status": "available" if config_hash else "missing",
            "evidence": "prespecified temporal split, calibration, external validation and go/no-go gates",
            "formal_risk_of_bias_judgment": "not_assigned",
        },
    ]
    return pd.DataFrame(rows)


def _render_external_performance(report: dict[str, Any], path: Path) -> None:
    labels: list[str] = []
    pr_auc: list[float] = []
    prevalence: list[float] = []
    for task_name in _TASKS:
        task = _task_index(report)[task_name]
        model = _selected_model(task)
        metrics = model["external_metrics_calibrated"]
        labels.append(task_name.replace("_", "\n"))
        pr_auc.append(float(metrics["pr_auc"]))
        prevalence.append(float(metrics["prevalence"]))
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = range(len(labels))
    ax.plot(positions, pr_auc, marker="o", label="External PR-AUC")
    ax.plot(positions, prevalence, marker="o", label="Outcome prevalence")
    ax.set_xticks(list(positions), labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Value")
    ax.set_title("External discrimination relative to outcome prevalence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _render_calibration(report: dict[str, Any], path: Path) -> None:
    labels: list[str] = []
    slopes: list[float] = []
    intercepts: list[float] = []
    for task_name in _TASKS:
        task = _task_index(report)[task_name]
        model = _selected_model(task)
        metrics = model["external_metrics_calibrated"]
        labels.append(task_name.replace("_", "\n"))
        slopes.append(float(metrics["calibration_slope"]))
        intercepts.append(float(metrics["calibration_intercept"]))
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = range(len(labels))
    ax.plot(positions, slopes, marker="o", label="Calibration slope")
    ax.plot(positions, intercepts, marker="o", label="Calibration intercept")
    ax.axhline(1.0, linestyle="--", linewidth=1, label="Ideal slope")
    ax.axhline(0.0, linestyle=":", linewidth=1, label="Ideal intercept")
    ax.set_xticks(list(positions), labels)
    ax.set_ylabel("Calibration parameter")
    ax.set_title("External calibration transportability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_reporting_dry_run_plan(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    contract_path = root / "schemas/reporting_contract.yaml"
    contract = _load_contract(contract_path)
    return {
        "status": "aggregate_reporting_plan_only",
        "contract_version": contract["contract_version"],
        "source_artifact": contract["source_artifact"],
        "outputs": contract["outputs"],
        "privacy": contract["privacy"],
        "formal_probast_ai_judgment_generated": False,
        "tripod_ai_checklist_text_reproduced": False,
    }


def generate_publication_bundle(
    *,
    repo_root: str | Path,
    phase0_report: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> ReportingBundle:
    """Create a deterministic aggregate publication bundle from a verified Phase 0 report."""

    root = Path(repo_root).resolve()
    output = Path(output_root).expanduser().resolve()
    if output == root or root in output.parents:
        raise ValueError("Publication reporting output must be outside the public repository.")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        output.chmod(0o700)
    except OSError:
        pass

    report = load_and_verify_phase0_report(phase0_report)
    contract_path = root / "schemas/reporting_contract.yaml"
    contract = _load_contract(contract_path)

    table1 = output / "table_1_cohort_summary.csv"
    table2 = output / "table_2_model_performance.csv"
    table3 = output / "table_3_external_robustness.csv"
    tripod = output / "tripod_ai_traceability.csv"
    probast = output / "probast_ai_self_audit.csv"
    figure1 = output / "figure_external_performance.png"
    figure2 = output / "figure_calibration_transportability.png"

    _cohort_table(report).to_csv(table1, index=False)
    _performance_table(report).to_csv(table2, index=False)
    _robustness_table(report).to_csv(table3, index=False)
    _tripod_traceability().to_csv(tripod, index=False)
    _probast_self_audit(report).to_csv(probast, index=False)
    _render_external_performance(report, figure1)
    _render_calibration(report, figure2)

    generated = [table1, table2, table3, tripod, probast, figure1, figure2]
    manifest_payload: dict[str, Any] = {
        "study": report.get("study"),
        "phase": report.get("phase"),
        "source_report_sha256": report["report_sha256"],
        "reporting_contract_sha256": _sha256_file(contract_path),
        "config_sha256": report.get("config_sha256"),
        "feature_contract_sha256": report.get("feature_contract_sha256"),
        "outcome_contract_sha256": report.get("outcome_contract_sha256"),
        "all_tasks_continue": bool(report.get("all_tasks_continue")),
        "formal_probast_ai_judgment_generated": False,
        "tripod_ai_checklist_text_reproduced": False,
        "files": [
            {
                "name": path.name,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in generated
        ],
        "manifest_sha256": "",
    }
    manifest_hash = _sha256_bytes(
        json.dumps(
            manifest_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    manifest_payload["manifest_sha256"] = manifest_hash
    manifest = output / "reproducibility_manifest.json"
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    generated.append(manifest)

    try:
        for path in generated:
            os.chmod(path, 0o600)
    except OSError:
        pass

    return ReportingBundle(
        output_root=str(output),
        source_report_sha256=str(report["report_sha256"]),
        reporting_contract_sha256=_sha256_file(contract_path),
        manifest_sha256=manifest_hash,
        files=tuple(path.name for path in generated),
    )
