import copy
import json
from pathlib import Path

import pytest

from trust_icu.reporting import (
    _canonical_report_hash,
    build_reporting_dry_run_plan,
    generate_publication_bundle,
    load_and_verify_phase0_report,
)

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "invasive_mechanical_ventilation",
    "vasopressor_initiation",
    "renal_replacement_therapy",
)


def _metrics(n: int, events: int, pr_auc: float) -> dict[str, float | int]:
    prevalence = events / n
    return {
        "n": n,
        "events": events,
        "prevalence": prevalence,
        "pr_auc": pr_auc,
        "pr_auc_prevalence_ratio": pr_auc / prevalence,
        "roc_auc": min(0.99, pr_auc + 0.15),
        "brier": 0.12,
        "calibration_slope": 0.95,
        "calibration_intercept": 0.05,
    }


def _task(task: str, offset: float) -> dict:
    temporal = _metrics(300, 60, 0.60 + offset)
    external = _metrics(500, 100, 0.55 + offset)
    model = {
        "model": "logistic_regression",
        "calibration_metrics": _metrics(300, 60, 0.58 + offset),
        "temporal_metrics_raw": temporal,
        "temporal_metrics_calibrated": temporal,
        "external_metrics_raw": external,
        "external_metrics_calibrated": external,
        "selected_from_development_only": True,
    }
    return {
        "task": task,
        "development_rows": 2000,
        "development_events": 400,
        "external_rows": 3000,
        "external_events": 600,
        "split": {},
        "selected_model": "logistic_regression",
        "selection_rule": "development_only",
        "models": [model],
        "external_cluster_bootstrap": [
            {
                "metric": "pr_auc",
                "estimate": external["pr_auc"],
                "lower": external["pr_auc"] - 0.03,
                "upper": external["pr_auc"] + 0.03,
                "successful_iterations": 2000,
                "requested_iterations": 2000,
            },
            {
                "metric": "roc_auc",
                "estimate": external["roc_auc"],
                "lower": external["roc_auc"] - 0.02,
                "upper": external["roc_auc"] + 0.02,
                "successful_iterations": 2000,
                "requested_iterations": 2000,
            },
            {
                "metric": "brier",
                "estimate": external["brier"],
                "lower": 0.10,
                "upper": 0.14,
                "successful_iterations": 2000,
                "requested_iterations": 2000,
            },
        ],
        "hospital_robustness": {
            "total_hospitals": 50,
            "eligible_metric_hospitals": 35,
            "minimum_rows": 100,
            "minimum_events": 10,
            "median_pr_auc": 0.56 + offset,
            "tenth_percentile_pr_auc": 0.45 + offset,
            "worst_pr_auc": 0.35 + offset,
            "median_brier": 0.13,
            "worst_brier": 0.20,
        },
        "missingness_shift": {},
        "feasibility_decision": {
            "task": task,
            "continue_to_architecture_development": False,
            "passed_checks": 6,
            "total_checks": 8,
            "recommended_action": "freeze_or_pivot",
            "checks": [],
        },
    }


def _report() -> dict:
    raw = {
        "study": "TRUST-ICU",
        "phase": "phase_0_feasibility",
        "config_sha256": "a" * 64,
        "feature_contract_sha256": "b" * 64,
        "outcome_contract_sha256": "c" * 64,
        "mimic_input": {"report_sha256": "d" * 64},
        "eicu_input": {"report_sha256": "e" * 64},
        "feature_audits": {},
        "tasks": [_task(task, index * 0.02) for index, task in enumerate(TASKS)],
        "all_tasks_continue": False,
        "report_sha256": "",
    }
    raw["report_sha256"] = _canonical_report_hash(raw)
    return raw


def _write_report(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "phase0_go_no_go.json"
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_reporting_dry_run_has_no_formal_probast_judgment() -> None:
    plan = build_reporting_dry_run_plan(ROOT)
    assert plan["formal_probast_ai_judgment_generated"] is False
    assert plan["tripod_ai_checklist_text_reproduced"] is False
    assert "reproducibility_manifest.json" in plan["outputs"]


def test_tampered_phase0_report_is_rejected(tmp_path: Path) -> None:
    raw = _report()
    raw["all_tasks_continue"] = True
    path = _write_report(tmp_path, raw)
    with pytest.raises(ValueError, match="SHA-256"):
        load_and_verify_phase0_report(path)


def test_identifier_bearing_report_is_rejected(tmp_path: Path) -> None:
    raw = _report()
    raw["patient_id"] = "forbidden"
    raw["report_sha256"] = _canonical_report_hash(raw)
    path = _write_report(tmp_path, raw)
    with pytest.raises(ValueError, match="identifier keys"):
        load_and_verify_phase0_report(path)


def test_publication_bundle_generation(tmp_path: Path) -> None:
    report_path = _write_report(tmp_path, _report())
    output = tmp_path / "bundle"
    bundle = generate_publication_bundle(
        repo_root=ROOT,
        phase0_report=report_path,
        output_root=output,
    )
    expected = {
        "reproducibility_manifest.json",
        "tripod_ai_traceability.csv",
        "probast_ai_self_audit.csv",
        "table_1_cohort_summary.csv",
        "table_2_model_performance.csv",
        "table_3_external_robustness.csv",
        "figure_external_performance.png",
        "figure_calibration_transportability.png",
    }
    assert set(bundle.files) == expected
    assert all((output / name).is_file() for name in expected)
    manifest = json.loads((output / "reproducibility_manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_sha256"] == bundle.manifest_sha256
    assert manifest["formal_probast_ai_judgment_generated"] is False
    assert len(manifest["files"]) == 7


def test_reporting_output_must_remain_outside_repository(tmp_path: Path) -> None:
    raw = _report()
    report_path = _write_report(tmp_path, raw)
    with pytest.raises(ValueError, match="outside the public repository"):
        generate_publication_bundle(
            repo_root=ROOT,
            phase0_report=report_path,
            output_root=ROOT / "unsafe_reporting_output",
        )


def test_report_hash_is_deterministic() -> None:
    raw = _report()
    clone = copy.deepcopy(raw)
    assert _canonical_report_hash(raw) == _canonical_report_hash(clone)
