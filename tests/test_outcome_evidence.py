import json
from dataclasses import asdict
from pathlib import Path

import yaml

from trust_icu.outcome_evidence import (
    LocalOutcomeSummary,
    OutcomeSummaryTask,
    _canonical_hash,
    prepare_locked_runtime_context,
    validate_outcome_evidence,
)
from trust_icu.outcomes import contract_sha256, load_outcome_contracts

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "invasive_mechanical_ventilation",
    "vasopressor_initiation",
    "renal_replacement_therapy",
)


def _summary(dataset: str) -> LocalOutcomeSummary:
    tasks = tuple(
        OutcomeSummaryTask(
            task=task,
            eligible_stays=1000,
            event_rows=160,
            event_stays=140,
            active_at_landmark_stays=40,
            prediction_window_event_rows=90,
            prediction_window_event_stays=80,
            incident_hospitals=25 if dataset == "eicu_crd_2_0" else 1,
            source_tables=2,
            source_codes=4,
            invalid_intervals=0,
        )
        for task in TASKS
    )
    provisional = LocalOutcomeSummary(
        dataset=dataset,
        credentialed_run_report_sha256="a" * 64,
        credentialed_audit_sha256="b" * 64,
        tasks=tasks,
    )
    payload = asdict(provisional)
    payload["summary_sha256"] = ""
    return LocalOutcomeSummary(
        **{**provisional.__dict__, "summary_sha256": _canonical_hash(payload, "summary_sha256")}
    )


def _evidence(mimic: LocalOutcomeSummary, eicu: LocalOutcomeSummary) -> dict:
    contract = load_outcome_contracts(ROOT / "schemas/outcome_contracts.yaml")
    task_evidence = {}
    for task in TASKS:
        task_evidence[task] = {
            "mimic": {"decision": "approved"},
            "eicu": {
                "decision": "approved",
                "candidate_terms_reviewed": 12,
                "unresolved_candidate_terms": 0,
                "locked_positive_mappings": 3,
            },
            "equivalence": {
                "decision": "approved",
                "same_clinical_event": True,
                "same_active_support_exclusion": True,
                "same_time_boundaries": True,
                "same_primary_estimand": True,
                "rationale": "Both databases represent the same incident organ-support estimand.",
            },
        }
    return {
        "evidence_version": "0.1.0",
        "public_contract_sha256": contract_sha256(contract),
        "mimic_summary_sha256": mimic.summary_sha256,
        "eicu_summary_sha256": eicu.summary_sha256,
        "reviewers": [
            {"name": "Data Reviewer", "role": "data_reviewer", "reviewed_on": "2026-08-06"},
            {
                "name": "Clinical Reviewer",
                "role": "clinical_reviewer",
                "reviewed_on": "2026-08-06",
            },
        ],
        "tasks": task_evidence,
    }


def _write_summary(path: Path, summary: LocalOutcomeSummary) -> None:
    path.write_text(json.dumps(asdict(summary)), encoding="utf-8")


def test_complete_evidence_is_ready() -> None:
    mimic = _summary("mimic_iv_3_1")
    eicu = _summary("eicu_crd_2_0")
    public = load_outcome_contracts(ROOT / "schemas/outcome_contracts.yaml")
    report = validate_outcome_evidence(
        public_contract=public,
        evidence=_evidence(mimic, eicu),
        mimic_summary=mimic,
        eicu_summary=eicu,
    )
    assert report.ready_for_runtime_lock is True
    assert all(item.ready for item in report.task_reports)


def test_unresolved_eicu_terms_block_lock() -> None:
    mimic = _summary("mimic_iv_3_1")
    eicu = _summary("eicu_crd_2_0")
    evidence = _evidence(mimic, eicu)
    evidence["tasks"][TASKS[0]]["eicu"]["unresolved_candidate_terms"] = 1
    public = load_outcome_contracts(ROOT / "schemas/outcome_contracts.yaml")
    report = validate_outcome_evidence(
        public_contract=public,
        evidence=evidence,
        mimic_summary=mimic,
        eicu_summary=eicu,
    )
    assert report.ready_for_runtime_lock is False
    assert "eicu_unresolved_candidate_terms" in report.task_reports[0].blockers


def test_private_runtime_context_is_created_without_modifying_public_contract(tmp_path: Path) -> None:
    mimic = _summary("mimic_iv_3_1")
    eicu = _summary("eicu_crd_2_0")
    evidence = _evidence(mimic, eicu)
    evidence_path = tmp_path / "evidence.json"
    mimic_path = tmp_path / "mimic.json"
    eicu_path = tmp_path / "eicu.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    _write_summary(mimic_path, mimic)
    _write_summary(eicu_path, eicu)
    public_path = ROOT / "schemas/outcome_contracts.yaml"
    public_before = public_path.read_bytes()

    output = tmp_path / "runtime"
    report = prepare_locked_runtime_context(
        repo_root=ROOT,
        evidence_path=evidence_path,
        mimic_summary_path=mimic_path,
        eicu_summary_path=eicu_path,
        output_root=output,
    )
    assert report.ready_for_runtime_lock is True
    locked = yaml.safe_load((output / "schemas/outcome_contracts.yaml").read_text())
    assert all(item["task_status"] == "locked" for item in locked["outcomes"].values())
    assert public_path.read_bytes() == public_before
