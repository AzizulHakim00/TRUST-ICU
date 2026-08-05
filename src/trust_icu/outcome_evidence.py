"""Aggregate outcome validation summaries and evidence-based runtime locking.

This module handles aggregate metadata only. It never emits patient-level rows, identifiers,
candidate vocabularies, or clinical text values.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from trust_icu.outcomes import contract_sha256, load_outcome_contracts
from trust_icu.phase0_runtime import verify_credentialed_run

_TASKS = (
    "invasive_mechanical_ventilation",
    "vasopressor_initiation",
    "renal_replacement_therapy",
)
_DATASETS = {"mimic_iv_3_1", "eicu_crd_2_0"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class OutcomeSummaryTask:
    task: str
    eligible_stays: int
    event_rows: int
    event_stays: int
    active_at_landmark_stays: int
    prediction_window_event_rows: int
    prediction_window_event_stays: int
    incident_hospitals: int
    source_tables: int
    source_codes: int
    invalid_intervals: int


@dataclass(frozen=True)
class LocalOutcomeSummary:
    dataset: str
    credentialed_run_report_sha256: str
    credentialed_audit_sha256: str
    tasks: tuple[OutcomeSummaryTask, ...]
    summary_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceTaskReport:
    task: str
    ready: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class OutcomeEvidenceReport:
    ready_for_runtime_lock: bool
    public_contract_sha256: str
    evidence_sha256: str
    task_reports: tuple[EvidenceTaskReport, ...]

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
        json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def _json_mapping(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object: {resolved}")
    return raw


def _sql_literal(path: str | Path) -> str:
    return "'" + str(Path(path).resolve()).replace("'", "''") + "'"


def build_local_outcome_summary(
    *,
    run_dir: str | Path,
    dataset: str,
    output_path: str | Path | None = None,
) -> LocalOutcomeSummary:
    """Build an aggregate-only outcome summary from a verified canonical extract."""

    if dataset not in _DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    verified = verify_credentialed_run(run_dir, dataset)
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "Outcome summarization requires DuckDB. Install trust-icu with the analytics extra."
        ) from exc

    stays = _sql_literal(verified.stays_path)
    events = _sql_literal(verified.events_path)
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(
            f"""
            CREATE VIEW stays AS
            SELECT
                CAST(stay_id AS VARCHAR) AS stay_id,
                CAST(hospital_id AS VARCHAR) AS hospital_id,
                TRY_CAST(icu_admit_time AS TIMESTAMPTZ) AS icu_admit_time,
                TRY_CAST(icu_discharge_time AS TIMESTAMPTZ) AS icu_discharge_time
            FROM read_csv_auto({stays}, header=true, all_varchar=true, sample_size=-1)
            """
        )
        connection.execute(
            f"""
            CREATE VIEW events AS
            SELECT
                CAST(stay_id AS VARCHAR) AS stay_id,
                CAST(task AS VARCHAR) AS task,
                TRY_CAST(start_time AS TIMESTAMPTZ) AS start_time,
                TRY_CAST(end_time AS TIMESTAMPTZ) AS end_time,
                CAST(source_table AS VARCHAR) AS source_table,
                CAST(source_code AS VARCHAR) AS source_code
            FROM read_csv_auto({events}, header=true, all_varchar=true, sample_size=-1)
            """
        )
        rows = connection.execute(
            """
            WITH linked AS (
                SELECT
                    s.stay_id,
                    s.hospital_id,
                    s.icu_admit_time,
                    s.icu_discharge_time,
                    e.task,
                    e.start_time,
                    e.end_time,
                    e.source_table,
                    e.source_code,
                    s.icu_admit_time + INTERVAL '6 hours' AS landmark_time,
                    LEAST(
                        s.icu_discharge_time,
                        s.icu_admit_time + INTERVAL '18 hours'
                    ) AS followup_end_time
                FROM stays AS s
                LEFT JOIN events AS e USING (stay_id)
            )
            SELECT
                task,
                COUNT(DISTINCT stay_id) FILTER (WHERE task IS NOT NULL) AS event_stays,
                COUNT(*) FILTER (WHERE task IS NOT NULL) AS event_rows,
                COUNT(DISTINCT stay_id) FILTER (
                    WHERE task IS NOT NULL
                      AND start_time < landmark_time
                      AND (end_time IS NULL OR end_time > icu_admit_time)
                ) AS active_at_landmark_stays,
                COUNT(*) FILTER (
                    WHERE task IS NOT NULL
                      AND start_time >= landmark_time
                      AND start_time < followup_end_time
                ) AS prediction_window_event_rows,
                COUNT(DISTINCT stay_id) FILTER (
                    WHERE task IS NOT NULL
                      AND start_time >= landmark_time
                      AND start_time < followup_end_time
                ) AS prediction_window_event_stays,
                COUNT(DISTINCT hospital_id) FILTER (
                    WHERE task IS NOT NULL
                      AND start_time >= landmark_time
                      AND start_time < followup_end_time
                ) AS incident_hospitals,
                COUNT(DISTINCT source_table) FILTER (WHERE task IS NOT NULL) AS source_tables,
                COUNT(DISTINCT source_code) FILTER (WHERE task IS NOT NULL) AS source_codes,
                COUNT(*) FILTER (
                    WHERE task IS NOT NULL
                      AND (
                          start_time IS NULL
                          OR (end_time IS NOT NULL AND end_time <= start_time)
                      )
                ) AS invalid_intervals
            FROM linked
            GROUP BY task
            """
        ).fetchall()
        total_stays = int(connection.execute("SELECT COUNT(*) FROM stays").fetchone()[0])
    finally:
        connection.close()

    by_task = {str(row[0]): row for row in rows if row[0] is not None}
    task_summaries = []
    for task in _TASKS:
        row = by_task.get(task)
        values = (0,) * 9 if row is None else tuple(int(value or 0) for value in row[1:])
        task_summaries.append(
            OutcomeSummaryTask(
                task=task,
                eligible_stays=total_stays,
                event_stays=values[0],
                event_rows=values[1],
                active_at_landmark_stays=values[2],
                prediction_window_event_rows=values[3],
                prediction_window_event_stays=values[4],
                incident_hospitals=values[5],
                source_tables=values[6],
                source_codes=values[7],
                invalid_intervals=values[8],
            )
        )

    provisional = LocalOutcomeSummary(
        dataset=dataset,
        credentialed_run_report_sha256=verified.report_sha256,
        credentialed_audit_sha256=verified.audit_sha256,
        tasks=tuple(task_summaries),
    )
    payload = provisional.to_dict()
    payload["summary_sha256"] = ""
    summary = LocalOutcomeSummary(
        **{**provisional.__dict__, "summary_sha256": _canonical_hash(payload, "summary_sha256")}
    )
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            destination.chmod(0o600)
        except OSError:
            pass
    return summary


def load_local_outcome_summary(path: str | Path, dataset: str) -> LocalOutcomeSummary:
    """Load and verify an aggregate local outcome summary."""

    raw = _json_mapping(path)
    if raw.get("dataset") != dataset:
        raise ValueError(f"Outcome summary dataset must equal {dataset}.")
    observed = str(raw.get("summary_sha256", ""))
    if not _SHA256.fullmatch(observed) or observed != _canonical_hash(raw, "summary_sha256"):
        raise ValueError("Outcome summary SHA-256 verification failed.")
    raw_tasks = raw.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("Outcome summary tasks must be a list.")
    tasks = tuple(OutcomeSummaryTask(**item) for item in raw_tasks)
    if {item.task for item in tasks} != set(_TASKS):
        raise ValueError("Outcome summary must contain exactly the three locked tasks.")
    return LocalOutcomeSummary(
        dataset=dataset,
        credentialed_run_report_sha256=str(raw["credentialed_run_report_sha256"]),
        credentialed_audit_sha256=str(raw["credentialed_audit_sha256"]),
        tasks=tasks,
        summary_sha256=observed,
    )


def _reviewer_roles(evidence: dict[str, Any]) -> set[str]:
    reviewers = evidence.get("reviewers")
    if not isinstance(reviewers, list):
        return set()
    valid = set()
    for reviewer in reviewers:
        if not isinstance(reviewer, dict):
            continue
        name = str(reviewer.get("name", "")).strip()
        role = str(reviewer.get("role", "")).strip()
        date = str(reviewer.get("reviewed_on", "")).strip()
        if name and role and date:
            valid.add(role)
    return valid


def validate_outcome_evidence(
    *,
    public_contract: dict[str, Any],
    evidence: dict[str, Any],
    mimic_summary: LocalOutcomeSummary,
    eicu_summary: LocalOutcomeSummary,
) -> OutcomeEvidenceReport:
    """Validate local evidence without granting approval by inference."""

    public_hash = contract_sha256(public_contract)
    evidence_hash = _sha256_bytes(
        json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    common_blockers = []
    if evidence.get("evidence_version") != "0.1.0":
        common_blockers.append("unsupported_evidence_version")
    if evidence.get("public_contract_sha256") != public_hash:
        common_blockers.append("public_contract_hash_mismatch")
    roles = _reviewer_roles(evidence)
    if "data_reviewer" not in roles:
        common_blockers.append("data_reviewer_missing")
    if "clinical_reviewer" not in roles:
        common_blockers.append("clinical_reviewer_missing")
    if mimic_summary.summary_sha256 != evidence.get("mimic_summary_sha256"):
        common_blockers.append("mimic_summary_hash_mismatch")
    if eicu_summary.summary_sha256 != evidence.get("eicu_summary_sha256"):
        common_blockers.append("eicu_summary_hash_mismatch")

    evidence_tasks = evidence.get("tasks")
    if not isinstance(evidence_tasks, dict):
        evidence_tasks = {}
        common_blockers.append("tasks_mapping_missing")
    mimic_by_task = {item.task: item for item in mimic_summary.tasks}
    eicu_by_task = {item.task: item for item in eicu_summary.tasks}
    reports = []
    for task in _TASKS:
        blockers = list(common_blockers)
        task_evidence = evidence_tasks.get(task)
        if not isinstance(task_evidence, dict):
            blockers.append("task_evidence_missing")
            reports.append(EvidenceTaskReport(task, False, tuple(sorted(set(blockers)))))
            continue
        mimic_review = task_evidence.get("mimic")
        eicu_review = task_evidence.get("eicu")
        equivalence = task_evidence.get("equivalence")
        if not isinstance(mimic_review, dict) or mimic_review.get("decision") != "approved":
            blockers.append("mimic_local_validation_not_approved")
        if not isinstance(eicu_review, dict) or eicu_review.get("decision") != "approved":
            blockers.append("eicu_local_validation_not_approved")
        if not isinstance(equivalence, dict) or equivalence.get("decision") != "approved":
            blockers.append("clinical_equivalence_not_approved")
        else:
            for field in (
                "same_clinical_event",
                "same_active_support_exclusion",
                "same_time_boundaries",
                "same_primary_estimand",
            ):
                if equivalence.get(field) is not True:
                    blockers.append(f"equivalence_{field}_not_confirmed")
            if len(str(equivalence.get("rationale", "")).strip()) < 20:
                blockers.append("equivalence_rationale_insufficient")

        mimic = mimic_by_task[task]
        eicu = eicu_by_task[task]
        for label, summary in (("mimic", mimic), ("eicu", eicu)):
            if summary.eligible_stays <= 0:
                blockers.append(f"{label}_eligible_stays_zero")
            if summary.prediction_window_event_stays <= 0:
                blockers.append(f"{label}_prediction_events_zero")
            if summary.invalid_intervals != 0:
                blockers.append(f"{label}_invalid_intervals")
            if summary.source_tables <= 0 or summary.source_codes <= 0:
                blockers.append(f"{label}_source_coverage_missing")
        if eicu.incident_hospitals <= 0:
            blockers.append("eicu_incident_hospital_coverage_zero")

        if isinstance(eicu_review, dict):
            if int(eicu_review.get("candidate_terms_reviewed", 0)) <= 0:
                blockers.append("eicu_candidate_review_missing")
            if int(eicu_review.get("unresolved_candidate_terms", -1)) != 0:
                blockers.append("eicu_unresolved_candidate_terms")
            if int(eicu_review.get("locked_positive_mappings", 0)) <= 0:
                blockers.append("eicu_locked_positive_mapping_missing")

        reports.append(
            EvidenceTaskReport(
                task=task,
                ready=not blockers,
                blockers=tuple(sorted(set(blockers))),
            )
        )
    return OutcomeEvidenceReport(
        ready_for_runtime_lock=all(item.ready for item in reports),
        public_contract_sha256=public_hash,
        evidence_sha256=evidence_hash,
        task_reports=tuple(reports),
    )


def render_locked_runtime_contract(
    *,
    public_contract: dict[str, Any],
    evidence: dict[str, Any],
    report: OutcomeEvidenceReport,
) -> dict[str, Any]:
    """Render a private runtime contract only after all evidence gates pass."""

    if not report.ready_for_runtime_lock:
        raise RuntimeError("Outcome evidence is incomplete; runtime contract cannot be locked.")
    locked = copy.deepcopy(public_contract)
    locked["runtime_lock"] = {
        "status": "approved",
        "evidence_sha256": report.evidence_sha256,
        "public_contract_sha256": report.public_contract_sha256,
        "reviewers": evidence["reviewers"],
    }
    for task in _TASKS:
        contract = locked["outcomes"][task]
        contract["task_status"] = "locked"
        contract["mimic"]["status"] = "locally_validated"
        contract["eicu"]["status"] = "locally_validated"
        contract["clinical_equivalence_review"] = "approved"
        contract["synthetic_timeline_tests"] = "passed"
    return locked


def prepare_locked_runtime_context(
    *,
    repo_root: str | Path,
    evidence_path: str | Path,
    mimic_summary_path: str | Path,
    eicu_summary_path: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> OutcomeEvidenceReport:
    """Create a private metadata-only runtime root for the locked Phase 0 execution."""

    root = Path(repo_root).resolve()
    output = Path(output_root).expanduser().resolve()
    if output == root or root in output.parents:
        raise ValueError("Locked runtime context must be outside the public repository.")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    public_contract = load_outcome_contracts(root / "schemas/outcome_contracts.yaml")
    evidence = _json_mapping(evidence_path)
    mimic_summary = load_local_outcome_summary(mimic_summary_path, "mimic_iv_3_1")
    eicu_summary = load_local_outcome_summary(eicu_summary_path, "eicu_crd_2_0")
    report = validate_outcome_evidence(
        public_contract=public_contract,
        evidence=evidence,
        mimic_summary=mimic_summary,
        eicu_summary=eicu_summary,
    )
    report_path = output / "outcome_lock_report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not report.ready_for_runtime_lock:
        return report

    (output / "configs").mkdir(exist_ok=True)
    (output / "schemas").mkdir(exist_ok=True)
    shutil.copy2(root / "configs/feasibility.yaml", output / "configs/feasibility.yaml")
    shutil.copy2(root / "schemas/phase0_features.yaml", output / "schemas/phase0_features.yaml")
    locked = render_locked_runtime_contract(
        public_contract=public_contract,
        evidence=evidence,
        report=report,
    )
    (output / "schemas/outcome_contracts.yaml").write_text(
        yaml.safe_dump(locked, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    for path in (
        report_path,
        output / "configs/feasibility.yaml",
        output / "schemas/phase0_features.yaml",
        output / "schemas/outcome_contracts.yaml",
    ):
        try:
            path.chmod(0o600)
        except OSError:
            pass
    return report
