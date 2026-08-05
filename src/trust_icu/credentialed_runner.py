"""Secure orchestration for credentialed PostgreSQL source-adapter execution.

The module never stores credentials and never prints patient-level rows. It materializes
canonical relations inside PostgreSQL, runs aggregate fail-closed audits, and exports only
after the audit passes unless the caller explicitly requests failed artifacts for debugging.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import yaml

from trust_icu.adapter_manifest import load_and_validate_adapter_manifest
from trust_icu.features import FeatureContract, load_feature_contract
from trust_icu.source_validation import (
    CanonicalExtractAudit,
    load_source_adapter_contract,
    mapping_sha256,
)

_RELATION = re.compile(r"^[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*$")
_DATASETS = {"mimic_iv_3_1", "eicu_crd_2_0"}
_TASKS = (
    "invasive_mechanical_ventilation",
    "vasopressor_initiation",
    "renal_replacement_therapy",
)


class CursorLike(Protocol):
    def execute(self, query: str, params: Any = None) -> Any: ...
    def fetchone(self) -> tuple[Any, ...] | None: ...
    def fetchall(self) -> list[tuple[Any, ...]]: ...
    def copy(self, query: str) -> Any: ...


class ConnectionLike(Protocol):
    def cursor(self) -> Any: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...


@dataclass(frozen=True)
class ExportArtifact:
    name: str
    path: str
    row_count: int
    sha256: str
    bytes: int


@dataclass(frozen=True)
class CredentialedRunReport:
    dataset: str
    started_at_utc: str
    completed_at_utc: str
    manifest_version: str
    adapter_status: str
    database_server_version: str
    relations: dict[str, str]
    sql_sha256: dict[str, str]
    audit: dict[str, Any]
    exports: tuple[ExportArtifact, ...]
    report_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["exports"] = [asdict(item) for item in self.exports]
        return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return raw


def _safe_relation(value: str) -> str:
    if not _RELATION.fullmatch(value):
        raise ValueError(f"Unsafe PostgreSQL relation name: {value!r}")
    return value


def _strip_final_semicolon(sql_text: str) -> str:
    stripped = sql_text.strip()
    while stripped.endswith(";"):
        stripped = stripped[:-1].rstrip()
    if not stripped:
        raise ValueError("SQL file is empty.")
    return stripped


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relation_set(dataset: str, stays_relation: str) -> dict[str, str]:
    stays = _safe_relation(stays_relation)
    schema, name = stays.split(".", 1)
    if not name.endswith("_stays"):
        raise ValueError("materialized_stays_relation must end with '_stays'.")
    prefix = name[: -len("_stays")]
    return {
        "stays": stays,
        "events": _safe_relation(f"{schema}.{prefix}_events"),
        "observations": _safe_relation(f"{schema}.{prefix}_observations"),
    }


def _read_plan(
    repo_root: Path,
    dataset: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[str, Path]]:
    if dataset not in _DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    manifest_path = repo_root / "schemas/source_adapter_manifest.yaml"
    manifest_report = load_and_validate_adapter_manifest(manifest_path, repo_root)
    manifest = _load_yaml(manifest_path)
    spec = manifest["datasets"][dataset]
    dataset_report = {item.dataset: item for item in manifest_report.datasets}[dataset]
    if not dataset_report.files_present:
        raise RuntimeError(f"Adapter files are incomplete for {dataset}.")
    relations = _relation_set(dataset, str(spec["materialized_stays_relation"]))
    sql_paths = {
        name: repo_root / relative
        for name, relative in spec["execution_order"].items()
    }
    return manifest, spec, relations, sql_paths


def build_dry_run_plan(repo_root: str | Path, dataset: str) -> dict[str, Any]:
    """Return a credential-free execution plan with no secrets or patient data."""

    root = Path(repo_root).resolve()
    manifest, spec, relations, sql_paths = _read_plan(root, dataset)
    return {
        "dataset": dataset,
        "manifest_version": str(manifest["manifest_version"]),
        "adapter_status": str(spec["status"]),
        "relations": relations,
        "sql_files": {name: str(path.relative_to(root)) for name, path in sql_paths.items()},
        "sql_sha256": {
            name: _sha256_bytes(path.read_bytes()) for name, path in sql_paths.items()
        },
        "local_mappings_required": bool(spec["requires_local_vocabulary_tables"]),
        "credentialed_validation_required": bool(spec["credentialed_validation_required"]),
    }


def _scalar(cursor: CursorLike, query: str, params: Any = None) -> int:
    cursor.execute(query, params)
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("Aggregate query returned no row.")
    return int(row[0] or 0)


def _distinct_text(cursor: CursorLike, query: str, params: Any = None) -> tuple[str, ...]:
    cursor.execute(query, params)
    return tuple(sorted(str(row[0]) for row in cursor.fetchall() if row[0] is not None))


def _blank_expression(columns: tuple[str, ...]) -> str:
    return " OR ".join(f"{column} IS NULL OR BTRIM({column}::text) = ''" for column in columns)


def _postgres_audit(
    connection: ConnectionLike,
    *,
    dataset: str,
    relations: dict[str, str],
    source_contract: dict[str, Any],
    feature_contract: FeatureContract,
) -> CanonicalExtractAudit:
    stays = _safe_relation(relations["stays"])
    events = _safe_relation(relations["events"])
    observations = _safe_relation(relations["observations"])
    expected_units = {
        variable.name: variable.canonical_unit for variable in feature_contract.variables
    }
    allowed_tasks = tuple(source_contract["allowed_tasks"])
    window_minutes = int(source_contract["observation_window"]["end_minutes"])

    with connection.cursor() as cursor:
        stay_rows = _scalar(cursor, f"SELECT COUNT(*) FROM {stays}")
        event_rows = _scalar(cursor, f"SELECT COUNT(*) FROM {events}")
        observation_rows = _scalar(cursor, f"SELECT COUNT(*) FROM {observations}")
        unique_stays = _scalar(cursor, f"SELECT COUNT(DISTINCT stay_id) FROM {stays}")

        dataset_values = _distinct_text(
            cursor, f"SELECT DISTINCT dataset_id::text FROM {stays} WHERE dataset_id IS NOT NULL"
        )
        if dataset_values != (dataset,):
            raise RuntimeError(
                f"{stays}.dataset_id must contain only {dataset!r}; got {dataset_values}."
            )

        duplicate_stay_ids = _scalar(
            cursor,
            f"""
            SELECT COALESCE(SUM(n), 0)
            FROM (
                SELECT COUNT(*) AS n FROM {stays}
                GROUP BY stay_id HAVING COUNT(*) > 1
            ) AS duplicated
            """,
        )
        duplicate_observation_rows = _scalar(
            cursor,
            f"""
            SELECT COALESCE(SUM(n), 0)
            FROM (
                SELECT COUNT(*) AS n
                FROM {observations}
                GROUP BY stay_id, variable, event_time, value, unit, source_table, source_code
                HAVING COUNT(*) > 1
            ) AS duplicated
            """,
        )
        unlinked_event_rows = _scalar(
            cursor,
            f"""
            SELECT COUNT(*) FROM {events} e
            LEFT JOIN {stays} s ON e.stay_id::text = s.stay_id::text
            WHERE s.stay_id IS NULL
            """,
        )
        unlinked_observation_rows = _scalar(
            cursor,
            f"""
            SELECT COUNT(*) FROM {observations} o
            LEFT JOIN {stays} s ON o.stay_id::text = s.stay_id::text
            WHERE s.stay_id IS NULL
            """,
        )
        observations_before_icu = _scalar(
            cursor,
            f"""
            SELECT COUNT(*) FROM {observations} o
            JOIN {stays} s ON o.stay_id::text = s.stay_id::text
            WHERE o.event_time < s.icu_admit_time
            """,
        )
        observations_at_or_after_landmark = _scalar(
            cursor,
            f"""
            SELECT COUNT(*) FROM {observations} o
            JOIN {stays} s ON o.stay_id::text = s.stay_id::text
            WHERE o.event_time >= s.icu_admit_time + (%s * INTERVAL '1 minute')
            """,
            (window_minutes,),
        )
        invalid_observation_values = _scalar(
            cursor,
            f"""
            SELECT COUNT(*) FROM {observations}
            WHERE value IS NULL
               OR value::text IN ('NaN', 'Infinity', '-Infinity')
            """,
        )
        invalid_icu_intervals = _scalar(
            cursor,
            f"SELECT COUNT(*) FROM {stays} WHERE icu_discharge_time <= icu_admit_time",
        )
        invalid_event_intervals = _scalar(
            cursor,
            f"""
            SELECT COUNT(*) FROM {events}
            WHERE end_time IS NOT NULL AND end_time <= start_time
            """,
        )
        missing_stay_core_values = _scalar(
            cursor,
            f"""
            SELECT COUNT(*) FROM {stays}
            WHERE {_blank_expression(("dataset_id", "patient_id", "hospital_admission_id", "stay_id"))}
               OR icu_admit_time IS NULL OR icu_discharge_time IS NULL
            """,
        )
        missing_event_core_values = _scalar(
            cursor,
            f"""
            SELECT COUNT(*) FROM {events}
            WHERE {_blank_expression(("stay_id", "task"))} OR start_time IS NULL
            """,
        )
        missing_observation_core_values = _scalar(
            cursor,
            f"""
            SELECT COUNT(*) FROM {observations}
            WHERE {_blank_expression(("stay_id", "variable", "unit"))}
               OR event_time IS NULL OR value IS NULL
            """,
        )
        missing_event_provenance = _scalar(
            cursor,
            f"""
            SELECT COUNT(*) FROM {events}
            WHERE {_blank_expression(("source_table", "source_code"))}
            """,
        )
        missing_observation_provenance = _scalar(
            cursor,
            f"""
            SELECT COUNT(*) FROM {observations}
            WHERE {_blank_expression(("source_table", "source_code"))}
            """,
        )

        unknown_tasks = _distinct_text(
            cursor,
            f"SELECT DISTINCT task::text FROM {events} WHERE NOT (task = ANY(%s))",
            (list(allowed_tasks),),
        )
        known_variables = tuple(expected_units)
        unknown_variables = _distinct_text(
            cursor,
            f"""
            SELECT DISTINCT variable::text FROM {observations}
            WHERE NOT (variable = ANY(%s))
            """,
            (list(known_variables),),
        )
        cursor.execute(f"SELECT DISTINCT variable::text, unit::text FROM {observations}")
        mismatches: set[str] = set()
        for variable, unit in cursor.fetchall():
            expected = expected_units.get(str(variable))
            if expected is not None and str(unit) != expected:
                mismatches.add(f"{variable}:{unit}->{expected}")
        unit_mismatches = tuple(sorted(mismatches))

        warnings: list[str] = []
        if _scalar(cursor, f"SELECT COUNT(*) FROM {stays} WHERE hospital_id IS NULL") > 0:
            warnings.append("missing_hospital_id")
        if _scalar(cursor, f"SELECT COUNT(*) FROM {stays} WHERE sex IS NULL") > 0:
            warnings.append("missing_sex")
        if _scalar(cursor, f"SELECT COUNT(*) FROM {stays} WHERE age IS NULL") > 0:
            warnings.append("missing_or_nonnumeric_age")
        if event_rows == 0:
            warnings.append("no_event_rows")

    checks = (
        ("duplicate_stay_ids", duplicate_stay_ids),
        ("invalid_icu_intervals", invalid_icu_intervals),
        ("invalid_event_intervals", invalid_event_intervals),
        ("unlinked_event_rows", unlinked_event_rows),
        ("unlinked_observation_rows", unlinked_observation_rows),
        ("observations_before_icu", observations_before_icu),
        ("observations_at_or_after_landmark", observations_at_or_after_landmark),
        ("invalid_observation_values", invalid_observation_values),
        ("missing_stay_core_values", missing_stay_core_values),
        ("missing_event_core_values", missing_event_core_values),
        ("missing_observation_core_values", missing_observation_core_values),
        ("duplicate_observation_rows", duplicate_observation_rows),
        ("missing_event_provenance", missing_event_provenance),
        ("missing_observation_provenance", missing_observation_provenance),
    )
    failures = [name for name, count in checks if count > 0]
    if unknown_tasks:
        failures.append("unknown_tasks")
    if unknown_variables:
        failures.append("unknown_variables")
    if unit_mismatches:
        failures.append("unit_mismatches")
    if observation_rows == 0:
        failures.append("no_observation_rows")

    return CanonicalExtractAudit(
        dataset=dataset,
        source_contract_sha256=mapping_sha256(source_contract),
        feature_contract_sha256=mapping_sha256(asdict(feature_contract)),
        stay_rows=stay_rows,
        event_rows=event_rows,
        observation_rows=observation_rows,
        unique_stays=unique_stays,
        unknown_tasks=unknown_tasks,
        unknown_variables=unknown_variables,
        unit_mismatches=unit_mismatches,
        duplicate_stay_ids=duplicate_stay_ids,
        duplicate_observation_rows=duplicate_observation_rows,
        unlinked_event_rows=unlinked_event_rows,
        unlinked_observation_rows=unlinked_observation_rows,
        observations_before_icu=observations_before_icu,
        observations_at_or_after_landmark=observations_at_or_after_landmark,
        invalid_observation_values=invalid_observation_values,
        invalid_icu_intervals=invalid_icu_intervals,
        invalid_event_intervals=invalid_event_intervals,
        missing_stay_core_values=missing_stay_core_values,
        missing_event_core_values=missing_event_core_values,
        missing_observation_core_values=missing_observation_core_values,
        missing_event_provenance=missing_event_provenance,
        missing_observation_provenance=missing_observation_provenance,
        critical_failures=tuple(dict.fromkeys(failures)),
        warnings=tuple(dict.fromkeys(warnings)),
        ready_for_cohort_build=not failures,
    )


def _check_eicu_mappings(connection: ConnectionLike) -> dict[str, int]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT task, COUNT(*)
            FROM trust_icu_local.eicu_outcome_map
            WHERE status = 'locked' AND classification = 'positive'
            GROUP BY task
            """
        )
        counts = {str(task): int(count) for task, count in cursor.fetchall()}
        missing = [task for task in _TASKS if counts.get(task, 0) == 0]
        if missing:
            raise RuntimeError(
                "eICU execution is blocked; no locked positive mappings for: "
                + ", ".join(missing)
            )
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM trust_icu_local.eicu_feature_map
            WHERE status = 'locked'
            """
        )
        feature_count = int((cursor.fetchone() or (0,))[0])
        if feature_count == 0:
            raise RuntimeError("eICU execution is blocked; no locked local feature mappings.")
    counts["locked_feature_mappings"] = feature_count
    return counts


def _materialize(
    connection: ConnectionLike,
    *,
    relation: str,
    query: str,
) -> None:
    relation = _safe_relation(relation)
    schema = relation.split(".", 1)[0]
    with connection.cursor() as cursor:
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        cursor.execute(f"DROP TABLE IF EXISTS {relation}")
        cursor.execute(f"CREATE TABLE {relation} AS {_strip_final_semicolon(query)}")
    connection.commit()


def _export_relation(
    connection: ConnectionLike,
    relation: str,
    output_path: Path,
) -> ExportArtifact:
    relation = _safe_relation(relation)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    row_count: int
    try:
        with connection.cursor() as cursor:
            row_count = _scalar(cursor, f"SELECT COUNT(*) FROM {relation}")
            with gzip.open(partial, "wb", compresslevel=6) as handle:
                with cursor.copy(
                    f"COPY (SELECT * FROM {relation}) TO STDOUT WITH (FORMAT CSV, HEADER TRUE)"
                ) as copy:
                    for chunk in copy:
                        handle.write(bytes(chunk))
        os.replace(partial, output_path)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return ExportArtifact(
        name=relation.rsplit(".", 1)[1],
        path=str(output_path),
        row_count=row_count,
        sha256=_sha256_file(output_path),
        bytes=output_path.stat().st_size,
    )


def prepare_eicu_mapping_tables(
    *,
    repo_root: str | Path,
    dsn: str,
) -> None:
    """Create empty local eICU review tables without loading or approving mappings."""

    if not dsn.strip():
        raise ValueError("A non-empty PostgreSQL DSN is required.")
    ddl_path = Path(repo_root).resolve() / "sql/eicu/01a_create_local_mapping_tables.sql"
    if not ddl_path.is_file():
        raise FileNotFoundError(f"eICU mapping DDL not found: {ddl_path}")
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "Credentialed execution requires psycopg. Install with pip install -e '.[db]'."
        ) from exc
    connection = psycopg.connect(dsn, application_name="trust-icu-phase0")
    try:
        with connection.cursor() as cursor:
            cursor.execute(ddl_path.read_text(encoding="utf-8"), prepare=False)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def execute_credentialed_run(
    *,
    repo_root: str | Path,
    dataset: str,
    dsn: str,
    output_root: str | Path,
    allow_reviewed_eicu: bool = False,
    overwrite: bool = False,
) -> CredentialedRunReport:
    """Execute one source adapter, audit in PostgreSQL, then export compressed CSV files."""

    if not dsn.strip():
        raise ValueError("A non-empty PostgreSQL DSN is required.")
    root = Path(repo_root).resolve()
    output_base = Path(output_root).expanduser().resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    try:
        output_base.chmod(0o700)
    except OSError:
        pass

    manifest, spec, relations, sql_paths = _read_plan(root, dataset)
    if dataset == "eicu_crd_2_0" and not allow_reviewed_eicu:
        raise RuntimeError(
            "eICU execution requires --allow-reviewed-eicu after local mappings are reviewed."
        )

    run_dir = output_base / dataset
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {run_dir}. Use overwrite only deliberately."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    source_contract = load_source_adapter_contract(
        root / str(manifest["canonical_contract"])
    )
    feature_contract = load_feature_contract(root / str(manifest["feature_contract"]))
    sql_text = {name: path.read_text(encoding="utf-8") for name, path in sql_paths.items()}
    sql_sha = {name: _sha256_bytes(text.encode("utf-8")) for name, text in sql_text.items()}

    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "Credentialed execution requires psycopg. Install with pip install -e '.[db]'."
        ) from exc

    started = datetime.now(UTC)
    exports: tuple[ExportArtifact, ...] = ()
    connection = psycopg.connect(dsn, application_name="trust-icu-phase0")
    try:
        with connection.cursor() as cursor:
            cursor.execute("SHOW server_version")
            server_version = str((cursor.fetchone() or ("unknown",))[0])

        if dataset == "eicu_crd_2_0":
            _check_eicu_mappings(connection)

        _materialize(connection, relation=relations["stays"], query=sql_text["stays"])
        _materialize(connection, relation=relations["events"], query=sql_text["events"])
        _materialize(
            connection,
            relation=relations["observations"],
            query=sql_text["observations"],
        )

        audit = _postgres_audit(
            connection,
            dataset=dataset,
            relations=relations,
            source_contract=source_contract,
            feature_contract=feature_contract,
        )
        audit_path = run_dir / "canonical_audit.json"
        audit_path.write_text(
            json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not audit.ready_for_cohort_build:
            raise RuntimeError(
                "Canonical database audit failed: " + ", ".join(audit.critical_failures)
            )

        artifacts = []
        for name in ("stays", "events", "observations"):
            artifacts.append(
                _export_relation(
                    connection,
                    relations[name],
                    run_dir / f"{dataset}_{name}.csv.gz",
                )
            )
        exports = tuple(artifacts)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    completed = datetime.now(UTC)
    provisional = CredentialedRunReport(
        dataset=dataset,
        started_at_utc=started.isoformat(),
        completed_at_utc=completed.isoformat(),
        manifest_version=str(manifest["manifest_version"]),
        adapter_status=str(spec["status"]),
        database_server_version=server_version,
        relations=relations,
        sql_sha256=sql_sha,
        audit=audit.to_dict(),
        exports=exports,
    )
    payload = provisional.to_dict()
    payload["report_sha256"] = ""
    report_hash = _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    report = CredentialedRunReport(
        **{**provisional.__dict__, "report_sha256": report_hash}
    )
    (run_dir / "credentialed_run_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
