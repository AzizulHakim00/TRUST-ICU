from pathlib import Path

import pytest

from trust_icu.credentialed_runner import (
    _relation_set,
    _safe_relation,
    _strip_final_semicolon,
    build_dry_run_plan,
    execute_credentialed_run,
)


ROOT = Path(__file__).resolve().parents[1]


def test_repository_dry_run_plan_has_pinned_mimic_sql() -> None:
    plan = build_dry_run_plan(ROOT, "mimic_iv_3_1")
    assert plan["dataset"] == "mimic_iv_3_1"
    assert plan["adapter_status"] == "implemented_pending_credentialed_execution"
    assert set(plan["sql_files"]) == {"stays", "events", "observations"}
    assert all(len(value) == 64 for value in plan["sql_sha256"].values())
    assert plan["local_mappings_required"] is False


def test_repository_dry_run_plan_marks_eicu_mapping_requirement() -> None:
    plan = build_dry_run_plan(ROOT, "eicu_crd_2_0")
    assert plan["local_mappings_required"] is True
    assert plan["adapter_status"] == "templates_require_reviewed_local_mappings"


@pytest.mark.parametrize(
    "value",
    ["trust_icu_work.mimic_stays", "schema_name.table_1"],
)
def test_safe_relation_accepts_manifest_style_names(value: str) -> None:
    assert _safe_relation(value) == value


@pytest.mark.parametrize(
    "value",
    ["mimic_stays", "public.table;drop table x", "../schema.table", "A.b"],
)
def test_safe_relation_rejects_unsafe_names(value: str) -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        _safe_relation(value)


def test_relation_set_derives_canonical_names() -> None:
    assert _relation_set("mimic_iv_3_1", "trust_icu_work.mimic_stays") == {
        "stays": "trust_icu_work.mimic_stays",
        "events": "trust_icu_work.mimic_events",
        "observations": "trust_icu_work.mimic_observations",
    }


def test_strip_final_semicolon_preserves_query_body() -> None:
    assert _strip_final_semicolon(" SELECT 1;; \n") == "SELECT 1"


def test_credentialed_run_rejects_empty_dsn_before_database_access(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="DSN"):
        execute_credentialed_run(
            repo_root=ROOT,
            dataset="mimic_iv_3_1",
            dsn="",
            output_root=tmp_path,
        )


def test_eicu_requires_explicit_review_flag_before_database_access(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="allow-reviewed-eicu"):
        execute_credentialed_run(
            repo_root=ROOT,
            dataset="eicu_crd_2_0",
            dsn="postgresql://not-used",
            output_root=tmp_path,
        )
