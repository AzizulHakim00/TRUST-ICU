from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_canonical_dataset_ids_match_contract() -> None:
    mimic = _read("sql/mimic/02_base_landmark_cohort.sql")
    eicu = _read("sql/eicu/01_base_landmark_cohort.sql")
    assert "'mimic_iv_3_1'::text AS dataset_id" in mimic
    assert "'eicu_crd_2_0'::text AS dataset_id" in eicu
    assert " AS sex" in mimic
    assert " AS sex" in eicu


def test_mimic_observations_are_strictly_pre_landmark() -> None:
    sql = _read("sql/mimic/04_canonical_observations.sql")
    assert sql.count("< c.landmark_time") >= 7
    assert "COALESCE(bg.fio2, bg.fio2_chartevents) / 100.0" in sql
    for column in ("source_table", "source_code", "event_time", "variable", "unit"):
        assert column in sql


def test_mimic_event_adapter_contains_only_prespecified_tasks() -> None:
    sql = _read("sql/mimic/03_canonical_events.sql")
    for task in (
        "invasive_mechanical_ventilation",
        "vasopressor_initiation",
        "renal_replacement_therapy",
    ):
        assert task in sql
    for agent in (
        "norepinephrine",
        "epinephrine",
        "vasopressin",
        "phenylephrine",
        "dopamine",
    ):
        assert agent in sql
    assert "vaso.vaso_rate > 0" in sql


def test_eicu_templates_require_locked_local_mappings() -> None:
    observations = _read("sql/eicu/02_canonical_observations_template.sql")
    events = _read("sql/eicu/03_canonical_events_template.sql")
    assert "status = 'locked'" in observations
    assert "status = 'locked'" in events
    assert "pending_local_review" not in observations
    assert "pending_local_review" not in events


def test_eicu_observations_use_locked_six_hour_offsets() -> None:
    sql = _read("sql/eicu/02_canonical_observations_template.sql")
    assert sql.count("observationoffset < 360") == 2
    assert "labresultoffset < 360" in sql
    assert "observationoffset <= 360" not in sql
    assert "labresultoffset <= 360" not in sql
