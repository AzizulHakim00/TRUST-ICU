-- TRUST-ICU Phase 0: MIMIC-IV v3.1 canonical organ-support events.
-- PostgreSQL syntax.
-- Prerequisite: materialize sql/mimic/02_base_landmark_cohort.sql as
-- trust_icu_work.mimic_stays before running this query.
--
-- This export deliberately includes events overlapping the observation window so the
-- task-specific cohort code can exclude prevalent support. It includes no rows after the
-- task-specific follow-up end.

WITH cohort AS (
    SELECT
        stay_id::bigint AS stay_id,
        icu_admit_time,
        followup_end_time
    FROM trust_icu_work.mimic_stays
),
ventilation_events AS (
    SELECT
        c.stay_id::text AS stay_id,
        'invasive_mechanical_ventilation'::text AS task,
        v.starttime AS start_time,
        v.endtime AS end_time,
        'mimiciv_derived.ventilation'::text AS source_table,
        v.ventilation_status::text AS source_code
    FROM cohort AS c
    INNER JOIN mimiciv_derived.ventilation AS v
        ON c.stay_id = v.stay_id
    WHERE v.ventilation_status = 'InvasiveVent'
      AND v.starttime < c.followup_end_time
      AND COALESCE(v.endtime, c.followup_end_time) > c.icu_admit_time
),
vasopressor_events AS (
    SELECT
        c.stay_id::text AS stay_id,
        'vasopressor_initiation'::text AS task,
        vaso.starttime AS start_time,
        vaso.endtime AS end_time,
        vaso.source_table,
        vaso.agent AS source_code
    FROM cohort AS c
    INNER JOIN (
        SELECT stay_id, starttime, endtime, vaso_rate,
               'mimiciv_derived.norepinephrine'::text AS source_table,
               'norepinephrine'::text AS agent
        FROM mimiciv_derived.norepinephrine
        UNION ALL
        SELECT stay_id, starttime, endtime, vaso_rate,
               'mimiciv_derived.epinephrine', 'epinephrine'
        FROM mimiciv_derived.epinephrine
        UNION ALL
        SELECT stay_id, starttime, endtime, vaso_rate,
               'mimiciv_derived.vasopressin', 'vasopressin'
        FROM mimiciv_derived.vasopressin
        UNION ALL
        SELECT stay_id, starttime, endtime, vaso_rate,
               'mimiciv_derived.phenylephrine', 'phenylephrine'
        FROM mimiciv_derived.phenylephrine
        UNION ALL
        SELECT stay_id, starttime, endtime, vaso_rate,
               'mimiciv_derived.dopamine', 'dopamine'
        FROM mimiciv_derived.dopamine
    ) AS vaso
        ON c.stay_id = vaso.stay_id
    WHERE vaso.vaso_rate IS NOT NULL
      AND vaso.vaso_rate > 0
      AND vaso.starttime < c.followup_end_time
      AND COALESCE(vaso.endtime, c.followup_end_time) > c.icu_admit_time
),
rrt_events AS (
    SELECT
        c.stay_id::text AS stay_id,
        'renal_replacement_therapy'::text AS task,
        rrt.charttime AS start_time,
        NULL::timestamp AS end_time,
        'mimiciv_derived.rrt'::text AS source_table,
        'dialysis_active=1'::text AS source_code
    FROM cohort AS c
    INNER JOIN mimiciv_derived.rrt AS rrt
        ON c.stay_id = rrt.stay_id
    WHERE rrt.dialysis_active = 1
      AND rrt.charttime >= c.icu_admit_time
      AND rrt.charttime < c.followup_end_time
)
SELECT * FROM ventilation_events
UNION ALL
SELECT * FROM vasopressor_events
UNION ALL
SELECT * FROM rrt_events
ORDER BY stay_id, task, start_time, source_table, source_code;
