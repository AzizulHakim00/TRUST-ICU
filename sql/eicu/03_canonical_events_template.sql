-- TRUST-ICU Phase 0: eICU-CRD v2.0 canonical organ-support event template.
-- PostgreSQL syntax.
-- Prerequisites:
--   1. materialize sql/eicu/01_base_landmark_cohort.sql as trust_icu_work.eicu_stays;
--   2. run sql/eicu/01a_create_local_mapping_tables.sql;
--   3. load clinically reviewed exact vocabularies with status='locked'.
--
-- This query intentionally emits no event from an unreviewed keyword match.

WITH cohort AS (
    SELECT
        stay_id::bigint AS stay_id,
        icu_admit_time,
        followup_end_time
    FROM trust_icu_work.eicu_stays
),
locked_outcomes AS (
    SELECT
        task,
        source_table,
        LOWER(BTRIM(source_code_normalized)) AS source_code_normalized,
        classification
    FROM trust_icu_local.eicu_outcome_map
    WHERE status = 'locked'
),
ventilation_events AS (
    SELECT
        c.stay_id::text AS stay_id,
        'invasive_mechanical_ventilation'::text AS task,
        c.icu_admit_time + rc.ventstartoffset * INTERVAL '1 minute' AS start_time,
        CASE
            WHEN rc.ventendoffset IS NULL THEN NULL
            ELSE c.icu_admit_time + rc.ventendoffset * INTERVAL '1 minute'
        END AS end_time,
        'eicu_crd.respiratorycare'::text AS source_table,
        LOWER(BTRIM(rc.airwaytype)) AS source_code
    FROM cohort AS c
    INNER JOIN eicu_crd.respiratorycare AS rc
        ON c.stay_id = rc.patientunitstayid
    INNER JOIN locked_outcomes AS m
        ON m.task = 'invasive_mechanical_ventilation'
       AND m.source_table = 'respiratoryCare:airwayType'
       AND m.classification = 'positive'
       AND LOWER(BTRIM(rc.airwaytype)) = m.source_code_normalized
    WHERE rc.ventstartoffset IS NOT NULL
      AND c.icu_admit_time + rc.ventstartoffset * INTERVAL '1 minute' < c.followup_end_time
      AND (
          rc.ventendoffset IS NULL
          OR c.icu_admit_time + rc.ventendoffset * INTERVAL '1 minute' > c.icu_admit_time
      )
),
vasopressor_events AS (
    SELECT
        c.stay_id::text AS stay_id,
        'vasopressor_initiation'::text AS task,
        c.icu_admit_time + id.infusionoffset * INTERVAL '1 minute' AS start_time,
        NULL::timestamp AS end_time,
        'eicu_crd.infusiondrug'::text AS source_table,
        LOWER(BTRIM(id.drugname)) AS source_code
    FROM cohort AS c
    INNER JOIN eicu_crd.infusiondrug AS id
        ON c.stay_id = id.patientunitstayid
    INNER JOIN locked_outcomes AS m
        ON m.task = 'vasopressor_initiation'
       AND m.source_table = 'infusionDrug:drugName'
       AND m.classification = 'positive'
       AND POSITION(m.source_code_normalized IN LOWER(BTRIM(id.drugname))) > 0
    WHERE id.infusionoffset >= 0
      AND c.icu_admit_time + id.infusionoffset * INTERVAL '1 minute' < c.followup_end_time
      AND (
          (BTRIM(COALESCE(id.drugrate, '')) ~ '^[+]?[0-9]+([.][0-9]+)?$'
           AND BTRIM(id.drugrate)::double precision > 0)
          OR
          (BTRIM(COALESCE(id.infusionrate, '')) ~ '^[+]?[0-9]+([.][0-9]+)?$'
           AND BTRIM(id.infusionrate)::double precision > 0)
      )
),
rrt_events AS (
    SELECT
        c.stay_id::text AS stay_id,
        'renal_replacement_therapy'::text AS task,
        c.icu_admit_time + t.treatmentoffset * INTERVAL '1 minute' AS start_time,
        NULL::timestamp AS end_time,
        'eicu_crd.treatment'::text AS source_table,
        LOWER(BTRIM(t.treatmentstring)) AS source_code
    FROM cohort AS c
    INNER JOIN eicu_crd.treatment AS t
        ON c.stay_id = t.patientunitstayid
    INNER JOIN locked_outcomes AS m
        ON m.task = 'renal_replacement_therapy'
       AND m.source_table = 'treatment:treatmentString'
       AND m.classification = 'positive'
       AND LOWER(BTRIM(t.treatmentstring)) = m.source_code_normalized
    WHERE t.treatmentoffset >= 0
      AND c.icu_admit_time + t.treatmentoffset * INTERVAL '1 minute' < c.followup_end_time
)
SELECT * FROM ventilation_events
UNION ALL
SELECT * FROM vasopressor_events
UNION ALL
SELECT * FROM rrt_events
ORDER BY stay_id, task, start_time, source_table, source_code;
