-- TRUST-ICU Phase 0: eICU-CRD v2.0 canonical observation template.
-- PostgreSQL syntax.
-- Prerequisites:
--   1. materialize sql/eicu/01_base_landmark_cohort.sql as trust_icu_work.eicu_stays;
--   2. run sql/eicu/01a_create_local_mapping_tables.sql;
--   3. load only locally reviewed mappings and set their status to 'locked'.
--
-- Direct vital columns below are documented by the official eICU schema. Hospital-specific
-- laboratory names and units are exported only through locked local mappings.

WITH cohort AS (
    SELECT
        stay_id::bigint AS stay_id,
        icu_admit_time,
        landmark_time
    FROM trust_icu_work.eicu_stays
),
periodic_vitals AS (
    SELECT
        c.stay_id::text AS stay_id,
        x.variable,
        c.icu_admit_time + vp.observationoffset * INTERVAL '1 minute' AS event_time,
        x.value::double precision AS value,
        x.unit,
        'eicu_crd.vitalperiodic'::text AS source_table,
        x.source_code
    FROM cohort AS c
    INNER JOIN eicu_crd.vitalperiodic AS vp
        ON c.stay_id = vp.patientunitstayid
    CROSS JOIN LATERAL (
        VALUES
            ('heart_rate', vp.heartrate, 'beats/min', 'heartRate'),
            ('respiratory_rate', vp.respiration, 'breaths/min', 'respiration'),
            ('spo2', vp.sao2, 'percent', 'saO2'),
            ('systolic_bp', vp.systemicsystolic, 'mmHg', 'systemicSystolic'),
            ('diastolic_bp', vp.systemicdiastolic, 'mmHg', 'systemicDiastolic'),
            ('mean_arterial_pressure', vp.systemicmean, 'mmHg', 'systemicMean'),
            ('temperature_c', vp.temperature, 'degC', 'temperature')
    ) AS x(variable, value, unit, source_code)
    WHERE vp.observationoffset >= 0
      AND vp.observationoffset < 360
      AND x.value IS NOT NULL
),
aperiodic_vitals AS (
    SELECT
        c.stay_id::text AS stay_id,
        x.variable,
        c.icu_admit_time + va.observationoffset * INTERVAL '1 minute' AS event_time,
        x.value::double precision AS value,
        x.unit,
        'eicu_crd.vitalaperiodic'::text AS source_table,
        x.source_code
    FROM cohort AS c
    INNER JOIN eicu_crd.vitalaperiodic AS va
        ON c.stay_id = va.patientunitstayid
    CROSS JOIN LATERAL (
        VALUES
            ('systolic_bp', va.noninvasivesystolic, 'mmHg', 'nonInvasiveSystolic'),
            ('diastolic_bp', va.noninvasivediastolic, 'mmHg', 'nonInvasiveDiastolic'),
            ('mean_arterial_pressure', va.noninvasivemean, 'mmHg', 'nonInvasiveMean')
    ) AS x(variable, value, unit, source_code)
    WHERE va.observationoffset >= 0
      AND va.observationoffset < 360
      AND x.value IS NOT NULL
),
locked_lab_map AS (
    SELECT
        LOWER(BTRIM(source_code_normalized)) AS source_code_normalized,
        LOWER(BTRIM(source_unit_normalized)) AS source_unit_normalized,
        canonical_variable,
        canonical_unit,
        multiplier,
        offset
    FROM trust_icu_local.eicu_feature_map
    WHERE source_table = 'lab'
      AND status = 'locked'
),
lab_observations AS (
    SELECT
        c.stay_id::text AS stay_id,
        m.canonical_variable AS variable,
        c.icu_admit_time + l.labresultoffset * INTERVAL '1 minute' AS event_time,
        (l.labresult::double precision * m.multiplier + m.offset) AS value,
        m.canonical_unit AS unit,
        'eicu_crd.lab'::text AS source_table,
        CONCAT(
            LOWER(BTRIM(l.labname)),
            '|',
            LOWER(BTRIM(COALESCE(l.labmeasurenamesystem, l.labmeasurenameinterface, '')))
        ) AS source_code
    FROM cohort AS c
    INNER JOIN eicu_crd.lab AS l
        ON c.stay_id = l.patientunitstayid
    INNER JOIN locked_lab_map AS m
        ON LOWER(BTRIM(l.labname)) = m.source_code_normalized
       AND LOWER(BTRIM(COALESCE(l.labmeasurenamesystem, l.labmeasurenameinterface, '')))
            = m.source_unit_normalized
    WHERE l.labresultoffset >= 0
      AND l.labresultoffset < 360
      AND l.labresult IS NOT NULL
)
SELECT * FROM periodic_vitals
UNION ALL
SELECT * FROM aperiodic_vitals
UNION ALL
SELECT * FROM lab_observations
ORDER BY stay_id, event_time, variable, source_table, source_code;
