-- TRUST-ICU Phase 0: MIMIC-IV v3.1 canonical observations in [0 h, 6 h).
-- PostgreSQL syntax.
-- Prerequisite: materialize sql/mimic/02_base_landmark_cohort.sql as
-- trust_icu_work.mimic_stays before running this query.
--
-- The query relies on maintained MIMIC-IV derived concepts pinned in
-- schemas/source_adapter_manifest.yaml. Values are exported in the canonical units declared
-- by schemas/phase0_features.yaml. Physiological-range filtering remains an independent audit.

WITH cohort AS (
    SELECT
        stay_id::bigint AS stay_id,
        hospital_admission_id::bigint AS hadm_id,
        icu_admit_time,
        landmark_time
    FROM trust_icu_work.mimic_stays
),
vital_observations AS (
    SELECT
        c.stay_id::text AS stay_id,
        x.variable,
        v.charttime AS event_time,
        x.value::double precision AS value,
        x.unit,
        'mimiciv_derived.vitalsign'::text AS source_table,
        x.source_code
    FROM cohort AS c
    INNER JOIN mimiciv_derived.vitalsign AS v
        ON c.stay_id = v.stay_id
       AND v.charttime >= c.icu_admit_time
       AND v.charttime < c.landmark_time
    CROSS JOIN LATERAL (
        VALUES
            ('heart_rate', v.heart_rate, 'beats/min', 'heart_rate'),
            ('respiratory_rate', v.resp_rate, 'breaths/min', 'resp_rate'),
            ('spo2', v.spo2, 'percent', 'spo2'),
            ('systolic_bp', v.sbp, 'mmHg', 'sbp'),
            ('diastolic_bp', v.dbp, 'mmHg', 'dbp'),
            ('mean_arterial_pressure', v.mbp, 'mmHg', 'mbp'),
            ('temperature_c', v.temperature, 'degC', 'temperature'),
            ('glucose', v.glucose, 'mg/dL', 'glucose')
    ) AS x(variable, value, unit, source_code)
    WHERE x.value IS NOT NULL
),
chemistry_observations AS (
    SELECT
        c.stay_id::text AS stay_id,
        x.variable,
        ch.charttime AS event_time,
        x.value::double precision AS value,
        x.unit,
        'mimiciv_derived.chemistry'::text AS source_table,
        x.source_code
    FROM cohort AS c
    INNER JOIN mimiciv_derived.chemistry AS ch
        ON c.hadm_id = ch.hadm_id
       AND ch.charttime >= c.icu_admit_time
       AND ch.charttime < c.landmark_time
    CROSS JOIN LATERAL (
        VALUES
            ('bicarbonate', ch.bicarbonate, 'mmol/L', 'bicarbonate'),
            ('bun', ch.bun, 'mg/dL', 'bun'),
            ('chloride', ch.chloride, 'mmol/L', 'chloride'),
            ('creatinine', ch.creatinine, 'mg/dL', 'creatinine'),
            ('glucose', ch.glucose, 'mg/dL', 'glucose'),
            ('sodium', ch.sodium, 'mmol/L', 'sodium'),
            ('potassium', ch.potassium, 'mmol/L', 'potassium')
    ) AS x(variable, value, unit, source_code)
    WHERE x.value IS NOT NULL
),
cbc_observations AS (
    SELECT
        c.stay_id::text AS stay_id,
        x.variable,
        cbc.charttime AS event_time,
        x.value::double precision AS value,
        x.unit,
        'mimiciv_derived.complete_blood_count'::text AS source_table,
        x.source_code
    FROM cohort AS c
    INNER JOIN mimiciv_derived.complete_blood_count AS cbc
        ON c.hadm_id = cbc.hadm_id
       AND cbc.charttime >= c.icu_admit_time
       AND cbc.charttime < c.landmark_time
    CROSS JOIN LATERAL (
        VALUES
            ('hemoglobin', cbc.hemoglobin, 'g/dL', 'hemoglobin'),
            ('platelet_count', cbc.platelet, '10^9/L', 'platelet'),
            ('wbc', cbc.wbc, '10^9/L', 'wbc')
    ) AS x(variable, value, unit, source_code)
    WHERE x.value IS NOT NULL
),
enzyme_observations AS (
    SELECT
        c.stay_id::text AS stay_id,
        'bilirubin_total'::text AS variable,
        enz.charttime AS event_time,
        enz.bilirubin_total::double precision AS value,
        'mg/dL'::text AS unit,
        'mimiciv_derived.enzyme'::text AS source_table,
        'bilirubin_total'::text AS source_code
    FROM cohort AS c
    INNER JOIN mimiciv_derived.enzyme AS enz
        ON c.hadm_id = enz.hadm_id
       AND enz.charttime >= c.icu_admit_time
       AND enz.charttime < c.landmark_time
    WHERE enz.bilirubin_total IS NOT NULL
),
blood_gas_observations AS (
    SELECT
        c.stay_id::text AS stay_id,
        x.variable,
        bg.charttime AS event_time,
        x.value::double precision AS value,
        x.unit,
        'mimiciv_derived.bg'::text AS source_table,
        x.source_code
    FROM cohort AS c
    INNER JOIN mimiciv_derived.bg AS bg
        ON c.hadm_id = bg.hadm_id
       AND bg.charttime >= c.icu_admit_time
       AND bg.charttime < c.landmark_time
    CROSS JOIN LATERAL (
        VALUES
            ('lactate', bg.lactate, 'mmol/L', 'lactate'),
            ('ph', bg.ph, 'pH', 'ph'),
            ('pao2', bg.po2, 'mmHg', 'po2'),
            (
                'fio2',
                COALESCE(bg.fio2, bg.fio2_chartevents) / 100.0,
                'fraction',
                'coalesced_fio2_percent_to_fraction'
            )
    ) AS x(variable, value, unit, source_code)
    WHERE x.value IS NOT NULL
),
gcs_observations AS (
    SELECT
        c.stay_id::text AS stay_id,
        'gcs_total'::text AS variable,
        gcs.charttime AS event_time,
        gcs.gcs::double precision AS value,
        'score'::text AS unit,
        'mimiciv_derived.gcs'::text AS source_table,
        'gcs'::text AS source_code
    FROM cohort AS c
    INNER JOIN mimiciv_derived.gcs AS gcs
        ON c.stay_id = gcs.stay_id
       AND gcs.charttime >= c.icu_admit_time
       AND gcs.charttime < c.landmark_time
    WHERE gcs.gcs IS NOT NULL
),
urine_observations AS (
    SELECT
        c.stay_id::text AS stay_id,
        'urine_output_ml'::text AS variable,
        uo.charttime AS event_time,
        uo.urineoutput::double precision AS value,
        'mL'::text AS unit,
        'mimiciv_derived.urine_output'::text AS source_table,
        'urineoutput'::text AS source_code
    FROM cohort AS c
    INNER JOIN mimiciv_derived.urine_output AS uo
        ON c.stay_id = uo.stay_id
       AND uo.charttime >= c.icu_admit_time
       AND uo.charttime < c.landmark_time
    WHERE uo.urineoutput IS NOT NULL
)
SELECT * FROM vital_observations
UNION ALL
SELECT * FROM chemistry_observations
UNION ALL
SELECT * FROM cbc_observations
UNION ALL
SELECT * FROM enzyme_observations
UNION ALL
SELECT * FROM blood_gas_observations
UNION ALL
SELECT * FROM gcs_observations
UNION ALL
SELECT * FROM urine_observations
ORDER BY stay_id, event_time, variable, source_table, source_code;
