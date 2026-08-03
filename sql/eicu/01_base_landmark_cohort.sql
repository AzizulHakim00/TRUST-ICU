-- TRUST-ICU Phase 0: eICU-CRD v2.0 base landmark cohort.
-- PostgreSQL syntax. eICU records time as minutes from ICU admission; pseudo timestamps are
-- created only to provide a canonical relative-time interface. They are not calendar dates.
-- One first eligible ICU unit stay is retained per patientHealthSystemStayID. Cross-admission
-- chronology for the same uniquepid is deliberately not inferred.

WITH normalized AS (
    SELECT
        p.uniquepid::text AS patient_id,
        (p.hospitalid::text || ':' || p.patienthealthsystemstayid::text) AS hospital_admission_id,
        p.patientunitstayid::text AS stay_id,
        CASE
            WHEN BTRIM(p.age) = '> 89' THEN 90.0
            WHEN BTRIM(p.age) ~ '^[0-9]+([.][0-9]+)?$' THEN BTRIM(p.age)::double precision
            ELSE NULL
        END AS age,
        p.gender AS sex,
        p.hospitalid::text AS hospital_id,
        p.unittype AS unit_type,
        p.unitadmitsource AS admission_source,
        p.unitvisitnumber,
        TIMESTAMP '2000-01-01 00:00:00' AS icu_admit_time,
        TIMESTAMP '2000-01-01 00:00:00'
            + p.unitdischargeoffset * INTERVAL '1 minute' AS icu_discharge_time,
        CASE
            WHEN LOWER(COALESCE(p.unitdischargestatus, '')) = 'expired'
            THEN TIMESTAMP '2000-01-01 00:00:00'
                + p.unitdischargeoffset * INTERVAL '1 minute'
            ELSE NULL
        END AS death_time,
        TIMESTAMP '2000-01-01 06:00:00' AS landmark_time,
        TIMESTAMP '2000-01-01 18:00:00' AS administrative_end_time
    FROM eicu_crd.patient AS p
    WHERE p.unitdischargeoffset IS NOT NULL
      AND p.unitdischargeoffset > 0
),
eligible_at_landmark AS (
    SELECT *
    FROM normalized
    WHERE age >= 18
      AND icu_discharge_time > landmark_time
      AND (death_time IS NULL OR death_time > landmark_time)
),
first_eligible_unit_stay AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY hospital_admission_id
            ORDER BY unitvisitnumber, stay_id
        ) AS eligible_unit_rank
    FROM eligible_at_landmark
)
SELECT
    patient_id,
    hospital_admission_id,
    stay_id,
    age,
    sex,
    'eICU-CRD-2.0'::text AS dataset_id,
    hospital_id,
    unit_type,
    admission_source,
    icu_admit_time,
    landmark_time,
    LEAST(
        icu_discharge_time,
        administrative_end_time,
        COALESCE(death_time, administrative_end_time)
    ) AS followup_end_time,
    administrative_end_time,
    icu_discharge_time,
    death_time,
    (
        death_time IS NOT NULL
        AND death_time > landmark_time
        AND death_time < administrative_end_time
    ) AS death_before_administrative_end
FROM first_eligible_unit_stay
WHERE eligible_unit_rank = 1;
