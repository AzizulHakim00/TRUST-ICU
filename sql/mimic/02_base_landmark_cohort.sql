-- TRUST-ICU Phase 0: MIMIC-IV v3.1 base landmark cohort.
-- PostgreSQL syntax. This query creates no outcome labels and uses no post-landmark predictors.
-- Required upstream relations: mimiciv_icu.icustays, mimiciv_hosp.admissions,
-- mimiciv_hosp.patients, mimiciv_derived.age.

WITH candidate_stays AS (
    SELECT
        ie.subject_id::text AS patient_id,
        ie.hadm_id::text AS hospital_admission_id,
        ie.stay_id::text AS stay_id,
        age.age::double precision AS age,
        pat.gender::text AS sex,
        ie.intime AS icu_admit_time,
        ie.outtime AS icu_discharge_time,
        adm.deathtime AS death_time,
        ie.first_careunit AS unit_type,
        adm.admission_location AS admission_source,
        ie.intime + INTERVAL '6 hour' AS landmark_time,
        ie.intime + INTERVAL '18 hour' AS administrative_end_time
    FROM mimiciv_icu.icustays AS ie
    INNER JOIN mimiciv_hosp.admissions AS adm
        ON ie.hadm_id = adm.hadm_id
    INNER JOIN mimiciv_hosp.patients AS pat
        ON ie.subject_id = pat.subject_id
    INNER JOIN mimiciv_derived.age AS age
        ON ie.subject_id = age.subject_id
       AND ie.hadm_id = age.hadm_id
    WHERE age.age >= 18
),
eligible_at_landmark AS (
    SELECT *
    FROM candidate_stays
    WHERE icu_discharge_time > landmark_time
      AND (death_time IS NULL OR death_time > landmark_time)
),
first_eligible_unit_stay AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY hospital_admission_id
            ORDER BY icu_admit_time, stay_id
        ) AS eligible_unit_rank
    FROM eligible_at_landmark
)
SELECT
    patient_id,
    hospital_admission_id,
    stay_id,
    age,
    sex,
    'mimic_iv_3_1'::text AS dataset_id,
    'BIDMC'::text AS hospital_id,
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
