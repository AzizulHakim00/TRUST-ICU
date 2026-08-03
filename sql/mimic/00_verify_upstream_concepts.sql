-- TRUST-ICU Phase 0: MIMIC-IV v3.1 upstream concept verification.
-- Dialect: Google BigQuery Standard SQL.
-- This script reads aggregate metadata only and creates no patient-level table.
-- Upstream concept definitions are pinned in schemas/outcome_contracts.yaml.

DECLARE expected_mimic_version STRING DEFAULT '3.1';

SELECT
  expected_mimic_version AS expected_mimic_version,
  CURRENT_TIMESTAMP() AS audit_time,
  'physionet-data.mimiciv_v3_1_icu.icustays' AS raw_stay_relation,
  'physionet-data.mimiciv_derived' AS derived_dataset;

-- Required raw ICU relation.
SELECT
  COUNT(*) AS icu_stays,
  COUNT(DISTINCT subject_id) AS subjects,
  COUNT(DISTINCT hadm_id) AS admissions,
  MIN(intime) AS earliest_intime,
  MAX(intime) AS latest_intime
FROM `physionet-data.mimiciv_v3_1_icu.icustays`;

-- Invasive mechanical ventilation concept.
SELECT
  COUNT(*) AS invasive_ventilation_intervals,
  COUNT(DISTINCT stay_id) AS stays_with_invasive_ventilation,
  COUNTIF(starttime IS NULL OR endtime IS NULL) AS null_time_intervals,
  COUNTIF(endtime < starttime) AS reversed_intervals
FROM `physionet-data.mimiciv_derived.ventilation`
WHERE ventilation_status = 'InvasiveVent';

-- Vasopressor source views. Each row should represent a documented infusion interval.
WITH vaso AS (
  SELECT 'norepinephrine' AS agent, stay_id, starttime, endtime, vaso_rate
  FROM `physionet-data.mimiciv_derived.norepinephrine`
  UNION ALL
  SELECT 'epinephrine', stay_id, starttime, endtime, vaso_rate
  FROM `physionet-data.mimiciv_derived.epinephrine`
  UNION ALL
  SELECT 'vasopressin', stay_id, starttime, endtime, vaso_rate
  FROM `physionet-data.mimiciv_derived.vasopressin`
  UNION ALL
  SELECT 'phenylephrine', stay_id, starttime, endtime, vaso_rate
  FROM `physionet-data.mimiciv_derived.phenylephrine`
  UNION ALL
  SELECT 'dopamine', stay_id, starttime, endtime, vaso_rate
  FROM `physionet-data.mimiciv_derived.dopamine`
)
SELECT
  agent,
  COUNT(*) AS intervals,
  COUNT(DISTINCT stay_id) AS stays,
  COUNTIF(starttime IS NULL OR endtime IS NULL) AS null_time_intervals,
  COUNTIF(endtime < starttime) AS reversed_intervals,
  COUNTIF(vaso_rate IS NULL) AS null_rate_intervals,
  COUNTIF(COALESCE(vaso_rate, 0) <= 0) AS nonpositive_rate_intervals
FROM vaso
GROUP BY agent
ORDER BY agent;

-- Active renal-replacement therapy evidence. Catheter-only evidence is intentionally excluded.
SELECT
  COUNT(*) AS active_rrt_rows,
  COUNT(DISTINCT stay_id) AS stays_with_active_rrt,
  COUNTIF(charttime IS NULL) AS null_charttime_rows,
  ARRAY_AGG(DISTINCT dialysis_type IGNORE NULLS ORDER BY dialysis_type) AS observed_modalities
FROM `physionet-data.mimiciv_derived.rrt`
WHERE dialysis_active = 1;
