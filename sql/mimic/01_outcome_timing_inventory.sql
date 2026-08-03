-- TRUST-ICU Phase 0: MIMIC-IV event timing inventory.
-- Dialect: Google BigQuery Standard SQL.
-- IMPORTANT: query output is patient-level credentialed data. Do not export it to GitHub.
-- This is an audit inventory, not a final modelling cohort.

DECLARE observation_end_hours INT64 DEFAULT 6;
DECLARE prediction_end_hours INT64 DEFAULT 18;

WITH stays AS (
  SELECT
    stay_id,
    subject_id,
    hadm_id,
    intime,
    outtime,
    DATETIME_ADD(intime, INTERVAL observation_end_hours HOUR) AS observation_end,
    DATETIME_ADD(intime, INTERVAL prediction_end_hours HOUR) AS prediction_end
  FROM `physionet-data.mimiciv_v3_1_icu.icustays`
),
vent_events AS (
  SELECT stay_id, starttime, endtime
  FROM `physionet-data.mimiciv_derived.ventilation`
  WHERE ventilation_status = 'InvasiveVent'
    AND starttime IS NOT NULL
    AND endtime IS NOT NULL
    AND endtime >= starttime
),
vasopressor_events AS (
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
),
vasopressor_valid AS (
  SELECT agent, stay_id, starttime, endtime
  FROM vasopressor_events
  WHERE starttime IS NOT NULL
    AND endtime IS NOT NULL
    AND endtime >= starttime
    AND COALESCE(vaso_rate, 0) > 0
),
rrt_events AS (
  SELECT stay_id, charttime, dialysis_type
  FROM `physionet-data.mimiciv_derived.rrt`
  WHERE dialysis_active = 1
    AND charttime IS NOT NULL
),
vent AS (
  SELECT
    s.stay_id,
    LOGICAL_OR(v.starttime < s.observation_end AND v.endtime > s.intime)
      AS active_in_observation,
    MIN(IF(
      v.starttime >= s.observation_end AND v.starttime < s.prediction_end,
      v.starttime,
      NULL
    )) AS first_start_in_prediction,
    COUNTIF(v.starttime >= s.observation_end AND v.starttime < s.prediction_end)
      AS starts_in_prediction
  FROM stays AS s
  LEFT JOIN vent_events AS v USING (stay_id)
  GROUP BY s.stay_id
),
vaso AS (
  SELECT
    s.stay_id,
    LOGICAL_OR(v.starttime < s.observation_end AND v.endtime > s.intime)
      AS active_in_observation,
    MIN(IF(
      v.starttime >= s.observation_end AND v.starttime < s.prediction_end,
      v.starttime,
      NULL
    )) AS first_start_in_prediction,
    COUNTIF(v.starttime >= s.observation_end AND v.starttime < s.prediction_end)
      AS starts_in_prediction,
    ARRAY_AGG(
      DISTINCT IF(
        v.starttime >= s.observation_end AND v.starttime < s.prediction_end,
        v.agent,
        NULL
      ) IGNORE NULLS
      ORDER BY IF(
        v.starttime >= s.observation_end AND v.starttime < s.prediction_end,
        v.agent,
        NULL
      )
    ) AS agents_in_prediction
  FROM stays AS s
  LEFT JOIN vasopressor_valid AS v USING (stay_id)
  GROUP BY s.stay_id
),
rrt AS (
  SELECT
    s.stay_id,
    LOGICAL_OR(r.charttime >= s.intime AND r.charttime < s.observation_end)
      AS active_in_observation,
    MIN(IF(
      r.charttime >= s.observation_end AND r.charttime < s.prediction_end,
      r.charttime,
      NULL
    )) AS first_active_time_in_prediction,
    COUNTIF(r.charttime >= s.observation_end AND r.charttime < s.prediction_end)
      AS active_rows_in_prediction,
    ARRAY_AGG(
      DISTINCT IF(
        r.charttime >= s.observation_end AND r.charttime < s.prediction_end,
        r.dialysis_type,
        NULL
      ) IGNORE NULLS
      ORDER BY IF(
        r.charttime >= s.observation_end AND r.charttime < s.prediction_end,
        r.dialysis_type,
        NULL
      )
    ) AS modalities_in_prediction
  FROM stays AS s
  LEFT JOIN rrt_events AS r USING (stay_id)
  GROUP BY s.stay_id
)
SELECT
  s.stay_id,
  s.subject_id,
  s.hadm_id,
  s.intime,
  s.outtime,
  s.observation_end,
  s.prediction_end,
  s.outtime >= s.observation_end AS complete_observation_window,
  s.outtime >= s.prediction_end AS complete_prediction_window,
  COALESCE(vent.active_in_observation, FALSE) AS ventilation_active_in_observation,
  vent.first_start_in_prediction AS ventilation_first_start_in_prediction,
  vent.starts_in_prediction AS ventilation_starts_in_prediction,
  COALESCE(vaso.active_in_observation, FALSE) AS vasopressor_active_in_observation,
  vaso.first_start_in_prediction AS vasopressor_first_start_in_prediction,
  vaso.starts_in_prediction AS vasopressor_starts_in_prediction,
  vaso.agents_in_prediction AS vasopressor_agents_in_prediction,
  COALESCE(rrt.active_in_observation, FALSE) AS rrt_active_in_observation,
  rrt.first_active_time_in_prediction AS rrt_first_active_time_in_prediction,
  rrt.active_rows_in_prediction AS rrt_active_rows_in_prediction,
  rrt.modalities_in_prediction AS rrt_modalities_in_prediction
FROM stays AS s
LEFT JOIN vent USING (stay_id)
LEFT JOIN vaso USING (stay_id)
LEFT JOIN rrt USING (stay_id)
ORDER BY stay_id;
