# Phase 0 Canonical Pipeline

## Public canonical interfaces

Source-specific SQL must emit two restricted tables outside Git:

### Stay table

Required columns:

- `patient_id`
- `hospital_admission_id`
- `stay_id`
- `age`
- `icu_admit_time`
- `icu_discharge_time`
- optional `death_time`
- evaluation-only metadata such as `hospital_id` and `dataset_id`

### Long observation table

Required columns:

- `stay_id`
- `variable`
- `event_time`
- `value`

Only canonical variable names from `schemas/phase0_features.yaml` are accepted.

### Event table

Required columns:

- `stay_id`
- `task`
- `start_time`
- optional `end_time`

Events are generated only from locally validated and locked outcome definitions.

## Pipeline order

1. Build the six-hour landmark cohort.
2. Select one first eligible ICU unit stay per hospital admission.
3. Assign each task's eligibility and incident outcome.
4. Exclude support active during `[0 h, 6 h)` for that task.
5. Flag death without the outcome as a competing event.
6. Aggregate only observations in `[ICU admission, landmark)`.
7. Add missingness and measurement-density features.
8. Remove all identifiers and future/outcome metadata from predictors.
9. Split MIMIC-IV temporally and fit the locked development baseline.
10. Apply the unchanged pipeline to eICU for primary external evaluation.

## Six-hour summaries

For each time-varying variable:

- first;
- last;
- minimum;
- maximum;
- mean;
- linear slope per hour when at least two distinct timestamps exist;
- count;
- missingness indicator;
- hours since last measurement.

Duplicate values for one stay, variable and timestamp are mean-collapsed and counted in the
quality-control audit. Unknown variables and observations outside the window are excluded and
reported rather than silently clipped.

## Training lock

Public utilities allow training only when data are marked `synthetic` or
`credentialed_locked`. The latter classification may be used only after outcome-lock artifacts
confirm local MIMIC validation, local eICU validation, clinical-equivalence approval and passed
synthetic timeline tests.

## Privacy

The repository may contain SQL, schemas, tests and aggregate reports. It must not contain
patient-level cohorts, feature matrices, event tables, row-level predictions or credentials.
