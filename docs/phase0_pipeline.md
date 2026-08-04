# Phase 0 Canonical Pipeline

## Public canonical interfaces

Source-specific SQL must emit three restricted tables outside Git. The exact columns and fail-closed rules are versioned in `schemas/source_adapter_contract.yaml`.

### Stay table

Required columns:

- `dataset_id`
- `patient_id`
- `hospital_admission_id`
- `stay_id`
- `hospital_id`
- `age`
- `sex`
- `icu_admit_time`
- `icu_discharge_time`

Optional static metadata includes `death_time`, admission weight/height, admission source and unit type. `stay_id` must be unique.

### Long observation table

Required columns:

- `stay_id`
- `variable`
- `event_time`
- `value`
- `unit`
- `source_table`
- `source_code`

Only canonical variable names and units from `schemas/phase0_features.yaml` are accepted. Source provenance is mandatory.

### Event table

Required columns:

- `stay_id`
- `task`
- `start_time`
- `end_time`
- `source_table`
- `source_code`

`end_time` may be null for an ongoing interval. Events are generated only from locally validated and locked outcome definitions.

## Source-adapter audit

Every MIMIC-IV and eICU export must pass the aggregate canonical audit before cohort construction:

```bash
python scripts/audit_canonical_extract.py \
  --dataset mimic_iv_3_1 \
  --stays /secure/exports/stays.csv.gz \
  --events /secure/exports/events.csv.gz \
  --observations /secure/exports/observations.csv.gz \
  --output /secure/audits/canonical_audit.json \
  --require-ready
```

The audit blocks duplicate stays, broken intervals, unknown tasks/variables, wrong units, unlinked rows, missing provenance, invalid values and observations outside `[0 h, 6 h)`. Reports contain aggregate counts and hashes only.

## Pipeline order

1. Run the source-adapter contract and canonical extract audit.
2. Build the six-hour landmark cohort.
3. Select one first eligible ICU unit stay per hospital admission.
4. Assign each task's eligibility and incident outcome.
5. Exclude support active during `[0 h, 6 h)` for that task.
6. Flag death without the outcome as a competing event.
7. Aggregate only observations in `[ICU admission, landmark)`.
8. Add missingness and measurement-density features.
9. Remove all identifiers and future/outcome metadata from predictors.
10. Split MIMIC-IV temporally and fit the locked development baseline.
11. Apply the unchanged pipeline to eICU for primary external evaluation.

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

Duplicate values for one stay, variable and timestamp are mean-collapsed and counted in the quality-control audit. Unknown variables, invalid values and observations outside the window are excluded and reported rather than silently clipped.

## Training lock

Public utilities allow training only when data are marked `synthetic` or `credentialed_locked`. The latter classification may be used only after outcome-lock artifacts confirm local MIMIC validation, local eICU validation, clinical-equivalence approval and passed synthetic timeline tests. A canonical source extract must additionally have `ready_for_cohort_build=true`.

## Privacy

The repository may contain SQL, schemas, tests and aggregate reports. It must not contain patient-level cohorts, feature matrices, event tables, row-level predictions or credentials.
