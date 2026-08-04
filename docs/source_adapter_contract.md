# Canonical Source-Adapter Contract

## Purpose

MIMIC-IV and eICU use different table structures, identifiers, labels, units and timing conventions. Source-specific SQL must therefore export three canonical tables before any cohort construction or modelling:

1. `stays`
2. `events`
3. `observations`

The canonical audit is fail-closed. A failed extract must be corrected at the source adapter. Rows are not silently repaired, clipped or relabelled inside the modelling pipeline.

## Canonical stays table

Required columns:

- `dataset_id`
- `patient_id`
- `hospital_admission_id`
- `stay_id`
- `hospital_id`
- `icu_admit_time`
- `icu_discharge_time`
- `age`
- `sex`

Optional static metadata:

- `death_time`
- `admission_weight_kg`
- `admission_height_cm`
- `admission_source`
- `unit_type`

`stay_id` must be unique. All timestamps must be parseable as UTC-aware timestamps after local normalization. ICU discharge must occur after ICU admission.

## Canonical events table

Required columns:

- `stay_id`
- `task`
- `start_time`
- `end_time`
- `source_table`
- `source_code`

Allowed tasks:

- `invasive_mechanical_ventilation`
- `vasopressor_initiation`
- `renal_replacement_therapy`

`end_time` may be null for an ongoing interval. When present, it must be strictly after `start_time`. `source_table` and `source_code` preserve provenance and may not be blank.

## Canonical observations table

Required columns:

- `stay_id`
- `variable`
- `event_time`
- `value`
- `unit`
- `source_table`
- `source_code`

Variables and canonical units are defined by `schemas/phase0_features.yaml`. Source adapters must perform explicit unit conversion before export. Observations must be inside the locked `[ICU admission, ICU admission + 6 hours)` window. The six-hour boundary is excluded.

## Critical audit failures

The extract is not allowed to proceed when any of the following is present:

- duplicate `stay_id` values;
- missing core identifiers or timestamps;
- non-positive ICU intervals;
- unknown outcome tasks;
- invalid event intervals;
- event or observation rows not linked to a canonical stay;
- observations before ICU admission;
- observations at or after the six-hour landmark;
- unknown canonical variables;
- canonical-unit mismatches;
- missing or nonnumeric observation values;
- duplicate observation rows;
- missing source provenance.

The audit report contains aggregate counts and contract hashes only. It does not include patient-level examples.

## Local execution

```bash
python scripts/validate_source_adapter_contract.py

python scripts/audit_canonical_extract.py \
  --dataset mimic_iv_3_1 \
  --stays /secure/exports/mimic_stays.csv.gz \
  --events /secure/exports/mimic_events.csv.gz \
  --observations /secure/exports/mimic_observations.csv.gz \
  --output /secure/audits/mimic_canonical_audit.json \
  --require-ready
```

Use `eicu_crd_2_0` for the eICU adapter. CSV, compressed CSV and Parquet inputs are supported; Parquet requires a local Parquet engine.

Exit code `2` means the extract failed at least one critical rule. Baseline training must not continue.

## Privacy and governance

Canonical extracts remain inside the credentialed environment. Do not commit them, their row-level derivatives or row-level predictions. Only aggregate audit reports may be moved outside the secure environment when the applicable data-use agreement permits release.
