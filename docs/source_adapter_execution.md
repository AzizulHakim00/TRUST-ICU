# Source-Adapter Execution

## Purpose

This stage converts credentialed MIMIC-IV and eICU tables into the canonical `stays`, `events`
and `observations` contracts. The public manifest at
`schemas/source_adapter_manifest.yaml` fixes execution order, upstream commits and output
identifiers.

## MIMIC-IV v3.1

The public MIMIC adapter is implemented but unexecuted:

1. Materialize `sql/mimic/02_base_landmark_cohort.sql` as
   `trust_icu_work.mimic_stays`.
2. Export `sql/mimic/03_canonical_events.sql`.
3. Export `sql/mimic/04_canonical_observations.sql`.
4. Run `scripts/audit_canonical_extract.py --dataset mimic_iv_3_1 --require-ready`.

The event adapter uses maintained MIMIC concepts for invasive ventilation, five prespecified
vasopressors and active RRT. The observation adapter uses maintained concepts for vital signs,
chemistry, complete blood count, bilirubin, blood gas, GCS and urine output. Upstream concepts
remain evidence sources; credentialed frequency and timing audits are still required.

## eICU-CRD v2.0

eICU hospital interfaces and labels vary, so only documented direct vital columns are exported
without a local vocabulary. Laboratory and outcome mappings must be reviewed inside the
credentialed environment:

1. Materialize `sql/eicu/01_base_landmark_cohort.sql` as
   `trust_icu_work.eicu_stays`.
2. Run `sql/eicu/01a_create_local_mapping_tables.sql`.
3. Populate local mapping tables from the public CSV templates after frequency review.
4. Change only approved rows to `status='locked'` and record a reviewer.
5. Run the canonical observation and event templates.
6. Run the canonical extract audit with `--dataset eicu_crd_2_0 --require-ready`.

Unreviewed rows with `pending_local_review` are ignored by the SQL templates. Keyword discovery
alone never creates a positive outcome.

## Public validation

```bash
python scripts/validate_source_adapter_manifest.py
```

The report distinguishes public-file completeness from scientific readiness. The MIMIC adapter
can be ready for credentialed execution while eICU remains blocked pending local reviewed
mappings.

## Prohibited actions

- Do not change `dataset_id` values from `mimic_iv_3_1` or `eicu_crd_2_0`.
- Do not mark a mapping `locked` without local frequency review and clinical approval.
- Do not tune mappings using external model performance.
- Do not commit canonical extracts, patient rows or row-level predictions.
