# Outcome-definition workflow

TRUST-ICU does not permit model training until each outcome is locally validated in both
MIMIC-IV v3.1 and eICU-CRD v2.0. Upstream code and keyword matches are evidence sources,
not automatically valid labels.

## 1. Locked temporal semantics

- ICU admission is time zero.
- Observation window: `[0, 6 h)`.
- Prediction window: `[6 h, 18 h)`.
- An event at exactly 6 hours belongs to the prediction window.
- An event at exactly 18 hours is outside the prediction window.
- Any qualifying support active before 6 hours excludes that stay from that task.

These rules are tested in `tests/test_outcomes.py`.

## 2. Source provenance

The public contract is `schemas/outcome_contracts.yaml`. It pins:

- MIT-LCP/mimic-code commit `3a914fce11e05888a4b659c7788e207bc34d1728`;
- MIT-LCP/eicu-code commit `34cece8c70771a3fab48da84d4c47f0e133ca021`.

MIMIC concepts use maintained derived relations for ventilation, individual vasopressor
infusions and active renal-replacement therapy. They still require local schema, timing and
frequency checks on the installed MIMIC-IV v3.1 environment.

eICU has heterogeneous hospital interfaces. Absence from one table cannot be interpreted as
absence of treatment. Candidate vocabularies must be discovered across the documented source
tables and reviewed before a final rule is written.

## 3. MIMIC local audit

Run in the credentialed BigQuery environment:

```text
sql/mimic/00_verify_upstream_concepts.sql
sql/mimic/01_outcome_timing_inventory.sql
```

The first query checks source counts, time integrity and rate fields. The second produces a
restricted event-timing inventory. Never commit its output.

Required review:

1. confirm the exact v3.1 raw schema and derived concept relations;
2. inspect null, reversed and nonpositive intervals;
3. verify the observation/prediction boundary on sampled timelines;
4. verify that catheter-only RRT evidence is not treated as active therapy;
5. record aggregate counts and the query hash outside GitHub.

Only after this review may the MIMIC source status change to `locally_validated`.

## 4. eICU vocabulary discovery

Run in a credentialed PostgreSQL copy of eICU-CRD v2.0:

```text
sql/eicu/00_outcome_vocabulary_discovery.sql
```

The query emits candidate text and aggregate coverage from:

- `respiratoryCare`, `respiratoryCharting` and `treatment` for ventilation;
- `infusionDrug` for vasopressors;
- `treatment` and `intakeOutput` for RRT.

Keyword matches are not final labels. Review each candidate as one of:

- `positive_active_therapy`;
- `negative_not_active`;
- `history_or_plan`;
- `access_or_monitoring_only`;
- `ambiguous_requires_review`;
- `exclude_malformed`.

For each accepted positive term, record the source field, exact normalized text, row count,
unit-stay count, hospital count, reviewer and rationale. Keep credentialed aggregate exports in
the secure output directory, not in this repository.

## 5. Cross-source agreement

Before locking an eICU definition, measure agreement between independent sources where
possible. Examples:

- invasive airway plus vent timing versus respiratory chart settings;
- respiratory documentation versus structured treatment path;
- vasopressor alias plus a nonzero infusion rate;
- RRT treatment path versus dialysis-related intake/output evidence.

Report source-specific sensitivity proxies and hospital coverage. A source with low interface
coverage cannot define a reliable negative class by itself.

## 6. Clinical equivalence review

A clinician or appropriately qualified critical-care reviewer must confirm that the final
MIMIC and eICU rules represent the same intervention. Review must explicitly address:

- invasive versus non-invasive ventilation;
- vasopressors versus inotropes and vasodilators;
- active dialysis versus access placement, historical dialysis or planning;
- duplicate, stopped, held and zero-rate records;
- whether different documentation delays could move events across the 6-hour boundary.

Record approval as `clinical_equivalence_review: approved` in the task contract.

## 7. Synthetic timeline tests

Every final extractor must be tested on synthetic cases covering:

- support ending before ICU admission;
- support overlapping ICU admission;
- support starting at 5:59;
- support starting exactly at 6:00;
- support starting at 17:59;
- support starting exactly at 18:00;
- duplicate events;
- zero-rate or held infusion;
- access placement without active RRT.

Record success as `synthetic_timeline_tests: passed`.

## 8. Lock enforcement

Validate the public contracts:

```bash
python scripts/validate_outcome_contracts.py
```

Before model training, require all tasks to be locked:

```bash
python scripts/validate_outcome_contracts.py --require-locked
```

The command exits with status 2 while any required source validation, equivalence review or
timeline test is incomplete. Model entry points must additionally call
`assert_task_training_allowed()` so training fails closed even when invoked directly.

## 9. No-go conditions

Remove or repair a task before modelling when any of the following holds:

- cross-database clinical equivalence cannot be established;
- eICU hospital interface coverage is too sparse or concentrated;
- timing granularity makes the 6-hour boundary unreliable;
- active treatment cannot be separated from history, planning or access documentation;
- source disagreement cannot be adjudicated;
- local counts do not support the preregistered feasibility thresholds.
