# TRUST-ICU

**Transportable and Risk-Controlled Prediction of Acute Organ-Support Escalation Across Multiple ICUs**

TRUST-ICU is a protocol-first clinical prediction study designed to test whether early ICU models remain useful, calibrated and safe across independent hospitals and databases. The project does not proceed to a complex architecture unless locked outcome definitions and a fully external feasibility gate are passed.

## Current phase

**Phase 0: MIMIC-IV → eICU feasibility validation**

The study uses a six-hour landmark and predicts new organ-support initiation before ICU discharge during the following 12 hours:

1. invasive mechanical ventilation;
2. vasopressor therapy;
3. renal-replacement therapy.

Each task has its own eligibility rule. Support active during `[0 h, 6 h)` excludes that stay from the corresponding task. A new start at exactly six hours is an outcome event.

## Data roles

| Database | Version | Role |
|---|---:|---|
| MIMIC-IV | 3.1 | Development and temporal validation |
| eICU-CRD | 2.0 | Primary fully external multi-hospital validation |
| AmsterdamUMCdb | 1.5.0 | Later European validation after Phase 0 |
| HiRID | 1.1.1 | Later high-resolution European validation after Phase 0 |

These databases require their own credentialing and data-use agreements. No patient-level data are distributed here.

## Locked landmark unit

The sampling unit is the **first eligible ICU unit stay per hospital admission**. Cross-admission chronology for the same eICU patient is not inferred. A stay enters the landmark cohort only when the patient is an adult, alive and still in the ICU strictly after six hours.

Follow-up ends at the earliest of ICU discharge, death or 18 hours after ICU admission. Discharge without escalation is negative for the primary in-unit estimand. Death without the target support is recorded as a competing event and excluded from the primary binary model. See `docs/protocol_amendment_001.md`.

## Why Phase 0 comes first

Only regularized logistic regression and a fixed CatBoost baseline are allowed in the initial study. The goal is to establish:

- enough eligible events;
- equivalent cross-database outcome definitions;
- absence of post-index leakage;
- useful external PR-AUC relative to prevalence;
- acceptable external calibration;
- adequate hospital representation.

If the mandatory gates fail, architecture development stops for that task. A failed gate does not trigger repeated model searches on the same external cohort.

## Outcome lock before modelling

The public outcome contract is stored in `schemas/outcome_contracts.yaml`. Credentialed model training is prohibited until each task has:

- local MIMIC-IV v3.1 validation;
- local eICU-CRD v2.0 validation;
- approved cross-database clinical-equivalence review;
- passing synthetic timeline tests.

Run the public metadata checks:

```bash
python scripts/validate_outcome_contracts.py
```

A model entry point must require a complete lock:

```bash
python scripts/validate_outcome_contracts.py --require-locked
```

The second command intentionally fails while definitions are incomplete.

## Feature contract

`schemas/phase0_features.yaml` locks 24 canonical variables, units, broad physiological plausibility ranges and the allowed six-hour summaries:

- first and last value;
- minimum, maximum and mean;
- linear slope per hour;
- measurement count;
- missingness indicator;
- hours since the last measurement.

Unknown variables, observations outside `[0 h, 6 h)` and physiologically implausible values are audited and excluded rather than silently clipped.

Validate the public feature contract:

```bash
python scripts/validate_phase0_features.py
```

## Source adapters

The canonical adapter contract requires three restricted tables:

- `stays`;
- `events`;
- `observations`.

`schemas/source_adapter_manifest.yaml` fixes the source files, upstream repository commits, execution order and exact output dataset identifiers.

### MIMIC-IV

The public MIMIC adapter is implemented and ready for credentialed execution, but has not yet been run on patient data:

```text
sql/mimic/02_base_landmark_cohort.sql
sql/mimic/03_canonical_events.sql
sql/mimic/04_canonical_observations.sql
```

The event adapter uses maintained MIMIC concepts for invasive ventilation, norepinephrine, epinephrine, vasopressin, phenylephrine, dopamine and active RRT. The observation adapter uses maintained concepts for vital signs, chemistry, complete blood count, bilirubin, blood gas, GCS and urine output.

### eICU

eICU labels and interfaces vary by hospital. Direct documented vital columns can be exported, but hospital-specific labs and outcome vocabularies require local frequency review and clinical approval.

```text
sql/eicu/01_base_landmark_cohort.sql
sql/eicu/01a_create_local_mapping_tables.sql
sql/eicu/02_canonical_observations_template.sql
sql/eicu/03_canonical_events_template.sql
schemas/eicu_feature_map_template.csv
schemas/eicu_outcome_map_template.csv
```

Only mappings marked `status='locked'` participate in eICU exports. Unreviewed keyword matches cannot create a positive outcome.

Validate the public execution manifest:

```bash
python scripts/validate_source_adapter_manifest.py
```

Full execution instructions are in `docs/source_adapter_execution.md`.

## Canonical extract audit

Before cohort construction, every restricted extract must pass the aggregate-only source audit:

```bash
python scripts/audit_canonical_extract.py \
  --dataset mimic_iv_3_1 \
  --stays /secure/exports/mimic_stays.csv.gz \
  --events /secure/exports/mimic_events.csv.gz \
  --observations /secure/exports/mimic_observations.csv.gz \
  --output /secure/audits/mimic_canonical_audit.json \
  --require-ready
```

The audit blocks duplicate stay identifiers, unknown tasks or variables, wrong units, invalid intervals, unlinked rows, future predictors, missing provenance and malformed values. Reports contain aggregate counts and hashes only.

## Canonical pipeline

The public Python layer provides:

- six-hour landmark cohort construction;
- first eligible ICU unit selection per hospital admission;
- task-specific active-support exclusion;
- competing-death flags;
- long-to-wide feature aggregation;
- identifier and future-information leakage guards;
- source-adapter contract and manifest validation;
- aggregate-only canonical extract auditing;
- Logistic Regression and fixed CatBoost baselines;
- PR-AUC, prevalence-normalized PR-AUC, AUROC, Brier score and calibration slope/intercept.

The database-neutral interfaces and execution order are documented in `docs/phase0_pipeline.md`.

## Repository structure

```text
configs/
  feasibility.yaml

docs/
  phase0_protocol.md
  protocol_amendment_001.md
  phase0_pipeline.md
  source_adapter_contract.md
  source_adapter_execution.md
  data_access.md
  outcome_definition_workflow.md

schemas/
  common_variables.csv
  outcome_contracts.yaml
  phase0_features.yaml
  source_adapter_contract.yaml
  source_adapter_manifest.yaml
  eicu_feature_map_template.csv
  eicu_outcome_map_template.csv

sql/
  mimic/00_verify_upstream_concepts.sql
  mimic/01_outcome_timing_inventory.sql
  mimic/02_base_landmark_cohort.sql
  mimic/03_canonical_events.sql
  mimic/04_canonical_observations.sql
  eicu/00_outcome_vocabulary_discovery.sql
  eicu/01_base_landmark_cohort.sql
  eicu/01a_create_local_mapping_tables.sql
  eicu/02_canonical_observations_template.sql
  eicu/03_canonical_events_template.sql

scripts/
  check_environment.py
  validate_outcome_contracts.py
  validate_phase0_features.py
  validate_source_adapter_contract.py
  validate_source_adapter_manifest.py
  audit_canonical_extract.py
  run_synthetic_phase0.py

src/trust_icu/
  adapter_manifest.py
  source_validation.py
  config.py
  outcomes.py
  cohort.py
  features.py
  baseline.py
  validation.py

tests/
  test_adapter_manifest.py
  test_source_validation.py
  test_outcomes.py
  test_cohort.py
  test_features.py
  test_baseline.py
  test_validation.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
ruff check src tests scripts
pytest
python scripts/validate_outcome_contracts.py
python scripts/validate_phase0_features.py
python scripts/validate_source_adapter_contract.py
python scripts/validate_source_adapter_manifest.py
python scripts/run_synthetic_phase0.py --n 1200
```

Install CatBoost for the full fixed-baseline environment:

```bash
pip install -e ".[dev,ml]"
```

The synthetic command is a software dry-run only. Its metrics are not clinical results and must never be reported as study performance.

Configure secure paths outside the repository:

```bash
export TRUST_ICU_MIMIC_ROOT=/secure/path/to/mimiciv/3.1
export TRUST_ICU_EICU_ROOT=/secure/path/to/eicu/2.0
export TRUST_ICU_OUTPUT_ROOT=/secure/path/to/trust_icu_outputs
python scripts/check_environment.py
```

The environment checker reports only path readiness. It never reads clinical tables.

## Scientific safeguards

- Six-hour observation and 12-hour prediction windows are fixed before extraction.
- The intervals are `[0 h, 6 h)` and `[6 h, 18 h)`.
- Predictors at or after the six-hour boundary are prohibited.
- Patient, stay, hospital and dataset identifiers are evaluation metadata, not predictors.
- Dataset IDs are fixed as `mimic_iv_3_1` and `eicu_crd_2_0`.
- Upstream concepts and keyword matches are evidence sources, not automatically valid labels.
- eICU mappings must be locally reviewed and explicitly locked.
- eICU is not used for feature selection, model selection, threshold tuning or primary calibration.
- Headline evidence is fully external performance, not the best fold or best hospital.
- Missingness and measurement frequency are audited as transportability factors.
- Patient-level data, row-level predictions, credentials and trained restricted-data checkpoints are ignored by Git.
- Protocol deviations require a timestamped amendment written without reference to held-out performance.
- Reporting will follow TRIPOD+AI; risk of bias will be assessed with PROBAST+AI.

## Immediate roadmap

- [x] Initialize privacy-safe repository.
- [x] Lock Phase 0 windows, tasks, metrics and stop rules.
- [x] Pin upstream outcome sources and add fail-closed outcome contracts.
- [x] Add MIMIC and eICU outcome-discovery SQL.
- [x] Lock the cross-database landmark sampling unit and follow-up policy.
- [x] Add MIMIC and eICU base landmark cohort SQL.
- [x] Add canonical feature contract and physiological-bound audits.
- [x] Add database-neutral cohort, outcome and feature code.
- [x] Add locked Logistic Regression and CatBoost baseline code.
- [x] Add a synthetic end-to-end dry-run and CI enforcement.
- [x] Add canonical source-adapter contract and extract audit.
- [x] Implement public MIMIC stays, events and observations adapters.
- [x] Add reviewed-vocabulary-driven eICU adapter templates.
- [x] Add adapter execution manifest validation.
- [ ] Confirm MIMIC-IV v3.1 access and storage mode.
- [ ] Confirm eICU-CRD v2.0 access and storage mode.
- [ ] Execute and audit the MIMIC adapter in a credentialed environment.
- [ ] Run eICU vocabulary frequency discovery.
- [ ] Clinically review and lock eICU feature and outcome mappings.
- [ ] Execute and audit the eICU adapter.
- [ ] Lock cross-database outcome equivalence.
- [ ] Run cohort, missingness and leakage audits.
- [ ] Run locked MIMIC temporal and eICU external baselines.
- [ ] Produce the machine-readable Phase 0 go/no-go decision.

## Governance

This is a public code repository. Do not commit raw or derived patient-level data, credentials, private access material, row-level predictions, aggregate exports disallowed by a data-use agreement, or model artifacts trained on credentialed databases unless release is explicitly permitted.

## Primary references

- MIMIC-IV v3.1, PhysioNet, DOI: `10.13026/kpb9-mt58`
- eICU-CRD v2.0, PhysioNet, DOI: `10.13026/C2WM1R`
- HiRID v1.1.1, PhysioNet, DOI: `10.13026/nkwc-js72`
- AmsterdamUMCdb v1.5.0, Amsterdam UMC
- TRIPOD+AI, BMJ 2024, DOI: `10.1136/bmj-2023-078378`
- PROBAST+AI, BMJ 2025, DOI: `10.1136/bmj-2024-082505`
