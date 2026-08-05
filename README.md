# TRUST-ICU

**Transportable and Risk-Controlled Prediction of Acute Organ-Support Escalation Across Multiple ICUs**

TRUST-ICU is a protocol-first clinical prediction study. It tests whether an early ICU model remains useful, calibrated, and safe when moved from MIMIC-IV to independent eICU hospitals. Complex architecture development is prohibited until locked outcome definitions and a fully external feasibility gate are passed.

## Current phase

**Phase 0: MIMIC-IV v3.1 → eICU-CRD v2.0 feasibility validation**

Using only the first six hours after ICU admission, the study predicts new support initiation during `[6 h, 18 h)`:

1. invasive mechanical ventilation;
2. vasopressor therapy;
3. renal-replacement therapy.

Support already active during `[0 h, 6 h)` makes that stay ineligible for the corresponding task. A new start exactly at six hours is an outcome event. Death without the target support is retained as a competing event and excluded from the primary binary analysis.

## Data roles

| Database | Version | Locked role |
|---|---:|---|
| MIMIC-IV | 3.1 | Development, calibration, and temporal testing |
| eICU-CRD | 2.0 | Primary fully external multi-hospital validation |
| AmsterdamUMCdb | 1.5.0 | Later European validation after Phase 0 |
| HiRID | 1.1.1 | Later high-resolution European validation after Phase 0 |

These databases require their own credentialing and data-use agreements. No patient-level data are distributed in this repository.

## Scientific design

### Landmark cohort

The sampling unit is the first eligible ICU unit stay per hospital admission. A patient must be an adult, alive, and still in the ICU strictly after the six-hour landmark. Follow-up ends at the earliest of ICU discharge, death, or 18 hours after ICU admission.

### Development split

For each outcome, eligible MIMIC-IV admissions are ordered by ICU admission time:

```text
earliest 70%  → model training
next 15%      → Platt calibration
latest 15%    → untouched temporal testing
```

If a patient appears in an earlier split, that patient's rows are removed from every later split. Rows are never randomly moved backward. See `docs/protocol_amendment_002.md`.

### Locked Phase 0 models

Only these models are allowed:

- regularized Logistic Regression;
- fixed CatBoost.

The development-selected model is chosen by:

```text
MIMIC temporal calibrated PR-AUC
→ lower Brier score for a tie
→ Logistic Regression for a final simplicity tie
```

Both prespecified baselines may be transparently reported on eICU, but the go/no-go decision is applied only to the model selected using MIMIC-IV. eICU cannot be used for feature selection, model selection, hyperparameter tuning, threshold tuning, or recalibration.

## Outcome lock before modelling

`schemas/outcome_contracts.yaml` is fail-closed. Training is prohibited until every task has:

- local MIMIC-IV validation;
- local eICU validation;
- approved cross-database clinical equivalence;
- passing synthetic boundary tests.

```bash
python scripts/validate_outcome_contracts.py
python scripts/validate_outcome_contracts.py --require-locked
```

The second command intentionally fails while credentialed validation is incomplete.

## Feature contract

`schemas/phase0_features.yaml` locks 24 canonical variables, units, plausible ranges, and the allowed six-hour summaries:

- first and last;
- minimum, maximum, and mean;
- linear slope per hour;
- measurement count and missingness;
- hours since the last measurement.

```bash
python scripts/validate_phase0_features.py
```

Values outside the public plausible range do not enter model features. Observations before ICU admission or at/after the six-hour landmark are audited and block final execution.

## Source adapters

Every database adapter emits three canonical restricted tables:

```text
stays
events
observations
```

`schemas/source_adapter_manifest.yaml` fixes source files, upstream commits, execution order, and exact dataset identifiers.

### MIMIC-IV

Implemented public adapters:

```text
sql/mimic/02_base_landmark_cohort.sql
sql/mimic/03_canonical_events.sql
sql/mimic/04_canonical_observations.sql
```

### eICU

eICU interfaces and vocabularies vary across hospitals. Direct documented vital columns are available, but labs and outcome terms require local frequency review and clinical approval.

```text
sql/eicu/00_outcome_vocabulary_discovery.sql
sql/eicu/01_base_landmark_cohort.sql
sql/eicu/01a_create_local_mapping_tables.sql
sql/eicu/02_canonical_observations_template.sql
sql/eicu/03_canonical_events_template.sql
schemas/eicu_feature_map_template.csv
schemas/eicu_outcome_map_template.csv
```

Only rows explicitly marked `status='locked'` participate in eICU exports. Unreviewed keyword matches cannot create a positive outcome.

## Secure credentialed extraction

Install database support:

```bash
pip install -e ".[dev,db]"
```

Credential-free plan:

```bash
python scripts/run_credentialed_extract.py --dataset mimic_iv_3_1 --dry-run
python scripts/run_credentialed_extract.py --dataset eicu_crd_2_0 --dry-run
```

Inside the authorized PostgreSQL environment:

```bash
export TRUST_ICU_POSTGRES_DSN='use-a-protected-secret-source'
export TRUST_ICU_OUTPUT_ROOT=/secure/trust_icu_outputs

python scripts/run_credentialed_extract.py --dataset mimic_iv_3_1
```

For eICU, first create local review tables, review and lock mappings, then execute:

```bash
python scripts/run_credentialed_extract.py \
  --dataset eicu_crd_2_0 \
  --prepare-eicu-mappings

python scripts/run_credentialed_extract.py \
  --dataset eicu_crd_2_0 \
  --allow-reviewed-eicu
```

The runner materializes canonical relations, runs aggregate database-side audits, and writes compressed exports only after the critical audit passes. It never stores or prints the DSN.

Full instructions: `docs/credentialed_execution.md`.

## Locked Phase 0 baseline execution

Install modelling and scalable analytics support:

```bash
pip install -e ".[dev,phase0]"
```

Credential-free plan:

```bash
python scripts/run_phase0_baselines.py --dry-run
```

After both source runs and outcome locks are complete:

```bash
export TRUST_ICU_PHASE0_OUTPUT_ROOT=/secure/trust_icu_phase0_results

python scripts/run_phase0_baselines.py \
  --mimic-run-dir /secure/trust_icu_outputs/mimic_iv_3_1 \
  --eicu-run-dir /secure/trust_icu_outputs/eicu_crd_2_0
```

The baseline runner verifies:

- credentialed report hashes;
- canonical audit equality;
- every export SHA-256 and byte count;
- exact cross-database feature columns;
- patient isolation across temporal splits;
- two-class availability in every split;
- non-missing eICU hospital identifiers.

DuckDB builds the wide six-hour feature matrix directly from compressed observations without loading the raw observation table into pandas first.

### Aggregate-only outputs

```text
phase0_go_no_go.json
phase0_aggregate_summary.csv
```

The runner does **not** save patient-level predictions, row-level split assignments, hospital identifiers, fitted models, or calibration objects.

The report contains:

- input and contract hashes;
- source and feature audits;
- temporal split and purge counts;
- raw and calibrated temporal metrics;
- raw and calibrated external metrics;
- hospital-cluster bootstrap intervals;
- aggregate hospital robustness;
- cross-database missingness shift;
- deterministic task-wise go/no-go decisions.

Full instructions: `docs/phase0_baseline_execution.md`.

## Feasibility gates

Architecture development continues for a task only when every preregistered requirement passes:

- at least 2,000 MIMIC positive events;
- at least 500 eICU positive events;
- event rate between 1% and 50% in both datasets;
- external PR-AUC at least twice prevalence;
- external calibration slope within 0.35 of 1;
- absolute external calibration intercept at most 0.75;
- at least 20 external hospitals;
- no post-index leakage;
- equivalent cross-database outcome definition.

A failed gate is a scientific result. It must not trigger repeated searches against the same external cohort.

## Setup and public checks

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev,phase0]"

ruff check src tests scripts
pytest
python scripts/validate_outcome_contracts.py
python scripts/validate_phase0_features.py
python scripts/validate_source_adapter_contract.py
python scripts/validate_source_adapter_manifest.py
python scripts/run_credentialed_extract.py --dataset mimic_iv_3_1 --dry-run
python scripts/run_credentialed_extract.py --dataset eicu_crd_2_0 --dry-run
python scripts/run_phase0_baselines.py --dry-run
python scripts/run_synthetic_phase0.py --n 1200
```

Synthetic metrics are software checks only and must never be reported as clinical performance.

## Repository structure

```text
configs/
  feasibility.yaml

docs/
  phase0_protocol.md
  protocol_amendment_001.md
  protocol_amendment_002.md
  phase0_pipeline.md
  phase0_baseline_execution.md
  credentialed_execution.md
  source_adapter_contract.md
  source_adapter_execution.md
  outcome_definition_workflow.md
  data_access.md

schemas/
  common_variables.csv
  outcome_contracts.yaml
  phase0_features.yaml
  source_adapter_contract.yaml
  source_adapter_manifest.yaml
  eicu_feature_map_template.csv
  eicu_outcome_map_template.csv

scripts/
  check_environment.py
  validate_outcome_contracts.py
  validate_phase0_features.py
  validate_source_adapter_contract.py
  validate_source_adapter_manifest.py
  audit_canonical_extract.py
  run_credentialed_extract.py
  run_phase0_baselines.py
  run_synthetic_phase0.py

src/trust_icu/
  adapter_manifest.py
  credentialed_runner.py
  source_validation.py
  config.py
  outcomes.py
  cohort.py
  features.py
  baseline.py
  phase0_runner.py
  validation.py
```

## Scientific safeguards

- Six-hour observation and 12-hour prediction windows are fixed.
- Predictors at or after the landmark are prohibited.
- Identifiers are evaluation metadata, never predictors.
- External recalibration and external model selection are prohibited.
- Best-fold and best-hospital reporting are prohibited.
- Hospital uncertainty uses cluster bootstrap.
- Patient overlap across development splits is removed downstream.
- Patient-level data, predictions, credentials, and model artifacts are ignored by Git.
- Protocol changes require a timestamped prospective amendment.
- Reporting will follow TRIPOD+AI and risk of bias will be assessed with PROBAST+AI.

## Roadmap

- [x] Initialize a privacy-safe repository.
- [x] Lock windows, outcomes, metrics, and stop rules.
- [x] Add fail-closed outcome contracts.
- [x] Implement canonical MIMIC adapters.
- [x] Add reviewed-vocabulary-driven eICU templates.
- [x] Add secure PostgreSQL extraction and aggregate audits.
- [x] Add scalable DuckDB feature construction.
- [x] Lock patient-purged 70/15/15 temporal validation.
- [x] Add development-only model selection and MIMIC-only calibration.
- [x] Add eICU hospital-cluster bootstrap and aggregate robustness.
- [x] Add machine-readable Phase 0 go/no-go generation.
- [ ] Confirm MIMIC-IV v3.1 credentialed access.
- [ ] Confirm eICU-CRD v2.0 credentialed access.
- [ ] Execute and audit the MIMIC adapter.
- [ ] Review and lock eICU vocabularies.
- [ ] Execute and audit the eICU adapter.
- [ ] Approve cross-database outcome equivalence.
- [ ] Run the locked Phase 0 baselines exactly once.
- [ ] Continue or stop each task using the preregistered decision.

## Governance

This is a public code repository. Do not commit raw or derived patient-level data, credentials, private access material, row-level predictions, disallowed aggregate exports, or restricted-data model artifacts.

## Primary references

- MIMIC-IV v3.1, PhysioNet, DOI: `10.13026/kpb9-mt58`
- eICU-CRD v2.0, PhysioNet, DOI: `10.13026/C2WM1R`
- HiRID v1.1.1, PhysioNet, DOI: `10.13026/nkwc-js72`
- AmsterdamUMCdb v1.5.0, Amsterdam UMC
- TRIPOD+AI, BMJ 2024, DOI: `10.1136/bmj-2023-078378`
- PROBAST+AI, BMJ 2025, DOI: `10.1136/bmj-2024-082505`
