# TRUST-ICU

**Transportable and Risk-Controlled Prediction of Acute Organ-Support Escalation Across Multiple ICUs**

TRUST-ICU is a protocol-first clinical prediction study designed to test whether early ICU models remain useful, calibrated and safe across independent hospitals and databases. The project does not proceed to a complex architecture unless locked outcome definitions and a fully external feasibility gate are passed.

## Current phase

**Phase 0: MIMIC-IV → eICU feasibility validation**

Planned primary tasks predict new initiation during the 12 hours after a six-hour ICU observation window:

1. invasive mechanical ventilation;
2. vasopressor therapy;
3. renal-replacement therapy.

Each task has its own eligible cohort. Patients already receiving the relevant support during the observation window are excluded from that task.

## Data roles

| Database | Version | Role |
|---|---:|---|
| MIMIC-IV | 3.1 | Development and temporal validation |
| eICU-CRD | 2.0 | Primary fully external multi-hospital validation |
| AmsterdamUMCdb | 1.5.0 | Later European validation after Phase 0 |
| HiRID | 1.1.1 | Later high-resolution European validation after Phase 0 |

These databases require their own credentialing and data-use agreements. No patient-level data are distributed here.

## Why Phase 0 comes first

Only regularized logistic regression and CatBoost are allowed in the initial study. The goal is to establish:

- enough eligible events;
- equivalent cross-database outcome definitions;
- absence of post-index leakage;
- useful external PR-AUC relative to prevalence;
- acceptable external calibration;
- adequate hospital representation.

If the mandatory gates fail, architecture development stops for that task. A failed gate does not trigger repeated model searches on the same external cohort.

## Outcome lock before modelling

The public outcome contract is stored in `schemas/outcome_contracts.yaml`. Model training is prohibited until each task has:

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

## Repository structure

```text
configs/
  feasibility.yaml                    # Machine-readable preregistration and stop rules
docs/
  phase0_protocol.md                  # Study design, estimands and validation protocol
  data_access.md                      # Credentialing and data-governance checklist
  outcome_definition_workflow.md      # Definition discovery, review and lock procedure
schemas/
  common_variables.csv                # Initial cross-database variable dictionary
  outcome_contracts.yaml              # Versioned outcome sources and lock requirements
sql/
  mimic/00_verify_upstream_concepts.sql
  mimic/01_outcome_timing_inventory.sql
  eicu/00_outcome_vocabulary_discovery.sql
scripts/
  check_environment.py                # Access-path readiness check; reads no patient data
  validate_outcome_contracts.py       # Public fail-closed outcome lock report
src/trust_icu/
  config.py                           # Typed feasibility configuration validation
  outcomes.py                         # Outcome contract hashing and training guard
  validation.py                       # Aggregate-only deterministic feasibility gates
tests/
  test_outcomes.py                    # Lock and exact boundary tests
  test_validation.py                  # Synthetic go/no-go tests
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
```

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
- The intervals are `[0 h, 6 h)` and `[6 h, 18 h)`; an event at exactly 6 hours is an outcome event.
- Predictors at or after the six-hour boundary are prohibited.
- Upstream concepts and keyword matches are evidence sources, not automatically valid labels.
- eICU is not used for feature selection, model selection, threshold tuning or primary calibration.
- Headline evidence is fully external performance, not the best fold or best hospital.
- Missingness and measurement frequency are audited as transportability factors.
- Patient-level data, row-level predictions, credentials and trained restricted-data checkpoints are ignored by Git.
- Protocol deviations require a timestamped amendment written without reference to held-out performance.
- Reporting will follow TRIPOD+AI; risk of bias will be assessed with PROBAST+AI.

## Immediate roadmap

- [x] Initialize privacy-safe repository.
- [x] Lock Phase 0 windows, tasks, metrics and stop rules.
- [x] Add initial harmonized variable schema.
- [x] Add deterministic feasibility-gate code and tests.
- [x] Pin official upstream source repositories and outcome contracts.
- [x] Add MIMIC upstream-concept and event-timing audit SQL.
- [x] Add eICU vocabulary-discovery SQL.
- [x] Add fail-closed training guard and exact boundary tests.
- [ ] Confirm MIMIC-IV v3.1 access and storage mode.
- [ ] Confirm eICU-CRD v2.0 access and storage mode.
- [ ] Run local source audits in credentialed environments.
- [ ] Clinically review and lock database-specific outcome vocabularies.
- [ ] Add final reviewed eICU outcome extractors.
- [ ] Build harmonized six-hour feature extractors.
- [ ] Run cohort and leakage audits before model development.
- [ ] Run locked logistic-regression and CatBoost feasibility baselines.
- [ ] Produce machine-readable Phase 0 go/no-go decision.

## Governance

This is a public code repository. Do not commit raw or derived patient-level data, credentials, private access material, row-level predictions, aggregate exports disallowed by a data-use agreement, or model artifacts trained on credentialed databases unless release is explicitly permitted.

## Primary references

- MIMIC-IV v3.1, PhysioNet, DOI: `10.13026/kpb9-mt58`
- eICU-CRD v2.0, PhysioNet, DOI: `10.13026/C2WM1R`
- HiRID v1.1.1, PhysioNet, DOI: `10.13026/nkwc-js72`
- AmsterdamUMCdb v1.5.0, Amsterdam UMC
- TRIPOD+AI, BMJ 2024, DOI: `10.1136/bmj-2023-078378`
- PROBAST+AI, BMJ 2025, DOI: `10.1136/bmj-2024-082505`
