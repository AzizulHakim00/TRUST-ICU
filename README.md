# TRUST-ICU

**Transportable and Risk-Controlled Prediction of Acute Organ-Support Escalation Across Multiple ICUs**

TRUST-ICU is a protocol-first clinical prediction study designed to test whether early ICU models remain useful, calibrated and safe across independent hospitals and databases. The project does not proceed to a complex architecture unless a locked external-validation feasibility gate is passed.

## Current phase

**Phase 0: MIMIC-IV → eICU feasibility validation**

Planned primary tasks predict new initiation during the 12 hours after a six-hour ICU observation window:

1. invasive mechanical ventilation;
2. vasopressor therapy;
3. renal-replacement therapy.

Each task has its own eligible cohort. Patients already receiving the relevant support by the observation-window end are excluded from that task.

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

## Repository structure

```text
configs/
  feasibility.yaml          # Machine-readable preregistration and stop rules
docs/
  phase0_protocol.md        # Study design, estimands and validation protocol
  data_access.md            # Credentialing and data-governance checklist
schemas/
  common_variables.csv      # Initial cross-database variable dictionary
scripts/
  check_environment.py      # Access-path readiness check; reads no patient data
src/trust_icu/
  config.py                 # Typed configuration validation
  validation.py             # Aggregate-only deterministic feasibility gates
tests/
  test_validation.py        # Synthetic tests for go/no-go decisions
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
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
- Predictors at or after the six-hour boundary are prohibited.
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
- [ ] Confirm MIMIC-IV v3.1 access and storage mode.
- [ ] Confirm eICU-CRD v2.0 access and storage mode.
- [ ] Lock database-specific outcome definitions and code lists.
- [ ] Build synthetic timeline fixtures for outcome unit tests.
- [ ] Create MIMIC-IV extraction queries.
- [ ] Create eICU extraction queries.
- [ ] Run cohort and leakage audits before model development.
- [ ] Run locked logistic-regression and CatBoost feasibility baselines.
- [ ] Produce machine-readable Phase 0 go/no-go decision.

## Governance

This is a public code repository. Do not commit raw or derived patient-level data, credentials, private access material, row-level predictions, or model artifacts trained on credentialed databases unless release is explicitly permitted by the source-data agreement.

## Primary references

- MIMIC-IV v3.1, PhysioNet, DOI: `10.13026/kpb9-mt58`
- eICU-CRD v2.0, PhysioNet, DOI: `10.13026/C2WM1R`
- HiRID v1.1.1, PhysioNet, DOI: `10.13026/nkwc-js72`
- AmsterdamUMCdb v1.5.0, Amsterdam UMC
- TRIPOD+AI, BMJ 2024, DOI: `10.1136/bmj-2023-078378`
- PROBAST+AI, BMJ 2025, DOI: `10.1136/bmj-2024-082505`
