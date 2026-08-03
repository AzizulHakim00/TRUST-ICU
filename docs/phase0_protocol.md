# TRUST-ICU Phase 0 Feasibility Protocol

## 1. Purpose

TRUST-ICU will test whether routinely collected ICU data can support transportable, calibrated prediction of **new organ-support initiation** across independent databases and hospitals. Phase 0 is a feasibility study, not an architecture competition. No deep model will be developed until the predefined feasibility gates are passed.

## 2. Primary research question

Using information available during the first six hours after ICU admission, can a model predict initiation of each organ support during the following 12 hours and retain useful discrimination and calibration in a fully external multi-hospital cohort?

## 3. Target population

- Adults aged 18 years or older.
- First eligible ICU stay per patient.
- ICU stay must cover the complete six-hour observation window.
- Patients already receiving the organ support of interest by the end of the observation window are excluded from that task only.
- Dataset-specific exclusions must be documented before outcomes are inspected.

Each outcome uses a task-specific eligible cohort. Cohort counts, exclusions and event prevalence must be reported separately for every database and task.

## 4. Index, observation and prediction windows

- Index time: ICU admission.
- Observation window: `[0 h, 6 h)` after ICU admission.
- Prediction window: `[6 h, 18 h)` after ICU admission.
- Predictors occurring at or after 6 h are prohibited.
- Outcome evidence before 6 h is used only for task-specific exclusion, never as a predictor.

Time boundaries are left-closed and right-open to prevent duplicate assignment at exactly 6 h.

## 5. Outcomes

Three independent binary tasks are planned:

1. New invasive mechanical ventilation.
2. New vasopressor initiation.
3. New renal-replacement therapy.

Before extraction, each outcome requires a versioned operational definition for every database containing:

- source tables and fields;
- accepted item, drug or procedure identifiers;
- start-time derivation;
- exclusion logic;
- handling of short interruptions and duplicate records;
- unit tests using synthetic timelines;
- clinician review or authoritative code-list provenance.

The definitions must represent clinically equivalent interventions across databases. If equivalence cannot be established, the affected task fails Phase 0.

## 6. Predictors

Only variables plausibly available by the observation-window end are eligible. The initial harmonized dictionary is stored in `schemas/common_variables.csv`.

For each time-varying variable, Phase 0 may derive only prespecified summaries:

- first value;
- last value;
- minimum;
- maximum;
- mean;
- simple slope when at least two timestamps exist;
- measurement count;
- missingness indicator;
- time since last measurement.

Dataset identity, hospital identity, discharge information, future treatment information and post-index severity scores are not predictors. Hospital identifiers may be retained only for grouped evaluation.

## 7. Data sources and roles

- **MIMIC-IV v3.1:** development and temporal validation.
- **eICU-CRD v2.0:** primary external multi-hospital validation.
- **AmsterdamUMCdb v1.5.0:** later European external validation after Phase 0.
- **HiRID v1.1.1:** later high-resolution European external validation after Phase 0.

Phase 0 proceeds with MIMIC-IV and eICU only. AmsterdamUMCdb and HiRID are not required to decide whether architecture development is justified.

## 8. Phase 0 models

Only two prespecified models are allowed:

- regularized logistic regression;
- CatBoost.

No model is selected using external eICU results. Hyperparameters are selected only within MIMIC-IV development data. Any preprocessing, imputation, feature filtering, class weighting or calibration is fitted without access to held-out outcomes.

## 9. Validation design

### 9.1 Development validation

MIMIC-IV is divided temporally using calendar-time information available under the data agreement. Earlier admissions are used for development and later admissions for temporal evaluation. The split date is chosen from data availability, not performance.

### 9.2 Primary external validation

The locked MIMIC-IV pipeline is applied unchanged to eICU. No eICU labels may be used for model fitting, feature selection, threshold selection or calibration in the primary external analysis.

### 9.3 Hospital-level analysis

Within eICU, performance is summarized across hospitals meeting prespecified minimum sample and event-count criteria. Report:

- median hospital performance;
- interquartile range;
- worst decile;
- number of hospitals with estimable metrics;
- relation between hospital sample size, prevalence and performance.

Hospital-level results are evaluation only and must not be used to tune the primary model.

## 10. Primary estimands and metrics

The primary evidence is fully external eICU performance.

Primary metrics:

- area under the precision-recall curve;
- Brier score;
- calibration slope;
- calibration intercept.

Secondary metrics:

- ROC-AUC;
- sensitivity and specificity at a development-locked threshold;
- F1 and MCC;
- decision-curve net benefit;
- subgroup and hospital-level performance.

All key estimates require confidence intervals. Best-fold or best-hospital results are prohibited as headline evidence.

## 11. Missing data

Missingness is treated as part of transportability, not hidden by complete-case analysis.

- Missing-data handling is fitted on development data only.
- Missingness indicators and measurement frequency are permitted when prespecified.
- Dataset and hospital interface differences must be audited.
- Complete-case analysis is sensitivity analysis only.
- Variables unavailable in an external database cannot be silently replaced by proxies after outcome inspection.

## 12. Feasibility decision

The machine-readable gates are defined in `configs/feasibility.yaml`. Phase 0 continues to architecture development only when all mandatory checks pass, including:

- sufficient positive events in development and external cohorts;
- clinically equivalent outcome definitions;
- no post-index leakage;
- external PR-AUC meaningfully above prevalence;
- recoverable external calibration;
- adequate multi-hospital representation.

Failure means the affected task is stopped, redesigned before modelling, or removed. It does not trigger additional architecture searches on the same frozen data.

## 13. Reproducibility and data governance

- Restricted raw data and patient-level derivatives remain outside Git.
- Repository code receives only configurable local paths.
- All cohort definitions, code lists and configuration files are versioned.
- Runs record dependency versions, dataset versions, configuration hash and Git commit.
- Public artifacts contain aggregate statistics only unless the data-use agreement explicitly permits more.
- Models trained on credentialed data are treated as potentially sensitive and are not publicly released without source-dataset approval.

## 14. Reporting and risk-of-bias control

The final study will follow TRIPOD+AI reporting guidance and will be self-assessed using PROBAST+AI before submission. Deviations from this protocol must be timestamped, justified without reference to held-out performance, and documented in a separate amendment file.

## 15. Phase 0 deliverables

1. Access and environment audit.
2. Versioned cross-database outcome definitions.
3. Harmonized variable availability matrix.
4. Cohort flow and event counts.
5. Leakage audit.
6. Logistic-regression and CatBoost temporal baselines.
7. Locked external eICU evaluation.
8. Hospital-level heterogeneity analysis.
9. Machine-readable go/no-go decision.
