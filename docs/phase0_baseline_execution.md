# Phase 0 Baseline Execution

This stage starts only after both canonical source adapters pass their credentialed audits and all three outcome contracts are locally validated, clinically equivalent, and locked.

## Locked analysis design

For each outcome separately:

1. Construct the six-hour landmark cohort.
2. Exclude support already active during `[0 h, 6 h)`.
3. Exclude death without the target support from the primary binary analysis.
4. Build the same canonical feature matrix in MIMIC-IV and eICU.
5. Order MIMIC-IV admissions by ICU admission time.
6. Assign the earliest 70% to training, the next 15% to calibration, and the latest 15% to temporal testing.
7. Purge a patient from every later split when that patient appeared in an earlier split.
8. Fit the locked Logistic Regression and CatBoost baselines on MIMIC training data only.
9. Fit Platt calibration using only the MIMIC calibration split.
10. Select the baseline using calibrated MIMIC temporal PR-AUC, then Brier score, then Logistic Regression as the simplicity tie-breaker.
11. Apply both prespecified models and the unchanged MIMIC calibration maps to eICU once.
12. Apply the preregistered feasibility gate to the development-selected model.

External labels cannot be used for feature selection, model selection, hyperparameter tuning, threshold tuning, or recalibration.

## Input verification

The runner verifies before loading patient rows:

- dataset identity;
- `credentialed_run_report.json` SHA-256;
- equality between the embedded audit and `canonical_audit.json`;
- `ready_for_cohort_build=true`;
- absence of critical audit failures;
- SHA-256 and byte count of stays, events, and observations exports.

A mismatch stops execution.

## Scalable feature construction

The observation export can contain millions of rows. DuckDB reads the compressed CSV directly and performs the six-hour aggregation without first loading raw observations into pandas. The output matrix contains only the locked summaries:

- first, last, minimum, maximum, mean;
- linear slope per hour;
- count and missingness;
- hours since the last measurement.

Values outside the public physiological bounds do not enter model features. Observations before ICU admission or at/after the landmark cause the final execution to fail.

## Run

Install all Phase 0 dependencies:

```bash
pip install -e ".[dev,ml,analytics]"
```

Credential-free plan:

```bash
python scripts/run_phase0_baselines.py --dry-run
```

Real execution inside the authorized environment:

```bash
export TRUST_ICU_PHASE0_OUTPUT_ROOT=/secure/trust_icu_phase0_results

python scripts/run_phase0_baselines.py \
  --mimic-run-dir /secure/trust_icu_outputs/mimic_iv_3_1 \
  --eicu-run-dir /secure/trust_icu_outputs/eicu_crd_2_0
```

## Aggregate-only outputs

```text
$TRUST_ICU_PHASE0_OUTPUT_ROOT/
  phase0_go_no_go.json
  phase0_aggregate_summary.csv
```

The runner does not save:

- patient-level predictions;
- patient or hospital identifiers;
- fitted models;
- calibration objects;
- row-level split assignments.

The JSON includes input and contract hashes, feature audits, temporal split counts, internal and external metrics, hospital-cluster bootstrap intervals, aggregate hospital robustness, missingness shift, and the deterministic task-wise go/no-go decisions.

## Hospital analysis

The selected baseline receives a hospital-cluster bootstrap on eICU. Hospital-specific metrics are summarized only when a site has the configured minimum rows, positive events, and negative events. The public report contains aggregate quantiles and worst-case values, not hospital identifiers.

## Stop behavior

Execution stops when:

- an outcome contract is not fully locked;
- an input report or export hash is invalid;
- cross-database model columns differ;
- a temporal split becomes empty after patient purging;
- any split lacks both classes;
- external hospital identifiers are missing;
- feature leakage or timing violations are found;
- required modelling dependencies are absent.

A failed feasibility gate is a scientific result. It must not trigger repeated searches against the same external cohort.
