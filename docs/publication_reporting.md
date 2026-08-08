# Publication Reporting Bundle

This stage begins only after a valid aggregate `phase0_go_no_go.json` has been produced by the locked Phase 0 pipeline.

The reporting layer is deliberately downstream of modelling. It cannot select models, retune thresholds, recalibrate on eICU, or alter the preregistered feasibility decision.

## Standards used

The bundle is structured to support transparent reporting under TRIPOD+AI (BMJ 2024;385:e078378) and an internal evidence review across the four PROBAST+AI domains (BMJ 2025;388:e082505): participants and data sources, predictors, outcome, and analysis.

The repository does **not** reproduce the full copyrighted checklist text. `tripod_ai_traceability.csv` is a project-specific evidence-to-section map. `probast_ai_self_audit.csv` is not a formal low/high risk-of-bias judgment and cannot replace independent methodological assessment.

## Integrity rules

Before creating any publication artifact, the generator verifies:

- the Phase 0 report SHA-256;
- that exactly the three primary outcome tasks are present;
- that no patient, stay, admission, hospital, or row-level prediction identifiers are embedded in the report;
- that the selected model is explicitly marked as selected using development data only;
- the reporting contract and all generated file hashes.

Tampered or identifier-bearing inputs stop execution.

## Dry run

```bash
pip install -e ".[dev,phase0,reporting]"
python scripts/generate_publication_bundle.py --dry-run
```

## Real aggregate run

Keep generated files outside the Git repository until manuscript and disclosure review are complete.

```bash
export TRUST_ICU_REPORTING_OUTPUT_ROOT=/secure/trust_icu_reporting

python scripts/generate_publication_bundle.py \
  --phase0-report /secure/trust_icu_phase0_results/phase0_go_no_go.json
```

## Outputs

```text
trust_icu_reporting/
├── reproducibility_manifest.json
├── tripod_ai_traceability.csv
├── probast_ai_self_audit.csv
├── table_1_cohort_summary.csv
├── table_2_model_performance.csv
├── table_3_external_robustness.csv
├── figure_external_performance.png
└── figure_calibration_transportability.png
```

### Table 1

Contains task-level development/external sample sizes, event counts, prevalence, development-selected model, and the deterministic feasibility decision.

### Table 2

Contains both prespecified models across MIMIC temporal and eICU external evaluations, before and after the locked MIMIC-only calibration map. Metrics include PR-AUC, prevalence-normalized PR-AUC, ROC-AUC, Brier score, calibration slope, and calibration intercept.

### Table 3

Contains aggregate hospital robustness and hospital-cluster bootstrap confidence intervals. It does not expose hospital identifiers.

### Figures

The first figure compares external PR-AUC with outcome prevalence for the development-selected model. The second summarizes external calibration slope and intercept relative to their ideal values.

The figures are descriptive reporting outputs only. They do not introduce new selection criteria or statistical tests.

## Reproducibility manifest

`reproducibility_manifest.json` records:

- source Phase 0 report hash;
- reporting contract hash;
- config, feature-contract, and outcome-contract hashes;
- every generated file SHA-256 and byte count;
- whether all preregistered feasibility gates passed.

This makes the manuscript reporting bundle traceable to one immutable Phase 0 result.

## Publication rule

A failed go/no-go decision must remain visible in Table 1 and the manifest. The reporting layer must not suppress a failed task, replace the selected model using external performance, or generate a new post-hoc primary analysis.
