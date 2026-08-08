# Conditional Phase 1 — Deployment Certification

Phase 1 is deliberately inactive until at least one task passes the locked Phase 0 feasibility gate. This document defines the next scientific study without prematurely implementing a new architecture.

## Motivation

The August 2026 collision audit found that joint ICU intervention forecasting, multi-database transfer learning, generic conformal selective prediction, and hierarchical/site-specific conformal calibration are already materially represented in recent literature. See `docs/literature_collision_audit_2026-08-08.md`.

The Phase 1 focus is therefore operational transportability:

> When a frozen intervention-forecasting model is moved to a new hospital, can we identify unsafe deployment, explain the failure using prespecified shift diagnostics, and determine the amount of local labeled data required to restore calibration and risk control?

## Activation

Credential-free validation:

```bash
python scripts/validate_phase1_plan.py
```

Expected public status before real Phase 0 completion:

```text
awaiting_verified_phase0_report
architecture_or_method_development_allowed = false
```

After the locked clinical Phase 0 run:

```bash
python scripts/validate_phase1_plan.py \
  --phase0-report /secure/trust_icu_phase0_results/phase0_go_no_go.json \
  --require-active
```

Only tasks whose Phase 0 `continue_to_architecture_development` flag is true become eligible. Failed tasks stay frozen.

## Frozen inputs

For an eligible task, Phase 1 inherits unchanged:

- outcome definition;
- six-hour observation window;
- 12-hour prediction horizon;
- feature contract;
- development-selected model;
- MIMIC-only calibration map;
- Phase 0 external evaluation;
- Phase 0 go/no-go result.

Phase 1 is not permission to reopen the Phase 0 leaderboard.

## Hospital-level deployment unit

The primary deployment unit is an eICU hospital. A hospital enters the primary site-metric analysis when it has at least:

- 200 eligible rows;
- 20 positive events;
- 20 negative events.

Smaller hospitals remain in aggregate descriptive counts but cannot produce unstable primary site metrics.

Hospital identifiers may exist inside secure runtime but must never appear in public outputs.

## Site certification envelope

A site is not certified when any required prespecified safety condition fails. The initial envelope inherits the Phase 0 calibration and normalized PR-AUC requirements:

- PR-AUC / prevalence >= 2.0;
- calibration slope within 0.35 of 1;
- absolute calibration intercept <= 0.75;
- no post-index leakage.

These thresholds cannot be relaxed after hospital-level results are inspected.

## Shift diagnostics

Prespecified label-free diagnostics include:

- absolute missingness difference;
- measurement-frequency shift;
- standardized mean difference;
- median shift;
- univariate Wasserstein distance;
- categorical case-mix shift.

Outcome-prevalence shift is secondary because it requires labels.

These diagnostics are used to characterize transportability failure. They are not automatically a new predictive model and must not be turned into a post-hoc score after inspecting the final site test results.

## Local-label budget experiment

Fixed local sample budgets:

```text
0, 50, 100, 250, 500, 1000
```

For each feasible budget, repeated local samples will evaluate:

1. frozen model with no update;
2. intercept-only recalibration;
3. Platt recalibration.

Model retraining, fine-tuning, and target-site feature selection are prohibited in this study. The objective is to estimate the data required to make an existing model locally usable, not to train a new target model.

## Secondary uncertainty benchmarks

The following may be benchmarked but are not novelty claims:

- split conformal prediction;
- Mondrian/group-conditional conformal prediction;
- weighted conformal prediction;
- fixed-cost selective deferral.

The project must not call these methods new merely because they are applied to TRUST-ICU outcomes.

## Primary Phase 1 outputs

For each activated task:

- number of hospitals eligible for site-level evaluation;
- fraction of hospitals failing certification;
- median, lower-tail, and worst-site calibration;
- median, lower-tail, and worst-site normalized PR-AUC;
- association between prespecified shift descriptors and site failure;
- certification recovery rate by local-label budget;
- minimum local-label budget required for recovery where recoverable;
- hospital-cluster uncertainty intervals.

No best-hospital result is a primary claim.

## Independent replication

AmsterdamUMCdb and HiRID are later database-level replication cohorts only after the eICU Phase 1 analysis plan is frozen. They may confirm or contradict the eICU findings, but they must not be used to tune the Phase 1 certification thresholds.

## What is intentionally not implemented yet

No new neural encoder, domain-adaptation architecture, conformal-risk-control method, site-shrinkage method, or deployment meta-model is implemented at this stage.

That restraint is intentional. The repository will only develop additional methods after:

1. a real Phase 0 task passes;
2. the literature audit is refreshed;
3. the exact remaining gap is still defensible;
4. the Phase 1 analysis plan is prospectively frozen.
