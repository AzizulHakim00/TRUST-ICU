# TRUST-ECG conditional Phase 1: label-efficient probability recovery

TRUST-ECG Phase 1 is a **conditional calibration-recovery experiment**, not a new model-development
stage. It is allowed only after the prospectively fixed ResNet completes Phase 0.

## Pre-ResNet implementation lock

The executable Phase-1 implementation was completed and passed both the ECG-specific integrity suite
and the full repository CI **before the primary fixed-ResNet real-data run was activated**. The
scientific implementation is locked to commit
`4b5189d693ddd9ede6715cc3210b9157972144e2` for the ResNet workflow. The workflow fails closed if
`src/trust_icu/ecg_phase1.py` or `scripts/run_open_ecg_phase1.py` drifts from that lock.

This timing prevents Phase-1 sampling, recalibration, or reporting logic from being changed after
observing the primary ResNet transportability result. It does not alter protocol v0.4, the frozen
ResNet architecture, label set, statistical splits, Phase-0 thresholds, or Phase-0 data boundary.

The conditional Phase-1 runner executes in the **same isolated GitHub Actions job** as the fixed
ResNet. This is deliberate: the checkpoint and fold-9 calibration remain local ephemeral files and
never need to be uploaded or committed. The recovery pool remains inaccessible until the completed
ResNet report is hash-verified and satisfies the prospective Phase-1 activation rule.

## Activation gate

The primary fixed-ResNet Phase-0 report is hash-verified before any recovery-pool waveform is
loaded. Phase 1 activates only when `calibration_recovery_candidate` pairs occur in at least two of
the three locked external domains:

- Georgia;
- CPSC2018;
- CPSC2018-Extra.

`certified`, `discrimination_failure`, and `insufficient_support` pairs cannot enter recovery.
Recalibration is therefore never used to rescue a ranking failure.

If fewer than two external domains contain a recovery candidate, Phase 1 remains disabled and the
negative/positive transportability result is reported as observed.

## Frozen recovery design

The untouched 40% external recovery pool is the only Phase-1 data source. The target-label budgets
remain exactly:

```text
0, 50, 100, 250, 500, 1000
```

For each eligible label-domain pair and nonzero budget, 100 deterministic repeated samples are drawn
uniformly without replacement. Sampling is **not label-stratified**. The same adaptation draw is
used for the frozen, intercept-only, and Platt comparisons within a repeat.

Adaptation records are removed from that repeat's evaluation set. A repeat is non-estimable rather
than imputed when its evaluation remainder contains fewer than 20 positives or fewer than 20
negatives for the target label. If a requested budget exceeds the frozen recovery-pool size, all
available records are used and the condition is explicitly flagged.

## Compared methods

The comparison contains exactly three states:

1. `frozen_no_update` — the fixed Phase-0 ResNet plus its PTB-XL fold-9 global Platt calibration;
2. `intercept_only_recalibration` — a target-domain log-odds intercept update with slope fixed at 1;
3. `platt_recalibration` — a scalar target-domain slope and intercept update.

Local recalibration operates on the logit of the already frozen Phase-0 probability. This preserves
the Phase-0 model, preprocessing, normalization, feature representation, and architecture.

At budget 0, only `frozen_no_update` is estimable because no target labels exist with which to fit a
local calibrator.

A nonzero adaptation draw containing only one class makes the two local recalibration methods
non-estimable for that repeat. The frozen comparator remains evaluable if the held-out remainder
satisfies the Phase-1 support rule.

## Recovery endpoint

For every estimable repeat, the implementation reports the same research envelope used in Phase 0:

- PR-AUC/prevalence ratio >= 2.0;
- absolute calibration-slope deviation from 1 <= 0.35;
- absolute calibration intercept <= 0.75;
- positive Brier skill versus prevalence.

The Phase-1 evaluation support floor remains 20 positives and 20 negatives, as locked in protocol
v0.4. No threshold is re-estimated from target-domain performance.

For each label-domain pair, budget, and method, public output contains only aggregate quantities:
estimable/non-estimable repeat counts, non-estimability reasons, recovery-envelope success rate, and
median with 2.5th/97.5th percentiles for the prespecified discrimination/calibration metrics.

## Prohibited operations

The Phase-1 runner does not permit:

- target-domain model retraining or fine-tuning;
- target-domain feature selection;
- target-domain normalization refit;
- architecture selection;
- post-hoc decision-threshold search;
- best-domain-only reporting;
- use of Phase-0 certification records as adaptation data.

## Hash and data boundary

Before inference, the runner verifies that the supplied checkpoint, global fold-9 calibration,
model index, label manifest, normalization statistics, protocol, and primary Phase-0 report belong
to the same frozen model state. The recovery runner loads only rows whose role is
`external_recovery_pool` and only from external domains containing prospectively eligible pairs.

No waveform, checkpoint, record-level prediction, sampled record identifier, or patient-level
artifact is written to the public repository. The Phase-1 JSON report is aggregate-only.
