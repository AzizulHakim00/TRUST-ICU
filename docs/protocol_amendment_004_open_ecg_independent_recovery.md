# Protocol Amendment 004 — Independent External Certification and Recovery Evaluation

**Date:** 2026-08-08  
**Status:** prospective; committed before any TRUST-ECG waveform-model performance was inspected.

## Problem identified

The initial Phase 0/Phase 1 concept evaluated a frozen model on an external source and then proposed using labeled examples from that same source to study recalibration recovery. If the entire source were first used to decide that a label/domain had failed and the same observations were subsequently reused to quantify recovery, the recovery estimate would be conditioned on an observed failure in the evaluation data. That can create regression-to-the-mean and selection bias.

A second problem was conceptual: simple probability recalibration cannot repair a model whose ranking/discrimination has collapsed. Treating all external failures as recalibration candidates would waste experiments and blur the scientific question.

## Prospective external partition

Before any model score is computed, every external source is deterministically divided by a label-blind SHA-256 function of:

- the fixed study seed `20260808`;
- source name;
- record identifier.

The locked fractions are:

- **60% certification partition** — Phase 0 external transportability only;
- **40% untouched recovery pool** — reserved for Phase 1 and never inspected for Phase 0 model decisions.

The partition is not label-stratified. This intentionally prevents outcome-informed partition construction. Because patient identifiers are not consistently available across the Challenge external sources, patient-level separation cannot be guaranteed for those sources; this will be reported as a limitation rather than silently inferred.

No repartitioning is allowed after label prevalence or model performance is inspected.

## Label-domain pair as the primary transportability unit

Multi-label ECG transportability is not reduced to one global score. Each prespecified diagnosis and external source forms a `label x domain` pair.

A pair is evaluated in Phase 0 only when the **certification partition** has at least:

- 50 positive records;
- 50 negative records.

Pairs below those counts are `insufficient_support`; the label is not deleted from the study.

## Prospective research envelope

The predeclared primary model is the fixed 1D ResNet. Logistic Regression is a low-capacity reference and cannot replace the primary model because it performs better after results are seen.

For each evaluable label/domain pair:

### Discrimination viability

`PR-AUC / outcome prevalence >= 2.0`

### Calibration envelope

All of the following are required:

- `|calibration slope - 1| <= 0.35`;
- `|calibration intercept| <= 0.75`;
- positive Brier skill relative to the constant-prevalence predictor.

These are research protocol thresholds, **not clinical, regulatory, or bedside safety standards**.

## Four Phase 0 pair states

1. `certified` — discrimination remains viable and every calibration check passes;
2. `calibration_recovery_candidate` — discrimination remains viable but at least one calibration check fails;
3. `discrimination_failure` — PR-AUC/prevalence viability fails;
4. `insufficient_support` — certification partition has inadequate positive/negative support.

The distinction matters because Platt or intercept recalibration changes probability mapping, not the ordering of predictions. A discrimination failure is therefore not presented as something that simple recalibration should rescue.

## Phase 1 activation

Primary label-efficient recovery analysis activates only if **at least two independent external domains** contain at least one `calibration_recovery_candidate`.

This requirement prevents a single-source calibration anomaly from being promoted into a general recovery claim.

## Untouched recovery pool analysis

For each activated external domain, the 40% recovery pool remains untouched until Phase 1.

For each budget `0, 50, 100, 250, 500, 1000`:

- draw target records uniformly without replacement;
- use them only for allowed probability-level localization;
- exclude those adaptation records from that repeat's evaluation set;
- repeat 100 times with the locked seed sequence;
- do not outcome-balance or label-stratify the acquisition sample;
- report a repeat as non-estimable instead of imputing a result when its remaining evaluation set lacks 20 positives or 20 negatives for a label.

If a requested budget exceeds the available recovery pool, use all available records and explicitly flag the budget as truncated.

## Allowed and prohibited recovery operations

Allowed:

- intercept-only recalibration;
- Platt recalibration.

Prohibited in the primary recovery study:

- target-domain model retraining;
- target-domain feature selection;
- target-domain normalization refitting;
- post-hoc operating-threshold search;
- selecting the best external domain for reporting.

## Baseline lock introduced at the same time

Because no waveform-model performance has yet been observed, this amendment also freezes the Phase 0 baselines:

- one 144-feature deterministic per-lead summary Logistic Regression reference;
- one ResNet18-style fixed 1D waveform baseline as the primary model;
- no architecture search;
- only fold-8 early-stopping epoch selection for the ResNet;
- independent Platt calibration on PTB-XL fold 9;
- PTB-XL fold 10 remains untouched internal testing.

This is deliberately conservative. The paper's candidate contribution is transportability evidence and recovery behavior, not optimization of a new ECG architecture.
