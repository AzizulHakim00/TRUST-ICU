# TRUST-ECG Locked Baseline and Certification Plan

This document describes the model and evaluation plan that was frozen before any TRUST-ECG waveform-model performance was inspected.

## Scientific role

TRUST-ECG is not an architecture-search study. The primary scientific target is cross-source transportability, calibration failure, and label-efficient probability recovery. The primary waveform model is therefore fixed prospectively and the low-capacity model is a reference only.

## Development split

PTB-XL v1.0.1 uses the official patient-wise `strat_fold` assignments after the Challenge/PTB-XL checksum crosswalk has been verified.

| Fold(s) | Locked role |
|---|---|
| 1-7 | model fitting and training-only statistics |
| 8 | ResNet early-stopping epoch only |
| 9 | per-label Platt calibration only |
| 10 | untouched internal test |

No external source participates in preprocessing, feature scaling, epoch selection, calibration, or model selection.

## Logistic Regression reference

The Logistic Regression model is deliberately low-capacity and is not eligible to replace the predeclared primary model after results are observed.

For each of the 12 leads, the following 12 features are computed from physical-mV samples inside the valid 10-second mask:

1. mean;
2. standard deviation;
3. minimum;
4. maximum;
5. median;
6. 5th percentile;
7. 25th percentile;
8. 75th percentile;
9. 95th percentile;
10. root-mean-square amplitude;
11. mean absolute first difference;
12. standard deviation of first differences.

This produces exactly `12 leads × 12 statistics = 144` features. Padding never contributes.

A `StandardScaler` is fitted on PTB-XL folds 1-7 only. Each locked diagnosis receives an independent binary Logistic Regression with:

- L2 penalty;
- `C=1.0`;
- `lbfgs` solver;
- balanced class weights;
- maximum 2,000 iterations.

## Fixed 1D ResNet primary model

The predeclared waveform baseline is a ResNet18-style 1D network, not a proposed novel architecture.

Locked specification:

- input `12 × 5000`;
- stem: 64 channels, kernel 15, stride 2, padding 7;
- max pool: kernel 3, stride 2, padding 1;
- residual stages: channels `[64, 128, 256, 512]`;
- blocks per stage `[2, 2, 2, 2]`;
- residual kernel size 7;
- downsample at the first block of stages 2-4;
- adaptive average pooling;
- one linear multi-label logit head;
- BCE-with-logits loss;
- positive class weights computed only from folds 1-7;
- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- batch size 64;
- maximum 50 epochs;
- gradient clipping norm 5;
- no primary data augmentation;
- seed `20260808`.

The only performance-informed training choice allowed is the stopping epoch using fold-8 macro PR-AUC, with patience 7 and minimum improvement `1e-4`. The selected epoch is not used to refit the network on folds 8 or 9.

## Probability calibration

Each diagnosis is calibrated independently on PTB-XL fold 9 by fitting an unregularized scalar Logistic Regression to the raw model logit/decision score.

No Phase-0 external recalibration is allowed.

## Independent external partition

For each external source, a label-blind SHA-256 function of the locked seed, source name, and record ID assigns records to:

- 60% `certification`;
- 40% `recovery_pool`.

This assignment occurs without diagnosis labels and cannot be changed after outcomes or performance are inspected.

The recovery pool is not used in Phase 0.

## Primary external evaluation unit

The unit is a prespecified `diagnosis × external-domain` pair, evaluated using the fixed ResNet on the 60% certification partition.

At least 50 positives and 50 negatives are required. Otherwise the pair is `insufficient_support` rather than silently removed.

### Discrimination viability

`PR-AUC / prevalence >= 2.0`

### Calibration research envelope

All are required:

- absolute calibration-slope deviation from 1 no greater than 0.35;
- absolute calibration intercept no greater than 0.75;
- positive Brier skill relative to the constant-prevalence predictor.

These thresholds are protocol-defined research endpoints. They are not claims of clinical, regulatory, or bedside safety.

## Four pair states

- `certified`: discrimination viable and all calibration checks pass;
- `calibration_recovery_candidate`: discrimination viable but at least one calibration check fails;
- `discrimination_failure`: discrimination viability fails;
- `insufficient_support`: certification partition lacks minimum class support.

Only `calibration_recovery_candidate` pairs are scientifically appropriate for simple recalibration recovery because probability mapping cannot repair failed ranking.

## Phase-1 activation

Recovery analysis requires at least two independent external domains containing at least one calibration-recovery candidate. It uses the untouched 40% recovery pool only.

For budgets `0, 50, 100, 250, 500, 1000`, 100 uniform-without-replacement target-label samples are repeated with the locked seed. Adaptation records are excluded from that repeat's evaluation records. Target-domain model retraining, feature selection, normalization refitting, and post-hoc threshold search remain prohibited.

## Implementation status

The deterministic 144-feature extractor, fixed Logistic Regression reference, per-label Platt calibration utilities, label-blind external partition, binary discrimination/calibration metrics, and pair-state logic are implemented and unit-tested.

The fixed ResNet architecture may be implemented as an execution dependency, but it must not be tuned or trained until the real header, label, and waveform audits all pass.
