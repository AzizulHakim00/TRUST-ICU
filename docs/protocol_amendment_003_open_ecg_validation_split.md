# Protocol Amendment 003 — Separate ECG Optimization, Calibration, and Test Data

**Date:** 2026-08-08  
**Status:** prospective; made before any TRUST-ECG waveform model was trained or any internal/external performance was inspected.

## Problem identified

The initial open ECG protocol assigned PTB-XL folds 1-8 to training, fold 9 to calibration, and fold 10 to internal testing. That is adequate only if every model-training detail, including the training duration of the fixed 1D ResNet, is fully predetermined without validation feedback.

For a neural baseline, using the calibration fold to choose an early-stopping epoch would contaminate probability calibration. Using the internal-test fold would invalidate the test. Fixing an arbitrary epoch without a development validation set would also be difficult to defend.

## Prospective change

The official patient-wise PTB-XL folds are now assigned as follows:

| Fold(s) | Locked purpose |
|---|---|
| 1-7 | model fitting and training-only preprocessing statistics |
| 8 | optimization validation; ResNet early-stopping epoch only |
| 9 | probability calibration only |
| 10 | untouched internal test |

All non-epoch Phase 0 hyperparameters must be fixed before training. Fold 8 is not folded back into the final model fit. Fold 9 is never used for architecture, feature, epoch, or hyperparameter selection. Fold 10 is never used for any model-development choice.

## Why this is more conservative

This design gives up part of the available fitting data in exchange for four non-overlapping statistical roles. It prevents a common source of optimistic bias: using the same observations both to tune a neural model and to estimate/calibrate its probabilities.

The official PTB-XL release documents that `strat_fold` is patient-wise and recommends folds 1-8 for training, 9 for validation, and 10 for testing. TRUST-ECG uses the same patient-wise fold construction but further subdivides the development portion prospectively to isolate optimization from calibration.

## Signal preprocessing change

The primary signal path is also clarified prospectively:

- physical-unit conversion to millivolts;
- resampling to 500 Hz only when required;
- deterministic 10-second center crop or symmetric zero padding with a mask;
- per-lead normalization fitted only on folds 1-7 with padded values excluded;
- **no additional band-pass, notch, baseline-removal, denoising, or source-specific filtering in the primary analysis.**

Any filtered representation requires a new prospective sensitivity-analysis amendment before results are inspected and cannot replace the primary analysis.

## No outcome or external-data change

This amendment does not change:

- source roles;
- label support thresholds;
- external datasets;
- primary metrics;
- Phase 0 go/no-go thresholds;
- Phase 1 label budgets;
- prohibition on external-domain tuning.

No ECG model performance was available when this amendment was committed.
