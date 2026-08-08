# TRUST-ECG Phase 0 model execution

This stage starts only after the header audit, label lock, waveform audit, and model-index audit have all passed on the real public PhysioNet/CinC Challenge 2020 data.

## Statistical roles

The model index is the source of truth for record roles:

- PTB-XL folds 1-7: model fitting.
- PTB-XL fold 8: optimization validation for the fixed ResNet early-stopping epoch only.
- PTB-XL fold 9: independent Platt calibration.
- PTB-XL fold 10: untouched internal test.
- Georgia/CPSC2018/CPSC2018-Extra deterministic 60% partitions: Phase 0 external certification.
- External deterministic 40% partitions: untouched recovery pools and prohibited in Phase 0.

The external partition is label-blind and derived from the frozen SHA-256 record-ID rule in the protocol. It must not be repartitioned after inspecting labels or performance.

## Install

For the Logistic Regression reference:

```bash
pip install -e ".[dev]"
```

For the primary fixed ResNet:

```bash
pip install -e ".[dev,ecg-deep]"
```

PyTorch is an optional dependency so the normal protocol/audit CI remains lightweight.

## Dry run

```bash
python scripts/run_open_ecg_phase0.py --model logistic --dry-run
python scripts/run_open_ecg_phase0.py --model resnet --dry-run
```

Dry-run mode does not read ECG data and does not import PyTorch for the ResNet plan.

## Logistic reference

The handcrafted Logistic Regression model is a low-capacity reference only. It cannot activate the primary Phase 0 gate even if it performs well.

```bash
python scripts/run_open_ecg_phase0.py \
  --model logistic \
  --challenge-training-root /secure/challenge-2020/training \
  --model-index /secure/trust_ecg/model_index/open_ecg_model_index.csv \
  --model-index-audit /secure/trust_ecg/model_index/open_ecg_model_index_audit.json \
  --label-manifest /secure/trust_ecg/open_ecg_label_manifest.json \
  --output-root /secure/trust_ecg/phase0
```

The runner:

1. verifies the model-index audit SHA-256;
2. recomputes the record-level index hash;
3. excludes every `external_recovery_pool` record before any waveform load;
4. extracts exactly 144 physical-mV handcrafted features;
5. fits one fixed Logistic Regression per locked label on PTB-XL folds 1-7;
6. fits independent Platt calibrators on PTB-XL fold 9;
7. evaluates PTB-XL fold 10;
8. evaluates only the 60% certification partitions of all three external sources;
9. writes an aggregate hashed report only.

Output:

- `open_ecg_phase0_logistic_report.json`

## Primary fixed ResNet

The ResNet is the prospectively declared primary Phase 0 model. No architecture search is allowed.

```bash
python scripts/run_open_ecg_phase0.py \
  --model resnet \
  --challenge-training-root /secure/challenge-2020/training \
  --model-index /secure/trust_ecg/model_index/open_ecg_model_index.csv \
  --model-index-audit /secure/trust_ecg/model_index/open_ecg_model_index_audit.json \
  --label-manifest /secure/trust_ecg/open_ecg_label_manifest.json \
  --waveform-audit /secure/trust_ecg/open_ecg_waveform_audit.json \
  --normalization-stats /secure/trust_ecg/open_ecg_normalization_stats.json \
  --output-root /secure/trust_ecg/phase0 \
  --device cuda \
  --num-workers 0
```

The runner verifies that the normalization-statistics hash is the one frozen by the waveform audit. It then uses the exact protocol values: fixed 1D ResNet, AdamW, learning rate 0.001, weight decay 0.0001, batch size 64, maximum 50 epochs, fold-8 macro PR-AUC early stopping, patience 7, minimum delta 0.0001, gradient clipping at 5.0, no primary augmentation, and seed 20260808.

The best fold-8 epoch is restored before fold-9 calibration. Fold 9 is not used for model fitting or early stopping. Fold 10 is not used until the model and calibrator are frozen.

Local outputs:

- `open_ecg_resnet_checkpoint.pt`
- `open_ecg_resnet_calibration.json`
- `open_ecg_resnet_training_history.json`
- `open_ecg_phase0_resnet_report.json`

The checkpoint binds the frozen state to protocol, model-index, label-manifest, normalization, ResNet-contract, seed, and best-epoch hashes. These artifacts are ignored by Git.

## External certification states

Each locked label x external-domain pair receives exactly one prospective state:

- `certified`: discrimination viable and every calibration envelope check passes;
- `calibration_recovery_candidate`: discrimination remains viable but one or more calibration checks fail;
- `discrimination_failure`: PR-AUC/prevalence ratio is below the prespecified viability threshold;
- `insufficient_support`: fewer than 50 positives or 50 negatives exist in the 60% certification partition.

Only the fixed ResNet report is eligible to drive the primary Phase 0 gate. Logistic results are descriptive/reference evidence.

## Phase 0 cannot touch recovery data

The 40% external recovery pools remain in the verified local index so their identity is frozen before performance inspection, but the Phase 0 execution code never loads their waveforms. Phase 1 may access them only after the primary fixed-ResNet Phase 0 report identifies prospectively eligible calibration-recovery pairs.

## Reporting language

`certified` is a research-protocol endpoint. It is not regulatory approval, bedside safety certification, or evidence that a model is clinically deployable without additional validation.
