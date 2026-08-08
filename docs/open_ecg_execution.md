# TRUST-ECG execution sequence

This document is the execution order for the open-data TRUST-ECG study. The sequence is intentionally fail-closed: a later stage must not run when an earlier integrity gate fails.

## 1. Install

Core research utilities:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Install the fixed PyTorch baseline only on a machine that will execute the waveform model:

```bash
pip install -e ".[dev,ecg-deep]"
```

PyTorch is optional so routine protocol validation and audit CI remain lightweight.

## 2. Header-only feasibility

Before downloading or processing every waveform, collect the official Challenge 2020 headers plus PTB-XL v1.0.1 metadata and original 500 Hz headers.

```bash
python scripts/prepare_open_ecg_data.py --dry-run
```

Then run the real header audit with the locally downloaded public files. The audit must verify exact source counts, the Challenge/PTB-XL checksum crosswalk, PTB-XL patient-wise folds, scored-label harmonization, and the prospective label-support rules.

Expected aggregate artifact:

```text
open_ecg_header_audit.json
```

Do not start waveform training if `ready_for_waveform_stage` is false.

## 3. Lock the common diagnosis set

```bash
python scripts/lock_open_ecg_labels.py --dry-run
```

The real run creates:

```text
open_ecg_label_manifest.json
```

The label manifest is SHA-256 protected. No diagnosis class may be added or removed after model performance is inspected.

## 4. Waveform audit and normalization

```bash
python scripts/prepare_open_ecg_waveforms.py --dry-run
```

The real waveform stage verifies every primary `.hea`/`.mat` pair, hashes each source corpus, converts to physical mV, resamples to 500 Hz where required, applies the deterministic 10-second crop/pad rule, and fits per-lead normalization statistics on PTB-XL folds 1-7 only.

Local outputs include:

```text
open_ecg_waveform_audit.json
open_ecg_normalization_stats.json
ptbxl_verified_assignment.csv
```

The assignment file contains no patient identifier, but it is still record-level derived data and must remain local.

## 5. Build the verified model index

```bash
python scripts/build_open_ecg_model_index.py --dry-run
```

Real execution:

```bash
python scripts/build_open_ecg_model_index.py \
  --challenge-training-root /secure/challenge-2020/training \
  --waveform-audit /secure/trust_ecg/open_ecg_waveform_audit.json \
  --label-manifest /secure/trust_ecg/open_ecg_label_manifest.json \
  --ptbxl-assignment /secure/trust_ecg/ptbxl_verified_assignment.csv \
  --output-root /secure/trust_ecg/model_index
```

The model-index stage re-hashes the complete primary header and waveform corpora. This intentionally costs an additional sequential read of the public data: it proves that the files used for modeling are byte-identical to the files that passed the waveform audit.

The local record index assigns only these roles:

```text
PTB-XL folds 1-7  -> model_fit
PTB-XL fold 8     -> optimization_validation
PTB-XL fold 9     -> calibration
PTB-XL fold 10    -> internal_test
external sources  -> label-blind SHA256 60% certification / 40% recovery pool
```

Outputs:

```text
open_ecg_model_index.csv          # local record-level artifact; never commit
open_ecg_model_index_audit.json   # aggregate integrity report
```

Training is allowed only when `ready_for_baseline_execution` is true.

## 6. Fixed Phase-0 models

The primary model is the prospectively specified ResNet1D implementation in `src/trust_icu/ecg_resnet.py`. Its architecture is not a novelty claim and architecture search is prohibited.

The low-capacity reference is independent one-vs-rest Logistic Regression using the locked 144 handcrafted waveform summaries.

The fixed ResNet training contract is:

```text
input              12 x 5000
optimizer          AdamW
learning rate      1e-3
weight decay       1e-4
batch size         64
max epochs         50
early stopping     fold-8 macro PR-AUC
patience           7 epochs
minimum delta      1e-4
gradient clip      5.0
augmentation       none in primary analysis
random seed        20260808
```

Only the best epoch may be selected from fold 8. Fold 9 is calibration-only. Fold 10 is untouched internal testing. Georgia, CPSC2018, and CPSC2018-Extra must not influence architecture, hyperparameters, normalization, label selection, or stopping epoch.

## 7. Calibration and external certification

Per-label Platt scaling is fitted only on PTB-XL fold 9. The frozen model plus frozen calibrators are then evaluated on PTB-XL fold 10 and the 60% certification partition of every external source.

Each label-domain pair receives exactly one prospective state:

```text
certified
calibration_recovery_candidate
discrimination_failure
insufficient_support
```

External Phase-0 results must not be used to change the model.

## 8. Conditional recovery study

Only label-domain pairs classified as `calibration_recovery_candidate` may enter the recovery experiment. Ranking failures are not rescued with probability recalibration.

The untouched 40% external recovery pool is used with fixed label budgets:

```text
0, 50, 100, 250, 500, 1000
```

Allowed updates are intercept-only recalibration and Platt recalibration. Target-domain model retraining, feature selection, normalization refit, and post-hoc threshold search remain prohibited.

## Reproducibility rule

No reported ECG metric is valid unless its provenance includes the hashes of the protocol, header audit, label manifest, waveform audit, model-index audit, model contract, and result report. Synthetic tests validate software behavior only; they are never scientific performance evidence.
