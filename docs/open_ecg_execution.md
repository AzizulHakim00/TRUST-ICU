# TRUST-ECG v0.4 execution sequence

This is the fail-closed execution order for the open-data TRUST-ECG study after the prospective v0.4 amendment. Original PTB-XL v1.0.1 is the sole development/internal-validation source. Challenge 2020 Georgia, CPSC2018 and CPSC2018-Extra are external domains. Challenge-renamed PTB-XL is never a model input.

## 1. Install

Core validation and header/label audit utilities:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Waveform execution additionally requires the official WFDB reader:

```bash
pip install -e ".[dev,ecg-waveform]"
```

The fixed ResNet runtime is optional and should only be installed on the compute machine that will execute the primary waveform baseline:

```bash
pip install -e ".[dev,ecg-deep]"
```

## 2. Data layout

Use one local primary-data root with this meaning:

```text
/secure/trust_ecg_data/
├── ptb-xl/                  # original PTB-XL v1.0.1
│   └── records500/          # official .hea/.dat records
├── georgia/                 # Challenge 2020 .hea/.mat
├── cpsc_2018/               # Challenge 2020 .hea/.mat
└── cpsc_2018_extra/         # Challenge 2020 .hea/.mat
```

Keep `ptbxl_database.csv` from PTB-XL v1.0.1 beside the secure study outputs or another stable local location.

Do **not** put the Challenge-renamed PTB-XL waveform directory into this modeling root. Under protocol v0.4 it is used only as aggregate pre-model label evidence and never enters training, calibration, internal testing, external certification, or recovery.

## 3. Challenge header label-support audit

Before waveform modeling, inspect the public Challenge headers only:

```bash
python scripts/prepare_open_ecg_data.py --dry-run
```

Real execution uses the four Challenge header sources plus official PTB-XL metadata for fold-integrity evidence:

```bash
python scripts/prepare_open_ecg_data.py \
  --header-root /secure/challenge-2020-header-only \
  --ptbxl-metadata /secure/ptb-xl/ptbxl_database.csv \
  --output /secure/trust_ecg/open_ecg_header_audit.json \
  --require-ready
```

This audit checks exact source counts, 12-lead/header integrity, PTB-XL patient-wise folds, Challenge scored-label equivalence groups, and the predeclared label-support thresholds. A reverse Challenge/PTB-XL record crosswalk is intentionally **not** required.

## 4. Original PTB-XL label-concordance audit

The development labels come directly from original PTB-XL `scp_codes`. The frozen canonical mapping is:

```text
RBBB   59118001   <- CRBBB
AF     164889003  <- AFIB
LBBB   164909002  <- CLBBB
IAVB   270492004  <- 1AVB
PAC    284470004  <- PAC OR SVARR
NSR    426783006  <- SR OR NORM
STach  427084000  <- STACH
```

Presence of any mapped SCP key in a record makes that canonical class positive; likelihood values are not thresholded. This rule was locked from aggregate evidence before waveform performance was inspected.

Run:

```bash
python scripts/audit_original_ptbxl_labels.py \
  --ptbxl-metadata /secure/ptb-xl/ptbxl_database.csv \
  --output /secure/trust_ecg/ptbxl_label_concordance_audit.json \
  --require-ready
```

All seven unions must exactly reproduce their corresponding Challenge PTB-XL aggregate positive counts. The real v0.2.0 concordance evidence currently pinned by protocol v0.4 has SHA-256:

```text
29813e879b6b53172661449a7543c300cbf7768fdc97cb218cc04b6ff9aa7fa1
```

## 5. Lock the two-source seven-label manifest

A label manifest is valid only when both independent evidence sources agree:

1. Challenge header audit confirms the seven classes satisfy the prospective development/external support rules.
2. Original PTB-XL concordance confirms the frozen SCP unions exactly match the corresponding Challenge aggregate counts.

Dry-run:

```bash
python scripts/lock_open_ecg_labels.py --dry-run
```

Real execution:

```bash
python scripts/lock_open_ecg_labels.py \
  --header-audit /secure/trust_ecg/open_ecg_header_audit.json \
  --ptbxl-label-concordance /secure/trust_ecg/ptbxl_label_concordance_audit.json \
  --output /secure/trust_ecg/open_ecg_label_manifest.json
```

The manifest is SHA-256 protected and explicitly records `challenge_ptbxl_model_input: false`. No diagnosis class may be added or removed after waveform/model performance is inspected.

## 6. Waveform audit and normalization

Dry-run:

```bash
python scripts/prepare_open_ecg_waveforms.py --dry-run
```

Real execution:

```bash
python scripts/prepare_open_ecg_waveforms.py \
  --primary-data-root /secure/trust_ecg_data \
  --ptbxl-metadata /secure/ptb-xl/ptbxl_database.csv \
  --header-audit /secure/trust_ecg/open_ecg_header_audit.json \
  --label-manifest /secure/trust_ecg/open_ecg_label_manifest.json \
  --output-root /secure/trust_ecg/waveform_audit \
  --require-ready
```

The waveform stage:

- reads original PTB-XL `records500` `.hea/.dat` through the official WFDB reader as physical mV;
- reads Georgia/CPSC/CPSC-Extra Challenge `.hea/.mat` and converts their digital samples to physical mV;
- canonicalizes only the documented augmented-lead capitalization difference (`AVR/AVL/AVF` to `aVR/aVL/aVF`);
- resamples to 500 Hz only where required;
- applies the deterministic 10-second center-crop or symmetric zero-pad rule;
- fits per-lead normalization statistics on original PTB-XL folds 1-7 only;
- hashes each source header and waveform corpus.

Local outputs:

```text
open_ecg_waveform_audit.json
open_ecg_normalization_stats.json
ptbxl_verified_assignment.csv
```

The assignment contains record/fold linkage but no patient identifier. It remains local and must not be committed.

## 7. Build the verified model index

Dry-run:

```bash
python scripts/build_open_ecg_model_index.py --dry-run
```

Real execution:

```bash
python scripts/build_open_ecg_model_index.py \
  --primary-data-root /secure/trust_ecg_data \
  --ptbxl-metadata /secure/ptb-xl/ptbxl_database.csv \
  --waveform-audit /secure/trust_ecg/waveform_audit/open_ecg_waveform_audit.json \
  --label-manifest /secure/trust_ecg/open_ecg_label_manifest.json \
  --ptbxl-assignment /secure/trust_ecg/waveform_audit/ptbxl_verified_assignment.csv \
  --output-root /secure/trust_ecg/model_index
```

The index re-hashes the exact corpora and creates canonical seven-label vectors using source-specific evidence:

```text
original PTB-XL       -> frozen SCP-key unions
Georgia/CPSC external -> frozen Challenge SNOMED groups
```

Statistical roles are fixed:

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

## 8. Fixed Phase-0 models

The low-capacity reference is independent one-vs-rest Logistic Regression using the locked 144 handcrafted waveform summaries. It is not allowed to determine the primary GO/NO-GO decision.

```bash
python scripts/run_open_ecg_phase0.py \
  --model logistic \
  --primary-data-root /secure/trust_ecg_data \
  --model-index /secure/trust_ecg/model_index/open_ecg_model_index.csv \
  --model-index-audit /secure/trust_ecg/model_index/open_ecg_model_index_audit.json \
  --label-manifest /secure/trust_ecg/open_ecg_label_manifest.json \
  --output-root /secure/trust_ecg/phase0
```

The primary model is the prospectively fixed ResNet1D. Architecture search is prohibited. Its locked training contract is:

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

Real primary execution:

```bash
python scripts/run_open_ecg_phase0.py \
  --model resnet \
  --primary-data-root /secure/trust_ecg_data \
  --model-index /secure/trust_ecg/model_index/open_ecg_model_index.csv \
  --model-index-audit /secure/trust_ecg/model_index/open_ecg_model_index_audit.json \
  --label-manifest /secure/trust_ecg/open_ecg_label_manifest.json \
  --waveform-audit /secure/trust_ecg/waveform_audit/open_ecg_waveform_audit.json \
  --normalization-stats /secure/trust_ecg/waveform_audit/open_ecg_normalization_stats.json \
  --output-root /secure/trust_ecg/phase0 \
  --device cuda
```

Only the best epoch may be selected from fold 8. Fold 9 is calibration-only. Fold 10 is untouched internal testing. Georgia, CPSC2018, and CPSC2018-Extra cannot influence architecture, hyperparameters, normalization, label selection, or stopping epoch.

## 9. Calibration and external certification

Per-label Platt scaling is fitted only on PTB-XL fold 9. The frozen model plus frozen calibrators are evaluated on PTB-XL fold 10 and the 60% certification partition of each external source.

Each label-domain pair receives exactly one prospective research state:

```text
certified
calibration_recovery_candidate
discrimination_failure
insufficient_support
```

These are protocol research states, not regulatory or bedside safety approval.

## 10. Conditional recovery study

Only `calibration_recovery_candidate` pairs may enter the recovery experiment. Probability recalibration is not used to rescue ranking failure.

The untouched 40% external recovery pool uses fixed label budgets:

```text
0, 50, 100, 250, 500, 1000
```

Allowed local updates are intercept-only recalibration and Platt recalibration. Target-domain model retraining, feature selection, normalization refit, and post-hoc threshold search remain prohibited.

## Historical reverse-crosswalk note

The repository contains diagnostic code from the pre-v0.4 attempt to establish Challenge-renamed PTB-XL record identity. Real public-data audits showed that neither numeric-rank pairing nor all-12-lead WFDB checksum identity can safely reconstruct the official PTB-XL patient/fold mapping. Those methods are retained only as historical diagnostics and must not be re-enabled for modeling.

## Reproducibility rule

No reported ECG metric is valid unless its provenance includes hashes for the protocol, Challenge label-support audit, original PTB-XL concordance audit, two-source label manifest, waveform audit, model-index audit, model contract/checkpoint or Logistic reference state, calibration state, and final result report. Synthetic tests validate software behavior only; they are never scientific performance evidence.
