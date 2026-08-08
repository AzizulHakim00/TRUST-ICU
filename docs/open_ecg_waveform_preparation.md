# TRUST-ECG Waveform Preparation Gate

This stage starts only after the header feasibility audit passes and the common diagnosis label set has been SHA-256 locked.

## Why this stage is separate from model training

Cross-source ECG studies are vulnerable to silent preprocessing leakage. Source-specific filters, target-domain normalization, inconsistent lead ordering, and undocumented crop rules can make an apparent generalization result impossible to interpret.

TRUST-ECG therefore generates a dataset-level waveform audit before fitting any classifier.

## Required inputs

- complete Challenge 2020 v1.0.2 primary training sources:
  - PTB-XL;
  - Georgia;
  - CPSC2018;
  - CPSC2018 Extra;
- PTB-XL v1.0.1 `ptbxl_database.csv`;
- PTB-XL v1.0.1 original `records500` headers;
- verified `open_ecg_header_audit.json`;
- locked `open_ecg_label_manifest.json`.

The PTB-XL version must remain v1.0.1 because this is the version from which the Challenge 2020 PTB-XL source was derived.

## Verified PTB-XL record assignment

Challenge filenames are not assumed to encode official PTB-XL `ecg_id` values.

The pipeline:

1. numerically sorts all Challenge PTB-XL record IDs;
2. sorts official PTB-XL metadata by `ecg_id`;
3. pairs them by rank;
4. opens each corresponding original `filename_hr` header;
5. requires equal sampling rate, sample count, lead order, and all 12 WFDB checksums for every pair.

Only after all pairs verify is the local `challenge_record_id -> ecg_id -> strat_fold` assignment written.

The assignment intentionally omits patient identifiers. Patient-wise fold integrity is checked separately from the official metadata.

## Locked development roles

| PTB-XL fold(s) | Purpose |
|---|---|
| 1-7 | model fitting and training-only normalization statistics |
| 8 | ResNet stopping-epoch selection only |
| 9 | probability calibration only |
| 10 | untouched internal test |

Georgia, CPSC2018, and CPSC2018 Extra are external domains and never contribute preprocessing statistics.

## Primary waveform path

For every record:

1. parse WFDB physical gain/baseline metadata;
2. convert digital values to millivolts;
3. reorder to the locked standard 12-lead sequence;
4. resample to 500 Hz if required using deterministic polyphase resampling;
5. center-crop recordings longer than 10 seconds;
6. symmetrically zero-pad recordings shorter than 10 seconds and retain a validity mask;
7. do not apply source-specific or target-fitted denoising/filtering.

The primary analysis deliberately contains no additional band-pass, notch, baseline-removal, or denoising operation. Such a representation may only be added later as a prospectively documented sensitivity analysis.

## Normalization

Per-lead mean and standard deviation are estimated in a streaming manner using **only PTB-XL folds 1-7**.

Padding is excluded from the estimates. The resulting normalization statistics are SHA-256 hashed and frozen. External domains use these same source-fitted statistics unchanged.

## Dataset integrity hashes

Because the data are open but the results must remain reproducible, the waveform stage computes one deterministic corpus SHA-256 per source for:

- all raw `.hea` files;
- all raw `.mat` files.

The corpus hash combines each relative path with that file's SHA-256. A one-byte mutation to a waveform therefore changes the source corpus signature.

## Outputs

Keep these files outside Git:

```text
open_ecg_waveform_outputs/
├── ptbxl_verified_assignment.csv
├── open_ecg_normalization_stats.json
└── open_ecg_waveform_audit.json
```

`ptbxl_verified_assignment.csv` is record-level derived data and is ignored by Git. The JSON outputs are also ignored until disclosure/publication review.

## Dry run

```bash
python scripts/prepare_open_ecg_waveforms.py --dry-run
```

## Real local run

```bash
python scripts/prepare_open_ecg_waveforms.py \
  --challenge-training-root /secure/challenge-2020/training \
  --ptbxl-metadata /secure/ptb-xl-1.0.1/ptbxl_database.csv \
  --ptbxl-original-root /secure/ptb-xl-1.0.1 \
  --header-audit /secure/trust_ecg/open_ecg_header_audit.json \
  --label-manifest /secure/trust_ecg/open_ecg_label_manifest.json \
  --output-root /secure/trust_ecg/waveform_stage \
  --require-ready
```

## Hard stop conditions

Model training remains blocked when any primary waveform is missing or invalid, the Challenge/PTB-XL crosswalk fails, the record-to-fold assignment cannot be reproduced, the label manifest does not match the header audit, normalization cannot be estimated exclusively from folds 1-7, or an input integrity check fails.

No record is silently dropped from the primary study to make the pipeline run.
