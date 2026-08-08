# TRUST-ECG Data Preparation Gate

This stage deliberately happens before waveform download, feature engineering, or model training.

## Rationale

The PhysioNet/Computing in Cardiology Challenge 2020 training collection is multi-source and openly accessible, but the sources differ in prevalence, duration, sampling rate, labeling practice, and demographic composition. A transportability paper is only defensible if the common target label set is fixed before external model performance is inspected.

The official Challenge resource reports these primary source totals:

| Source | Records | Role |
|---|---:|---|
| PTB-XL | 21,837 | development/internal validation |
| Georgia | 10,344 | external domain 1 |
| CPSC2018 | 6,877 | external domain 2 |
| CPSC2018 Extra | 3,453 | external domain 3 |

Official resource: `https://physionet.org/content/challenge-2020/1.0.2/`

PTB-XL metadata are taken from the official PTB-XL release. Its documented `strat_fold` assignment is patient-wise, with folds 1-8 proposed for training, fold 9 for validation, and fold 10 for test. The local audit independently verifies that no patient appears in multiple folds.

Official PTB-XL resource: `https://physionet.org/content/ptb-xl/1.0.1/`

## Header-first rule

Do not download the full waveform collection first. Start with the Challenge `.hea` files and `ptbxl_database.csv`.

The headers are sufficient to audit:

- record count by source;
- number of leads and lead order;
- sampling rate;
- recording duration;
- diagnosis SNOMED CT codes;
- label prevalence by source.

The waveform stage remains blocked until the header audit passes.

## Official scored labels

`schemas/challenge2020_scored_classes.csv` pins the scored diagnosis table published by the Challenge evaluation repository. Equivalent classes are collapsed before prevalence counting:

- `713427006` and `59118001` -> canonical RBBB group;
- `284470004` and `63593006` -> canonical PAC/SVPB group;
- `427172004` and `17338001` -> canonical PVC/VPB group.

A record with two equivalent member codes still contributes only one positive to the canonical class.

Official evaluation repository: `https://github.com/physionetchallenges/evaluation-2020`

## Prospective support rule

A canonical label enters the primary waveform task only when all of the following are true:

1. at least 500 positive PTB-XL records;
2. at least 100 positives in an external domain;
3. condition 2 holds in at least two of Georgia, CPSC2018, and CPSC2018 Extra.

These thresholds were frozen in `schemas/open_ecg_protocol.yaml` before any model result existed.

From the official published class counts, the expected candidate set is:

- IAVB (`270492004`);
- AF (`164889003`);
- LBBB (`164909002`);
- PAC/SVPB canonical group (`284470004`);
- RBBB/CRBBB canonical group (`59118001`);
- NSR (`426783006`);
- STach (`427084000`).

This list is an expectation, not the final manifest. The final eligible set must be re-derived from the locally downloaded headers by code. If it differs, training stops and the discrepancy is investigated before any model is fit.

## Download only headers first

Example for one source:

```bash
wget -r -N -c -np -A '*.hea' \
  https://physionet.org/files/challenge-2020/1.0.2/training/ptb-xl/
```

Repeat for:

- `georgia/`
- `cpsc_2018/`
- `cpsc_2018_extra/`

Download official PTB-XL metadata separately:

```bash
wget https://physionet.org/files/ptb-xl/1.0.1/ptbxl_database.csv
```

## Preparation command

The dry run contains no network or patient-level operation:

```bash
python scripts/prepare_open_ecg_data.py --dry-run
```

After the four header trees and PTB-XL metadata are local:

```bash
python scripts/prepare_open_ecg_data.py \
  --header-root /secure/challenge-2020/training \
  --ptbxl-metadata /secure/ptb-xl/ptbxl_database.csv \
  --output /secure/trust_ecg/open_ecg_header_audit.json \
  --require-ready
```

## Hard blockers

Waveform processing is prohibited when any of these occur:

- a primary source record count differs from the official release total;
- a header is not the expected standard 12-lead ordering;
- a header lacks diagnosis codes;
- PTB-XL patient-wise folds cannot be verified;
- no label meets the prespecified cross-source support thresholds.

The audit writes only aggregate counts and a SHA-256 manifest. It does not write record identifiers or row-level labels.

## Next gate

Only after `ready_for_waveform_stage=true` may the project implement and run deterministic waveform standardization. That next stage will verify physical units and sampling frequencies, resample to the locked 500 Hz target when needed, construct the exact 10-second window with pad/crop masks, and calculate normalization statistics from PTB-XL training folds only.
