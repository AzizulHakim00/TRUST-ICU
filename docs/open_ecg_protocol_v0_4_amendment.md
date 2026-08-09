# TRUST-ECG protocol v0.4 prospective amendment

## Scope

This amendment changes only the identity of the PTB-XL development release and the corresponding label-source plumbing. It does not change the seven prespecified canonical classes, external domains, external 60/40 partition, primary model, hyperparameters, calibration method, metrics, certification envelope, or Phase-1 label budgets.

The amendment was made **before any Logistic Regression or ResNet waveform performance was inspected**.

## Original plan

The pre-v0.4 implementation treated the Challenge 2020 PTB-XL copy as the development waveform source and attempted to recover original PTB-XL patient-wise fold assignments by linking Challenge-renamed records back to PTB-XL v1.0.1.

Two identity strategies were prospectively tested with real public header metadata:

1. numeric-rank pairing followed by structural/checksum verification;
2. exact ordered 12-lead WFDB checksum-signature identity.

Neither strategy was accepted by the gate.

## Real crosswalk evidence

The real public-data header audit contained all 21,837 Challenge PTB-XL headers and all 21,837 original PTB-XL `records500` headers.

The checksum-signature attempt produced:

```text
Challenge records                         21,837
Original metadata rows                    21,837
Exact checksum-resolved pairs                  3
Checksum-signature unmatched              21,764
Ambiguous checksum-signature records           70
Missing checksum values                         0
Sampling-rate mismatches                        0
Sample-count mismatches                         0
Lead-order mismatches after canonicalization    0
```

The data therefore did not support a safe one-to-one reverse record identity rule. The gate remained fail-closed and no model training was allowed from that linkage.

## v0.4 decision

Protocol v0.4 develops directly from **original PTB-XL v1.0.1**:

- official `ptbxl_database.csv` supplies `ecg_id`, `patient_id`, `strat_fold`, `filename_hr`, and `scp_codes`;
- official `records500` WFDB `.hea/.dat` files supply the 500-Hz development waveforms;
- official patient-wise folds are used directly, so no Challenge-to-PTB record crosswalk is required;
- Challenge-renamed PTB-XL waveforms are explicitly prohibited as model inputs.

Challenge PTB-XL headers remain useful only as aggregate evidence that the frozen development SCP groups correspond to the same seven canonical Challenge labels.

## Label-semantic concordance

The seven canonical labels had already been selected by the prospective support rule using Challenge header counts. Original PTB-XL SCP statements were then compared against those aggregate counts without using model performance.

Five mappings were one-to-one immediately:

```text
RBBB   <- CRBBB
AF     <- AFIB
LBBB   <- CLBBB
IAVB   <- 1AVB
STach  <- STACH
```

Two equivalent unions were resolved using aggregate exact-union diagnostics and the official PTB-XL SCP statement descriptions:

```text
PAC-equivalent <- PAC OR SVARR
NSR            <- SR OR NORM
```

The final count rule is **union of SCP key presence per record**. A mapped key counts as present regardless of the stored likelihood value; no post-hoc likelihood threshold is fitted.

A real PTB-XL v1.0.1 aggregate audit then reproduced all seven corresponding Challenge PTB-XL positive counts exactly. The locked audit is version `0.2.0` with SHA-256:

```text
29813e879b6b53172661449a7543c300cbf7768fdc97cb218cc04b6ff9aa7fa1
```

## Frozen development split

```text
PTB-XL folds 1-7  model fitting + normalization statistics
PTB-XL fold 8     optimization validation / ResNet stopping epoch only
PTB-XL fold 9     per-label Platt calibration only
PTB-XL fold 10    untouched internal test
```

The real metadata audit verified 21,837 rows, 18,885 unique patients, all folds 1-10, and zero patients spanning multiple folds.

## Unchanged external design

External primary sources remain:

```text
Georgia           10,344
CPSC2018           6,877
CPSC2018 Extra     3,453
```

Their deterministic label-blind split remains 60% Phase-0 certification and 40% untouched recovery pool. External data remain prohibited from model selection, architecture search, normalization fitting, label selection, stopping-epoch selection, and Phase-0 recalibration.

## Unchanged models and endpoints

The low-capacity Logistic Regression reference remains fixed. The predeclared primary model remains the fixed ResNet1D. Architecture search remains prohibited.

The prospective label-domain research envelope remains unchanged:

```text
PR-AUC / prevalence >= 2.0
absolute calibration-slope deviation <= 0.35
absolute calibration intercept <= 0.75
positive Brier skill versus prevalence
```

The possible label-domain states remain:

```text
certified
calibration_recovery_candidate
discrimination_failure
insufficient_support
```

These states are research endpoints and are not clinical, regulatory, or bedside deployment approval.

## Reproducibility consequence

All active v0.4 model-execution code must bind to:

1. original PTB-XL v1.0.1 metadata and `records500` waveforms for development;
2. Challenge Georgia/CPSC/CPSC-Extra waveforms for external evaluation;
3. the real PTB label-concordance audit hash above;
4. a two-source seven-label manifest whose `challenge_ptbxl_model_input` field is `false`;
5. source-specific corpus hashes and waveform-format declarations.

The old rank/checksum crosswalk utilities may remain only as historical diagnostics. Re-enabling a crosswalk requirement or using Challenge PTB-XL as model input is treated as protocol drift and fails validation.
