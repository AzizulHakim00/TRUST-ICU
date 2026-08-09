# TRUST-ECG open-data transportability study

## Current design

TRUST-ECG is an open-data study of cross-source ECG transportability, calibration failure, and label-efficient probability recovery. The study is not an architecture competition and does not claim that its fixed ResNet is novel.

Protocol v0.4 uses two public resources with different roles:

1. **Original PTB-XL v1.0.1** is the sole development and internal-validation source because it provides the official patient identifiers, stratified folds, SCP-ECG statements, and original 500-Hz WFDB records.
2. **PhysioNet/Computing in Cardiology Challenge 2020 v1.0.2** provides Georgia, CPSC2018 and CPSC2018-Extra as external domains and provides Challenge PTB-XL headers only as aggregate label-concordance evidence. Challenge-renamed PTB-XL waveforms are not model inputs.

Working title:

**TRUST-ECG: Transportability-Aware and Label-Efficient Deployment Certification for Multi-Source 12-Lead ECG Classification**

The word *certification* denotes a prospective research endpoint under this protocol. It is not regulatory approval, a bedside safety guarantee, or a claim of clinical deployment readiness.

## Primary sources

| Source | Records | Locked role |
|---|---:|---|
| Original PTB-XL v1.0.1 | 21,837 | development and internal validation |
| Georgia 12-lead ECG | 10,344 | external domain 1 |
| CPSC2018 | 6,877 | external domain 2 |
| CPSC2018 Extra | 3,453 | external domain 3 |

The primary modeling set therefore contains 42,511 records. PTB and INCART remain non-primary descriptive/schema sources. The hidden Challenge test set is not used because it is not publicly released.

## Why protocol v0.4 uses original PTB-XL directly

The initial implementation attempted to reuse Challenge-renamed PTB-XL records for development while recovering the official PTB-XL patient/fold assignment. Real public-data audits showed that this could not be done safely:

- numeric-rank pairing was not a valid identity rule;
- all-12-lead WFDB checksum signatures were not preserved as a usable one-to-one identity across the Challenge conversion;
- only 3 of 21,837 records were exactly resolved by the checksum-signature attempt, with most signatures unmatched and some ambiguous.

Those failures occurred before any waveform model performance was inspected. The protocol was therefore prospectively amended to develop directly from original PTB-XL v1.0.1. Reverse Challenge/PTB-XL crosswalks are now historical diagnostics only and are prohibited from modeling.

## Research question

The primary question is whether a frozen 12-lead ECG classifier developed on original PTB-XL retains useful discrimination and calibration across independent Georgia and CPSC source datasets.

If a label-domain pair retains discrimination but fails the prespecified calibration envelope, the conditional second-stage question is how much labeled target-domain data is required for simple recalibration to recover the research deployment envelope.

## Development split

The official PTB-XL patient-wise stratified folds are immutable:

```text
folds 1-7  model fitting and normalization statistics
fold 8     optimization validation / ResNet stopping epoch only
fold 9     per-label Platt calibration only
fold 10    untouched internal test
```

A patient may not span folds. Folds 8 or 9 are never folded back into model fitting. External domains cannot affect model selection, preprocessing, normalization, architecture, label selection, or stopping epoch.

## Frozen seven-label task

Challenge diagnosis labels use SNOMED CT groups; original PTB-XL uses SCP-ECG statement keys. The final seven canonical groups were selected by the prospective support rule before waveform model performance:

- at least 500 positive PTB-XL development records;
- at least 100 positive records in at least two external domains.

The frozen mapping is:

| Canonical class | SNOMED | Original PTB-XL SCP union |
|---|---:|---|
| RBBB | 59118001 | CRBBB |
| AF | 164889003 | AFIB |
| LBBB | 164909002 | CLBBB |
| IAVB | 270492004 | 1AVB |
| PAC-equivalent | 284470004 | PAC OR SVARR |
| NSR | 426783006 | SR OR NORM |
| STach | 427084000 | STACH |

Development labels use **union of SCP key presence per record**. Likelihood values are not thresholded. A real aggregate-only PTB-XL concordance audit reproduced all seven corresponding Challenge PTB-XL positive counts exactly and is pinned by SHA-256 in the protocol.

The locked concordance audit SHA-256 is:

```text
29813e879b6b53172661449a7543c300cbf7768fdc97cb218cc04b6ff9aa7fa1
```

No label may be added or removed after model-performance inspection in the primary study.

## Signal contract

Primary input is waveform only; a secondary analysis may add age and sex only if prospectively activated.

All sources are represented as standard 12-lead physical mV signals, deterministically transformed to 10 seconds at 500 Hz. Original PTB-XL is read from its WFDB `.hea/.dat` records; Georgia/CPSC external records are read from Challenge `.hea/.mat` pairs. Per-lead normalization statistics are fitted only on original PTB-XL folds 1-7.

Primary preprocessing does not introduce source-specific filtering. Longer signals are deterministically center-cropped; shorter signals are symmetrically zero-padded with a validity mask. External normalization refitting is prohibited.

## Fixed Phase-0 models

Phase 0 contains exactly two fixed model roles:

- independent binary Logistic Regression models on a locked 144-feature handcrafted waveform representation, used only as a low-capacity reference;
- one predeclared fixed 1D ResNet used for the primary research certification gate.

Architecture search is disabled. The ResNet may optimize only the stopping epoch using PTB-XL fold 8.

## External certification/recovery split

Each Georgia/CPSC record is assigned without labels by a deterministic seeded SHA-256 rule:

```text
60%  Phase-0 external certification
40%  untouched conditional recovery pool
```

The recovery pool cannot be loaded by the Phase-0 runner. Repartitioning after labels or performance are inspected is prohibited.

## Primary evaluation

The study reports discrimination and calibration rather than accuracy alone:

- macro and per-label PR-AUC;
- macro ROC-AUC;
- PR-AUC/prevalence ratio;
- Brier score and Brier skill versus prevalence;
- calibration slope and intercept;
- external/internal PR-AUC ratio;
- worst external-domain behavior;
- label-domain certification matrix.

Each external label-domain pair becomes exactly one of:

```text
certified
calibration_recovery_candidate
discrimination_failure
insufficient_support
```

A strong internal metric with poor external discrimination or calibration is a transportability result, not permission to search the external domains for another model.

## Conditional recovery experiment

Only `calibration_recovery_candidate` pairs may proceed. Recalibration is not used to rescue ranking failure.

The untouched recovery pool uses fixed target-label budgets:

```text
0, 50, 100, 250, 500, 1000
```

Allowed updates are intercept-only recalibration and Platt recalibration. Target-domain model retraining/fine-tuning, feature selection, normalization refit, architecture selection, and post-hoc threshold search are prohibited in the primary study.

## Reproducibility chain

A valid real result must be traceable through:

```text
protocol v0.4
  -> Challenge aggregate label-support audit
  -> original PTB-XL label-concordance audit
  -> two-source seven-label manifest
  -> waveform corpus audit + training-only normalization
  -> verified source-aware model index
  -> frozen Logistic reference / fixed ResNet state
  -> fold-9 calibration state
  -> internal + external certification report
```

Each machine-readable stage is hash-locked. Record-level indexes, predictions, waveforms, and model outputs remain outside Git. Only aggregate research evidence is intended for public reporting.

## Current evidence boundary

Header/source/fold/label-concordance work is real public-data evidence. Synthetic tests validate software invariants only. No Logistic or ResNet scientific performance result is claimed until the full waveform corpus is downloaded, audited, indexed, and executed under the locked protocol.
