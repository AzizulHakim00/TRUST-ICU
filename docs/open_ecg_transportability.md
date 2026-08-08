# TRUST-ECG Open-Data Pivot

## Why this dataset

The restricted MIMIC-IV/eICU path is preserved on `phase-0-feasibility`, but it cannot produce a real study without credentialed access. This branch uses the openly accessible PhysioNet/Computing in Cardiology Challenge 2020 training data instead.

Official resource:

- https://physionet.org/content/challenge-2020/1.0.2/
- Access policy: anyone may access the files under the listed license.
- File license: Creative Commons Attribution 4.0.
- Total uncompressed size reported by PhysioNet: approximately 7.5 GB.

The public training collection contains multiple source datasets rather than one homogeneous cohort. The primary sources used by this protocol are:

| Source | Records | Locked role |
|---|---:|---|
| PTB-XL | 21,837 | development/internal validation |
| Georgia 12-lead ECG | 10,344 | external domain 1 |
| CPSC2018 | 6,877 | external domain 2 |
| CPSC2018 Extra | 3,453 | external domain 3 |

PTB (516) and INCART (74) are too small for the primary source-level transportability analysis and are retained only for descriptive/schema checks.

The hidden Challenge test set is not used because it is not publicly released.

## Research objective

The study is not an architecture competition. The primary question is whether a frozen ECG classifier developed on PTB-XL transports to geographically and operationally different ECG datasets while preserving both discrimination and calibration.

If transportability fails, the conditional second-stage question is how much labeled target-domain data is needed for simple recalibration to recover reliable performance.

Working title:

**TRUST-ECG: Transportability-Aware and Label-Efficient Deployment Certification for Multi-Source 12-Lead ECG Classification**

## Prospective design

### Development

Use PTB-XL only for model fitting, model selection, preprocessing statistics, and calibration.

The official PTB-XL patient-wise stratification metadata must be used. The protocol locks folds 1-8 for training, fold 9 for calibration, and fold 10 for the internal test.

### External evaluation

After model and label-set lock:

1. Georgia
2. CPSC2018
3. CPSC2018 Extra

are evaluated without external tuning.

External labels, performance, prevalence, or calibration may not be used to select models, features, normalization constants, labels, or thresholds.

## Label harmonization

Challenge headers provide diagnosis labels as SNOMED CT codes. Before waveform model training:

1. inventory every diagnosis code by source;
2. apply the official Challenge equivalent-class mapping;
3. compute source-specific positive counts;
4. retain only labels meeting the prospective support rule in `schemas/open_ecg_protocol.yaml`;
5. write and hash the final label manifest;
6. freeze it before any waveform performance is inspected.

A label may not be added or removed after external performance is known unless a timestamped sensitivity-analysis amendment is created; such a change cannot replace the primary result.

## Signal contract

Primary input is the 12-lead waveform only. A secondary analysis may add age and sex.

All records are converted to a deterministic 10-second, 500-Hz representation with the standard 12 leads. Training-set statistics only are used for amplitude normalization. External-domain normalization fitting is prohibited.

## Phase 0 models

The first real run is deliberately small:

- logistic regression using deterministic handcrafted ECG summaries;
- one fixed 1D ResNet waveform baseline.

No architecture sweep is allowed in Phase 0. External domains are evaluated once after the development pipeline is frozen.

## Primary evaluation

Report discrimination and calibration, not accuracy alone:

- macro PR-AUC;
- macro ROC-AUC;
- per-label PR-AUC;
- Brier score;
- calibration slope and intercept;
- external/internal PR-AUC ratio;
- worst external-domain performance.

A strong internal AUROC with poor external PR-AUC or calibration is considered a transportability failure, not a reason to search the external datasets for a better model.

## Conditional Phase 1

Only labels/tasks passing the prospective Phase 0 transportability checks may continue.

For failed deployment domains, evaluate fixed target-label budgets:

`0, 50, 100, 250, 500, 1000`

Allowed updates are intercept-only and Platt recalibration. Full target-domain retraining, external feature selection, and post-hoc threshold search are prohibited in the primary study.

## Download

The complete public Challenge data can be retrieved from the official PhysioNet file endpoint:

```bash
wget -r -N -c -np https://physionet.org/files/challenge-2020/1.0.2/
```

PhysioNet also offers a ZIP download from the resource page. Keep the waveform files outside the Git repository.

For a smaller first download, start with PTB-XL and Georgia, validate the parser and label inventory, then add the two CPSC sources before any primary model result is finalized.

## Immediate implementation order

1. validate this prospective protocol;
2. build a header-only dataset inventory tool;
3. generate the frozen common-label manifest without loading waveform matrices;
4. implement waveform normalization and deterministic 10-second conversion;
5. add PTB-XL patient-wise split verification;
6. implement the two fixed Phase 0 baselines;
7. run PTB-XL internal evaluation;
8. freeze model and preprocessing hashes;
9. run Georgia/CPSC external evaluations exactly once;
10. apply the preregistered continue/stop decision.

No clinical or publication performance claims are made until those real-data steps are complete.
