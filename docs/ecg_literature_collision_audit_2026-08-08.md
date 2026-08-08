# TRUST-ECG Literature Collision Audit — 2026-08-08

## Purpose

This audit is prospective. It is written before TRUST-ECG waveform-model performance is inspected. Its purpose is to prevent a technically correct study from making novelty claims that are already occupied by recent ECG literature.

The literature should be refreshed again immediately before manuscript submission.

## Collision 1 — Cross-dataset ECG domain generalization already exists

Li et al., *Scientific Reports* 2026, “Research on cross-dataset cardiac signal domain generalization and feature interpretability,” explicitly studies ECG domain generalization across multiple arrhythmia datasets and proposes a Transformer-based domain-invariant framework.

DOI: `10.1038/s41598-025-33057-9`

Implication for TRUST-ECG:

- generic “cross-dataset ECG generalization” is not a sufficient novelty claim;
- a new domain-invariant encoder, contrastive alignment module, or domain-adaptation block would enter a crowded methodological space;
- TRUST-ECG must not claim to be the first cross-dataset ECG framework.

## Collision 2 — Large external validation of ECG-AI already exists

Lee et al., *npj Digital Medicine* 2026, externally validated an ECG-AI model for ten emergency/cardiac conditions across a large-scale multi-center U.S. healthcare system.

DOI: `10.1038/s41746-026-02682-7`

The paper itself identifies prospective calibration, clinical integration, and impact evaluation as subsequent needs.

Implication for TRUST-ECG:

- “external validation” by itself is not novel;
- reporting only AUROC/AUPRC on Georgia/CPSC after PTB-XL training would be insufficient as a strong contribution;
- calibration and explicit transportability failure should remain central.

## Collision 3 — ECG recalibration already exists

Lee et al., *JMIR Medical Informatics* 2026, developed a longitudinal patient-wise recalibration strategy for ECG-based LV systolic dysfunction prediction and evaluated it internally and externally.

DOI: `10.2196/83127`

Implication for TRUST-ECG:

- recalibration itself is not novel;
- TRUST-ECG should not claim the first calibrated or recalibrated ECG model;
- the useful question is how much target-domain labeled evidence is required to recover a prespecified deployment-certification criterion after transport failure.

## Collision 4 — Few-shot ECG learning is established

Pałczyński et al., *Sensors* 2022, studied few-shot ECG classification on PTB-XL.

DOI: `10.3390/s22030904`

Zeng et al., *Annals of Noninvasive Electrocardiology* 2026, studied extreme few-shot ECG classification with self-supervised contrastive pretraining on PTB-XL.

DOI: `10.1111/anec.70188`

Implication for TRUST-ECG:

- “few-shot ECG” or “label-efficient ECG learning” cannot be claimed as new;
- the fixed label budgets `0, 50, 100, 250, 500, 1000` are not an architecture contribution;
- they are used to estimate a deployment recovery curve for a frozen source-trained classifier.

## Collision 5 — Zero-shot external ECG performance is also occupied

Obermeyer et al., *Nature* 2026, studied an ECG biomarker for sudden cardiac death and reported lockbox and external validation without target fine-tuning to measure zero-shot performance in new datasets.

DOI: `10.1038/s41586-026-10674-6`

Implication for TRUST-ECG:

- “zero-shot transfer to another ECG dataset” is not sufficient novelty;
- TRUST-ECG must quantify when zero-shot transport is unacceptable and what evidence is required before deployment can be certified.

## Collision 6 — Local ECG fine-tuning platforms exist

ExChanGeAI, *Journal of Medical Internet Research* 2026, provides an end-to-end ECG analysis and model fine-tuning platform and evaluates heterogeneous external datasets.

Article: `https://www.jmir.org/2026/1/e81116`

Implication for TRUST-ECG:

- a platform for local fine-tuning is not our target contribution;
- primary TRUST-ECG Phase 1 prohibits target-domain model retraining and feature selection;
- only simple probability-level localization is allowed in the primary recovery analysis.

## Candidate gap retained by TRUST-ECG

After these collisions, the defensible candidate contribution is narrower:

> For a frozen multi-label 12-lead ECG classifier developed on one source, prospectively determine whether each independent source satisfies a prespecified discrimination-and-calibration deployment envelope, characterize the failure pattern without target-domain tuning, and estimate the labeled target-domain sample budget required for simple recalibration to restore certification.

The contribution is therefore a **deployment-certification study design and evidence framework**, not a new ECG backbone.

Key characteristics that must remain coupled for the candidate contribution to be meaningful:

1. one development source with all modelling choices frozen before external evaluation;
2. multiple independent external source datasets;
3. a label set frozen before waveform performance is inspected;
4. both discrimination and calibration in the certification envelope;
5. worst-domain reporting rather than best-domain reporting;
6. no external feature selection, architecture tuning, or target-domain model retraining in the primary study;
7. prespecified labeled target budgets for probability-level recovery;
8. reporting the smallest budget that restores certification, including failure to recover;
9. independent replication on later data if feasible.

## Claims currently prohibited

The manuscript must not state any of the following without a later targeted search proving the claim:

- first ECG classifier;
- first deep-learning ECG classifier;
- first multi-label ECG classifier;
- first cross-dataset ECG model;
- first ECG domain-generalization framework;
- first external ECG validation;
- first calibrated ECG-AI model;
- first ECG recalibration method;
- first few-shot ECG framework;
- first zero-shot ECG transfer study;
- first target-domain ECG adaptation framework.

## Candidate claim language

Until the final pre-submission search, use wording such as:

> We evaluate a prospectively specified deployment-certification framework for cross-source ECG transportability.

Do not use “first,” “novel,” or “unprecedented” as factual claims in the abstract or contribution list.

## Research consequence

Phase 0 should remain deliberately simple: a low-capacity linear reference and one fixed 1D ResNet. If these models reveal source-specific calibration or discrimination failure, that failure is scientifically informative. Searching external datasets for a more favorable architecture would invalidate the primary transportability experiment.
