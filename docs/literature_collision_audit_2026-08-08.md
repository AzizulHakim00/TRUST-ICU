# Literature Collision Audit — 2026-08-08

This audit is a prospective novelty check for work that could follow TRUST-ICU Phase 0. It is not a systematic review and it must be refreshed before any manuscript makes a priority or novelty claim.

## Why this audit was added

The original project direction considered joint organ-support forecasting, cross-database transportability, and conformal/selective risk control. By August 2026, several recent publications materially overlap those ideas. Architecture development is therefore prohibited until Phase 0 passes and the Phase 1 question is narrowed to a defensible gap.

## Material collisions

### Joint ICU intervention forecasting is no longer a defensible novelty claim

Aikodon et al., *FIRST-ICU: forecasting interventions and risk stratification in the ICU using graph neural network autoencoders*, npj Digital Medicine, 2026, DOI `10.1038/s41746-026-02890-1`, jointly predicts seven ICU interventions, develops on MIMIC-IV, and externally validates on AmsterdamUMCdb.

**Consequence:** TRUST-ICU must not claim novelty from multi-intervention prediction, intervention-interaction modelling, MIMIC-IV development, or cross-continental external validation alone.

### Generic conformal prediction plus cost-aware deferral is already published

Kwon and Kim, *Conformal selective prediction with cost aware deferral for safe clinical triage under distribution shift*, Scientific Reports, 2026, DOI `10.1038/s41598-026-40637-w`, combines calibrated probabilities, split/Mondrian/weighted conformal prediction, and a calibration-selected cost-aware deferral rule under temporal shift.

**Consequence:** TRUST-ICU must not claim novelty from conformal prediction, selective abstention, cost-aware deferral, Mondrian subgroup calibration, or weighted conformal shift handling alone.

### Hierarchical healthcare conformal calibration is also occupied

Shahbazi et al., *A hierarchical conformal framework for uncertainty-aware length of stay prediction in multi-hospital settings*, Scientific Reports, 2026, DOI `10.1038/s41598-026-37450-w`, addresses hierarchical multi-hospital uncertainty.

A June 2026 preprint, Shahid, *When Calibration Fails the Vulnerable Hospital: Federated Conformal Risk Control via Risk-Curve Shrinkage*, arXiv `2606.20115`, explicitly studies pooled conformal risk control failure at individual hospitals and shrinkage-based site-specific calibration.

**Consequence:** TRUST-ICU must not present generic hierarchical, hospital-conditional, or shrinkage conformal calibration as an unqualified new method.

### Multi-database ICU transfer/generalization comparisons are crowded

Tranchellini et al., *Evaluating deep learning sepsis prediction models in ICUs under distribution shift: a multi-centre retrospective cohort study*, npj Digital Medicine, 2026, DOI `10.1038/s41746-026-02364-4`, compares generalization, retraining, target training, domain adaptation, and fusion across HiRID, MIMIC-IV, and eICU under multiple target-data regimes.

A four-database external validation study in Critical Care, 2026, DOI `10.1186/s13054-026-06034-5`, evaluates pooling and transfer across eICU-CRD, MIMIC-IV, AmsterdamUMCdb, and HiRID.

**Consequence:** merely adding three or four ICU databases, transfer learning, or target-site fine-tuning is not sufficient novelty.

### Intervention-specific prediction is also partially occupied

Recent and prior work includes externally validated intubation prediction on MIMIC-IV/eICU, vasopressor prediction, and RRT/AKI-related prediction. A 2025 BMC Medical Informatics and Decision Making study (DOI `10.1186/s12911-025-03274-3`) predicts vasopressor initiation in MIMIC-IV and explicitly calls for multi-centre external validation.

**Consequence:** each organ-support endpoint remains clinically useful, but the manuscript cannot be framed as the first model to predict these interventions.

### Transportability estimation itself is not new

Methods already estimate external clinical-model performance from limited target summary statistics, including an npj Digital Medicine benchmark published in 2024 (`10.1038/s41746-024-01414-z`). Other work addresses unlabeled-data drift correction and transport without new target labels.

**Consequence:** a generic "predict external performance from shift statistics" claim is too broad.

## Revised Phase 1 question

The strongest remaining direction is a **pre-deployment certification benchmark for ICU intervention forecasting**, not a new neural architecture.

For each Phase-0 task that passes its locked external feasibility gate:

> Can a frozen ICU intervention model be certified as safe or unsafe at a new hospital using prespecified transportability diagnostics, and how much local outcome-labeled data is required to restore calibration and risk control when certification fails?

The study will focus on hospital-level failure heterogeneity, label-budgeted localization, and operational deployment decisions rather than leaderboard discrimination.

## Candidate contribution set

These are hypotheses to test, not novelty claims:

1. quantify how often an externally reasonable aggregate model still violates prespecified safety/calibration envelopes at individual eICU hospitals;
2. relate site-level failures to prespecified, mostly pre-deployment shift descriptors such as missingness, measurement intensity, covariate shift, case-mix shift, and outcome prevalence once labels are available;
3. compare no-update, intercept-only recalibration, Platt recalibration, and prespecified conformal/selective baselines across fixed local label budgets;
4. report the minimum local label budget needed to recover calibration/safety by task and hospital stratum;
5. replicate major conclusions at database level in later AmsterdamUMCdb and HiRID analyses if data access and label harmonization permit;
6. produce a reproducible deployment-certification protocol with failure-first reporting, rather than proposing another intervention-forecasting architecture.

## Claims explicitly prohibited

Until another literature refresh is completed, the project must not claim:

- first multi-intervention ICU forecasting model;
- first externally validated ICU intervention model;
- first conformal clinical triage framework;
- first cost-aware clinical deferral framework;
- first hierarchical or hospital-conditional conformal method;
- first site-shrinkage conformal calibration method;
- first multi-database ICU transfer-learning study;
- first method to estimate transportability from target summary statistics.

## Activation rule

Phase 1 may be activated only for tasks whose Phase-0 `continue_to_architecture_development` decision is true. A Phase-0 failure remains a publishable negative transportability result and must not be rescued by changing the endpoint, model, feature set, or threshold against the same external cohort.

## Refresh rule

Before Phase 1 analysis begins and again before manuscript submission:

1. rerun the literature search;
2. add newly found collisions;
3. downgrade or remove any threatened novelty wording;
4. preserve dated versions of this audit.
