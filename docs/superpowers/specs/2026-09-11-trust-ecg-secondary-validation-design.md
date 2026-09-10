# TRUST-ECG Secondary Validation Design

Date: 2026-09-11
Branch: `open-ecg-transportability`
Status: approved design

## Purpose

Strengthen the publication evidence around TRUST-ECG without modifying, replacing, or retrospectively tuning the frozen primary TRUST-ECG v0.4 experiment.

The primary scientific result remains the prospectively locked execution using:

- fixed 1D ResNet;
- primary seed `20260808`;
- PTB-XL folds 1-7 for fitting, fold 8 for early stopping, fold 9 for global calibration, fold 10 for untouched internal testing;
- label-blind 60/40 external certification/recovery partition;
- fixed Phase-0 certification envelope;
- frozen Phase-1 budgets and recalibration methods.

All work described below is secondary, sensitivity, robustness, interpretability, or replication analysis. None of it may alter the primary report, model hash, frozen protocol, label definitions, external partition, or primary conclusions after the fact.

## Scientific goals

The secondary package addresses the main remaining reviewer concerns:

1. dependence of the headline certification pattern on exact envelope thresholds;
2. dependence on neural-network initialization;
3. lack of waveform-level explainability;
4. lack of classical XAI views for the low-capacity reference model;
5. lack of signal robustness/stress testing;
6. lack of framework-level ablation/sensitivity analysis;
7. residual cross-partition duplicate/near-duplicate risk when external patient IDs are unavailable;
8. uncertainty around categorical certification decisions;
9. incomplete Phase-1 estimability reporting;
10. lack of concise compute/deployment profile;
11. limited subgroup evidence where reliable metadata are available.

## Design principles

### Primary analysis remains immutable

No secondary analysis may write into or overwrite canonical Phase-0, Phase-1, model, calibration, protocol, manifest, waveform-audit, or statistical-addendum artifacts. Secondary outputs must live under a separate aggregate-only output root and carry hashes linking them to the exact frozen primary artifacts used.

### Low-compute first

Execution order is designed to maximize scientific gain before expensive work:

1. envelope sensitivity;
2. framework ablation;
3. Phase-1 estimability summary;
4. certification uncertainty;
5. duplicate/near-duplicate audit;
6. lightweight XAI;
7. robustness analysis;
8. subgroup analysis where supported;
9. compute profile;
10. only then run the two additional ResNet seeds.

### No external tuning

No external result may be used to tune architecture, normalization, preprocessing, thresholds, feature selection, seed selection, or augmentation. Secondary perturbation levels and sensitivity grids must be fixed in code before their result tables are assembled.

## Analysis modules

### 1. Three-seed initialization stability

Use exactly three total seeds:

- `20260808` — existing primary seed;
- `20260809` — secondary replication;
- `20260810` — secondary replication.

The two additional runs use the exact frozen ResNet architecture, training hyperparameters, folds, fold-9 calibration, external partition, and certification rules. They are not eligible to replace the primary model.

Report:

- internal macro PR-AUC, ROC-AUC, and Brier by seed;
- per-label internal metrics by seed;
- external per-pair metrics by seed;
- pair-status agreement across seeds;
- proportion of evaluable label-domain pairs whose status matches the primary seed;
- summary mean/SD and median/range where useful.

The manuscript must call this a `secondary initialization-stability analysis`, not a new primary estimate.

### 2. Certification-envelope sensitivity

Reuse frozen primary predictions. No model inference is required if audited probabilities are available through the current statistical reconstruction path.

Evaluate nearby research envelopes while retaining the official threshold as the primary row:

- PR-AUC/prevalence ratio: `{1.5, 2.0, 2.5, 3.0}`;
- maximum absolute slope deviation: `{0.25, 0.35, 0.50}`;
- maximum absolute intercept: `{0.50, 0.75, 1.00}`;
- Brier-skill requirement: `{required, not required}` as a secondary ablation only.

For every setting report counts of:

- certified;
- calibration-recovery candidate;
- discrimination failure;
- insufficient support.

The key secondary endpoint is whether the qualitative finding `calibration failure is more common than discrimination failure among evaluable pairs` remains stable around the frozen envelope.

### 3. Framework ablation

This is an ablation of the evaluation framework, not the ResNet architecture.

Compare using the same frozen predictions:

- raw sigmoid probabilities vs fold-9 global Platt calibration;
- discrimination-only gate;
- calibration-only gate;
- full gate minus slope criterion;
- full gate minus intercept criterion;
- full gate minus positive Brier-skill criterion;
- alternative support thresholds as clearly secondary sensitivity settings.

Do not change residual blocks, kernels, channels, or architecture depth. TRUST-ECG does not claim a novel backbone.

### 4. Lightweight ResNet XAI

Waveform XAI must remain low-compute and clinically interpretable.

Use:

- lead-wise occlusion;
- temporal-window occlusion;
- Integrated Gradients on a small, fixed representative sample.

Do not run KernelSHAP, waveform-level LIME, or feature-wise PDP/ICE/ALE across 60,000 raw ECG samples.

Sampling must be deterministic and limited. The XAI output should emphasize:

- per-label lead importance;
- coarse temporal attribution;
- attribution stability between PTB-XL and external domains;
- comparison of certified vs calibration-failure cases when support permits.

XAI must never be presented as causal interpretation.

### 5. Classical XAI for the 144-feature Logistic reference

The handcrafted Logistic reference is the appropriate low-cost model for classical tabular XAI.

Generate:

- SHAP global bar plot;
- SHAP beeswarm;
- representative SHAP waterfall plots;
- LIME local explanations for a small fixed case set;
- PDP and ICE for the top 4-6 continuous features;
- ALE for the same top 4-6 features.

Prefer exact/linear SHAP appropriate for Logistic Regression. Avoid expensive model-agnostic KernelSHAP unless no exact implementation is available.

All XAI results are descriptive secondary evidence.

### 6. Signal robustness analysis

Use the frozen primary model only. Do not retrain.

Apply predefined mild perturbations at inference:

- additive Gaussian noise at a small fixed set of SNR levels;
- gain scaling;
- baseline-wander perturbation;
- small temporal shift/crop shift;
- selected single-lead dropout and grouped lead dropout.

Report relative changes in PR-AUC, ROC-AUC, Brier score, and calibration where estimable. The goal is not adversarial robustness; it is practical acquisition/signal sensitivity.

### 7. Certification uncertainty

Using the existing bootstrap/statistical infrastructure, estimate uncertainty for the metrics that determine certification.

Report, where estimable:

- intervals for PR-AUC/prevalence ratio;
- intervals for calibration slope;
- intervals for calibration intercept;
- intervals for Brier skill;
- proportion of bootstrap replicates satisfying the complete official gate.

Do not replace the primary point-estimate certification matrix. Present the bootstrap gate-satisfaction rate as secondary uncertainty evidence.

### 8. Duplicate and near-duplicate audit

For external certification vs recovery partitions:

- check exact waveform hashes where possible;
- derive lightweight waveform fingerprints for near-duplicate screening;
- flag suspicious high-similarity cross-partition pairs for manual aggregate audit;
- never commit patient- or record-level waveform data or identifiers.

Output only aggregate counts, maximum similarities, thresholds, and audit status.

This audit addresses residual dependence risk where patient IDs are unavailable; it must not claim proof of patient independence when identifiers do not exist.

### 9. Phase-1 estimability reporting

Reuse existing Phase-1 records/results.

For every candidate pair, label budget, and recalibration method report:

- repeats requested;
- estimable repeats;
- non-estimable repeats;
- reasons for non-estimability;
- recovery-envelope success count/rate among estimable repeats;
- uncertainty interval for the recovery rate.

No imputation of non-estimable repeats.

### 10. Statistical cleanup

Retain bootstrap confidence intervals for performance differences.

For formal model-comparison claims, add a paired randomization/permutation test where prediction-level pairing is available. Existing bootstrap tail probabilities must be labelled accurately and not described as classical null-hypothesis p-values unless the implementation is changed accordingly.

Benjamini-Hochberg multiplicity control remains appropriate for families of secondary tests.

### 11. Subgroup analysis

Run only where metadata are present, harmonized, and adequately supported.

Primary targets:

- sex;
- age bands.

Subgroup results are descriptive secondary analyses. No subgroup-specific recalibration, model selection, or threshold tuning is permitted.

If a source lacks reliable metadata, report `not estimable` rather than infer values.

### 12. Compute profile

Report a minimal practical profile:

- trainable parameter count;
- serialized model size;
- representative CPU inference throughput/latency;
- training duration and hardware from workflow metadata where reliably available.

No extensive hardware benchmarking is required.

## Output structure

Add a separate secondary-analysis namespace, for example:

`secondary_results/trust_ecg_v04/`

with aggregate-only artifacts such as:

- `envelope_sensitivity.csv/json`;
- `framework_ablation.csv/json`;
- `certification_uncertainty.csv/json`;
- `phase1_estimability.csv/json`;
- `duplicate_audit.json`;
- `robustness.csv/json`;
- `xai_resnet_summary.json`;
- `xai_logistic_summary.json`;
- figure files for publication;
- `seed_stability.csv/json`;
- `secondary_sha256_manifest.json`.

No record-level probabilities, raw waveforms, record IDs, or unrestricted per-record attributions may be committed.

## Code organization

Prefer new isolated modules rather than modifying frozen scientific files. Likely additions:

- `src/trust_icu/ecg_secondary_sensitivity.py`;
- `src/trust_icu/ecg_secondary_xai.py`;
- `src/trust_icu/ecg_secondary_robustness.py`;
- `src/trust_icu/ecg_secondary_overlap.py`;
- `src/trust_icu/ecg_secondary_statistics.py`;
- one orchestration/reporting script under `scripts/`;
- focused tests under `tests/`;
- one secondary GitHub Actions workflow that reuses verified primary artifacts and keeps the two extra seed trainings as separate expensive jobs.

Frozen files should not be changed unless a non-scientific adapter is strictly necessary. If any change to a frozen protocol or primary implementation appears necessary, stop and obtain explicit scientific approval instead of silently changing it.

## Dependency policy

Keep dependencies lightweight.

Preferred:

- existing NumPy/SciPy/scikit-learn/PyTorch/Matplotlib stack;
- `shap` only if needed for reproducible Logistic SHAP plots;
- `lime` only for the small tabular Logistic analysis;
- use a compact in-repo ALE implementation if practical rather than adding a large dependency;
- implement Integrated Gradients and occlusion directly with PyTorch unless a small, well-justified dependency is preferable.

Do not add heavy explainability frameworks solely for convenience.

## Testing and fail-closed behavior

Add tests for:

- sensitivity classification reproduces the official gate exactly at official thresholds;
- framework ablation never mutates primary artifacts;
- deterministic XAI sample selection;
- occlusion preserves tensor shape and masks only the intended leads/windows;
- robustness perturbations are deterministic under fixed seeds;
- duplicate fingerprinting detects exact synthetic duplicates and does not expose record-level output;
- Phase-1 estimability summaries preserve non-estimable repeats;
- secondary reports verify primary protocol/model/report hashes;
- output privacy checks reject record-level identifiers or raw waveform exports;
- seed-stability aggregation treats `20260808` as primary and cannot replace it automatically.

All existing TRUST-ECG integrity, unit, and synthetic end-to-end checks must continue to pass.

## Publication integration

The manuscript should preserve the primary Results section and add compact secondary evidence under headings such as:

- `Sensitivity and Initialization Stability`;
- `Explainability and Attribution Stability`;
- `Signal Robustness and Residual Leakage Audit`.

Use multi-panel figures to avoid page inflation. The manuscript must explicitly label all new analyses as secondary/sensitivity analyses.

The primary claims may be strengthened only when supported by these results. If sensitivity, robustness, or seed replication weakens the original interpretation, report that transparently rather than changing analysis definitions.

## Success criteria

The secondary package is complete when:

1. all zero/low-compute analyses run reproducibly against the frozen primary artifacts;
2. XAI and robustness outputs are aggregate/publication-safe;
3. exact and near-duplicate audit results are reported without overclaiming patient independence;
4. two additional seeds complete under the exact frozen training protocol;
5. three-seed status stability is summarized without replacing the primary seed;
6. all repository integrity and unit tests pass;
7. every secondary artifact carries provenance back to the frozen primary protocol/model/report hashes;
8. the manuscript distinguishes primary from secondary evidence unambiguously.
