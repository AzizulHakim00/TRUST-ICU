# TRUST-ECG Secondary Validation Design

Date: 2026-09-11
Branch: `open-ecg-transportability`
Status: approved design, revised to final direct-ResNet-only scope

## Purpose

Strengthen the publication evidence around TRUST-ECG without modifying, replacing, or retrospectively tuning the frozen primary TRUST-ECG v0.4 experiment.

The primary scientific result remains the prospectively locked execution using the fixed 1D ResNet, primary seed `20260808`, PTB-XL folds 1-7 for fitting, fold 8 for early stopping, fold 9 for global calibration, fold 10 for untouched internal testing, the label-blind 60/40 external certification/recovery partition, the fixed Phase-0 certification envelope, and the frozen Phase-1 budgets and recalibration methods.

All new work is secondary, sensitivity, robustness, interpretability, audit, or replication analysis. It must not overwrite or redefine the primary report, model hash, frozen protocol, labels, split, or gate. Secondary evidence is allowed to strengthen, qualify, or weaken the interpretation; if it contradicts the original interpretation, the manuscript must report that transparently rather than changing analysis definitions.

## Scientific goals

The secondary package addresses the main remaining reviewer concerns:

1. dependence of the headline certification pattern on exact envelope thresholds;
2. dependence on neural-network initialization;
3. lack of waveform-level explainability for the frozen ResNet;
4. lack of signal robustness/stress testing;
5. lack of framework-level ablation/sensitivity analysis;
6. residual cross-partition duplicate/near-duplicate risk where external patient IDs are unavailable;
7. uncertainty around categorical certification decisions;
8. incomplete Phase-1 estimability reporting;
9. limited subgroup evidence where metadata are reliable;
10. lack of a concise compute/deployment profile.

## Design principles

### Primary analysis remains immutable

Secondary code must read but never overwrite canonical Phase-0, Phase-1, checkpoint, calibration, protocol, manifest, waveform-audit, or statistical-addendum artifacts. Every secondary report must carry hashes that bind it to the exact frozen primary state.

### Direct ResNet only

Interpretability and robustness analyses use the actual frozen TRUST-ECG ResNet. No surrogate/lightweight model, extra CNN, alternate deep backbone, SHAP, LIME, PDP, ICE, or ALE is introduced for the secondary package.

### Low-compute first

Execution order:

1. envelope sensitivity;
2. framework ablation;
3. Phase-1 estimability summary;
4. certification uncertainty;
5. duplicate/near-duplicate audit;
6. direct-ResNet XAI;
7. robustness analysis;
8. subgroup analysis where supported;
9. compute profile;
10. two additional ResNet seeds last.

### No external tuning

External results may not tune architecture, normalization, preprocessing, feature selection, seed choice, augmentation, or the primary gate. Secondary perturbation levels, sensitivity grids, and XAI sampling rules must be fixed before their results are assembled.

## 1. Three-seed initialization stability

Use exactly three total seeds:

- `20260808` — existing primary seed;
- `20260809` — secondary replication;
- `20260810` — secondary replication.

The two additional runs use the exact frozen architecture, training hyperparameters, fold roles, fold-9 calibration, external partition, and certification rules. They cannot replace the primary model.

Report internal macro PR-AUC/ROC-AUC/Brier, per-label internal metrics, external per-pair metrics, pair-status agreement across seeds, the proportion of evaluable pairs whose status agrees with the primary seed, and concise mean/SD plus median/range summaries. Manuscript wording: `secondary initialization-stability analysis`.

## 2. Certification-envelope sensitivity

Reuse frozen primary predictions. Retain the official envelope as the primary row and evaluate only secondary nearby settings:

- PR-AUC/prevalence ratio: `{1.5, 2.0, 2.5, 3.0}`;
- maximum absolute slope deviation: `{0.25, 0.35, 0.50}`;
- maximum absolute intercept: `{0.50, 0.75, 1.00}`;
- Brier-skill requirement: `{required, not required}` as an ablation only.

For each setting report counts of certified, calibration-recovery candidate, discrimination failure, and insufficient support. The key secondary endpoint is whether calibration failure remains more common than discrimination failure among evaluable pairs around the frozen envelope.

## 3. Framework ablation

Ablate the evaluation framework, not the ResNet architecture. Compare the same frozen predictions under:

- raw sigmoid vs fold-9 global Platt calibration;
- discrimination-only gate;
- calibration-only gate;
- full gate minus slope criterion;
- full gate minus intercept criterion;
- full gate minus positive Brier-skill criterion;
- clearly labelled alternative support thresholds.

Do not change residual blocks, kernels, channels, depth, optimizer, or training procedure.

## 4. Direct frozen-ResNet XAI

Use waveform-appropriate methods on the actual frozen ResNet:

- Integrated Gradients;
- 1D Grad-CAM from the last convolutional/residual stage;
- lead-wise occlusion;
- temporal-window occlusion;
- grouped lead occlusion (limb vs precordial) as a compact clinical summary.

Hard compute caps:

- occlusion evaluation sample: at most **64 ECGs per source** (PTB-XL internal test plus each of the three external certification sources), selected deterministically with a fixed coverage-oriented rule;
- temporal occlusion: **10 non-overlapping 1-second windows** per ECG;
- lead occlusion: **12 individual leads** per ECG;
- Integrated Gradients: at most **8 ECGs per source**, **24 integration steps** each;
- Grad-CAM: at most **8 ECGs per source** using the same deterministic sample as Integrated Gradients;
- if a requested label/source stratum lacks support, report `not estimable` rather than expanding the sample.

Primary outputs are per-label lead importance, coarse temporal attribution, Grad-CAM temporal localization, and rank/stability comparisons between internal and external sources. Certified vs calibration-failure attribution comparisons are secondary when support permits. Later, the three-seed package may compare attribution summaries across `20260808`, `20260809`, and `20260810` without selecting a preferred seed.

XAI is descriptive and must never be presented as causal interpretation.

## 5. Signal robustness analysis

Use the frozen primary model only; do not retrain. Predefine a small practical perturbation suite:

- additive Gaussian noise at SNR 20 dB;
- gain scaling at 0.9x and 1.1x;
- mild baseline wander using a fixed low-frequency sinusoidal perturbation;
- temporal shift of ±100 ms with deterministic padding/cropping;
- limb-lead dropout group;
- precordial-lead dropout group.

Run robustness first on the internal test and external certification partitions only. Report relative changes in PR-AUC, ROC-AUC, Brier, and calibration where estimable. This is practical signal sensitivity, not adversarial robustness.

## 6. Certification uncertainty

Use existing bootstrap/statistical infrastructure to estimate uncertainty for the metrics that drive the official gate:

- PR-AUC/prevalence ratio;
- calibration slope;
- calibration intercept;
- Brier skill;
- proportion of bootstrap replicates satisfying the complete official gate.

Do not replace the point-estimate certification matrix. The bootstrap gate-satisfaction rate is secondary uncertainty evidence.

## 7. Duplicate and near-duplicate audit

Across external certification vs recovery partitions:

- test exact waveform hashes where possible;
- derive lightweight waveform fingerprints for near-duplicate screening;
- aggregate suspicious high-similarity cross-partition matches;
- never commit raw waveforms, record IDs, or unrestricted record-level similarity tables.

Output only aggregate counts, maximum/quantile similarity summaries, thresholds, and audit status. This audit may reduce overlap concern but must not claim patient independence when patient identifiers do not exist.

## 8. Phase-1 estimability reporting

Reuse existing Phase-1 outputs. For every candidate pair, budget, and method report:

- repeats requested;
- estimable repeats;
- non-estimable repeats;
- non-estimability reasons;
- recovery-envelope success count/rate among estimable repeats;
- Wilson 95% interval for the recovery rate.

No imputation of non-estimable repeats.

## 9. Statistical cleanup

Retain bootstrap confidence intervals. For formal model-comparison claims, add paired randomization/permutation tests when prediction-level pairing is available. Existing bootstrap tail probabilities must be labelled as bootstrap tail probabilities unless the implementation is changed to a formal null procedure. Apply Benjamini-Hochberg control to coherent secondary-test families.

## 10. Subgroup analysis

Run only where reliable harmonized metadata and adequate support exist. Primary subgroups are sex and age bands. Results are descriptive secondary analyses; no subgroup-specific recalibration, model selection, or threshold tuning is allowed. Missing/unreliable metadata produce `not estimable`.

## 11. Compute profile

Report trainable parameter count, serialized model size, representative CPU inference latency/throughput, and training duration/hardware from trustworthy workflow metadata. No extensive hardware benchmarking.

## Output structure

Use a separate aggregate-only namespace such as:

`secondary_results/trust_ecg_v04/`

Expected artifacts include:

- `envelope_sensitivity.csv/json`;
- `framework_ablation.csv/json`;
- `certification_uncertainty.csv/json`;
- `phase1_estimability.csv/json`;
- `duplicate_audit.json`;
- `robustness.csv/json`;
- `xai_resnet_summary.json`;
- publication figures;
- `seed_stability.csv/json`;
- `secondary_sha256_manifest.json`.

No record-level probabilities, raw waveforms, record IDs, or unrestricted per-record attributions may be committed.

## Code organization

Prefer new isolated modules rather than modifying frozen scientific files:

- `src/trust_icu/ecg_secondary_sensitivity.py`;
- `src/trust_icu/ecg_secondary_xai.py`;
- `src/trust_icu/ecg_secondary_robustness.py`;
- `src/trust_icu/ecg_secondary_overlap.py`;
- `src/trust_icu/ecg_secondary_statistics.py`;
- one orchestration/reporting script under `scripts/`;
- focused tests under `tests/`;
- one secondary GitHub Actions workflow, with the two extra seed trainings isolated as expensive jobs.

Frozen primary files must not change unless a non-scientific adapter is strictly necessary. If implementation appears to require a protocol/scientific change, stop for explicit scientific approval.

## Dependency policy

Keep dependencies lightweight. Prefer existing NumPy/SciPy/scikit-learn/PyTorch/Matplotlib. Implement Integrated Gradients, 1D Grad-CAM, and occlusion directly with PyTorch. Do not add a generic XAI framework solely for convenience.

## Testing and fail-closed behavior

Add tests that verify:

- official sensitivity settings reproduce the existing official gate exactly;
- secondary analyses cannot overwrite primary artifacts;
- deterministic XAI sample selection and hard sample caps;
- occlusion masks only intended leads/windows and preserves tensor shape;
- Integrated Gradients enforces the 8-per-source/24-step caps;
- Grad-CAM targets the intended final convolutional stage and returns finite temporal maps;
- robustness perturbations are deterministic;
- duplicate fingerprinting detects exact synthetic duplicates without exposing IDs;
- Phase-1 estimability preserves non-estimable repeats;
- secondary reports verify primary protocol/model/report hashes;
- privacy checks reject record-level identifiers and raw waveform exports;
- seed aggregation treats `20260808` as primary and cannot auto-replace it.

All existing TRUST-ECG integrity, unit, and synthetic end-to-end checks must continue to pass.

## Publication integration

Preserve the primary Results section. Add compact secondary evidence under headings such as `Sensitivity and Initialization Stability`, `Explainability and Attribution Stability`, and `Signal Robustness and Residual Leakage Audit`. Use multi-panel figures to control page count. Every new analysis must be labelled secondary/sensitivity evidence.

## Success criteria

The package is complete when:

1. zero/low-compute analyses run reproducibly against frozen primary artifacts;
2. direct-ResNet XAI and robustness outputs are aggregate/publication-safe and obey compute caps;
3. overlap auditing is reported without overclaiming patient independence;
4. the two additional seeds complete under the exact frozen training protocol;
5. three-seed status stability is summarized without replacing the primary seed;
6. all repository integrity and unit tests pass;
7. every secondary artifact is provenance-linked to frozen primary hashes;
8. the manuscript distinguishes primary from secondary evidence unambiguously and reports contrary secondary findings transparently.
