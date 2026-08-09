# TRUST-ECG v0.4 fixed-ResNet activation gate

The primary fixed ResNet must not start merely because the code path exists. It may start only after
the real v0.4 Logistic reference workflow completes successfully and the exact aggregate study state
has been frozen into a one-time activation manifest.

## Why this gate exists

The Logistic model is a pipeline reference, not the primary scientific model. Its purpose is to prove
that the real-data download, source-aware waveform loading, normalization, model-index partitioning,
fold-9 calibration, fold-10 internal evaluation, and 60% external certification chain can complete
without touching the 40% recovery pools.

A successful Logistic run therefore authorizes execution of the already-frozen ResNet implementation;
it does **not** authorize architecture tuning, label changes, split changes, threshold optimization, or
recovery-pool access.

## Required activation manifest

After the Logistic workflow is complete, write the activation JSON to:

`triggers/open-ecg-resnet-phase0/activation.json`

The manifest is validated by `scripts/validate_open_ecg_resnet_activation.py`. It must bind:

- completed successful GitHub Actions run ID, workflow path, branch, and 40-character head SHA;
- Logistic report SHA-256;
- header-audit, PTB-XL label-concordance, label-manifest, waveform-audit, normalization,
  verified PTB assignment, model-index, and model-index-audit SHA-256 values;
- the exact locked label-code order;
- `logistic_primary_gate_eligible = false`;
- `external_recovery_pool_used = false`;
- `challenge_ptbxl_model_input = false`;
- waveform-audit and model-index readiness.

## Runtime guards

The fixed-ResNet workflow then performs four fail-closed checks before training:

1. It verifies the referenced Logistic run directly against the GitHub Actions API and requires
   `status=completed` and `conclusion=success`.
2. It compares the protected ECG scientific implementation against the Logistic run head SHA.
   Model, protocol, source-loading, indexing, calibration, and Phase-0 execution code must be unchanged.
3. It reconstructs the real public v0.4 evidence from authoritative sources.
4. It requires all reconstructed aggregate hashes and the locked label order to exactly match the
   activation manifest.

Only after all four checks pass does the workflow run the prospectively fixed ResNet on CPU with the
protocol-declared hyperparameters. No result-dependent architecture or threshold change is permitted.

## Output boundary

The workflow prints aggregate research evidence only. It does not upload waveforms, checkpoints,
record-level indexes, calibrators, predictions, or patient-level artifacts.

The 40% external recovery pools remain prohibited in Phase 0. A ResNet pair can enter later
calibration-recovery analysis only if the prospectively defined primary Phase-0 report makes it eligible.
