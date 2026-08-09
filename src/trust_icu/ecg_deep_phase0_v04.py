"""Protocol-v0.4 adapter for the optional fixed TRUST-ECG ResNet runner."""

from __future__ import annotations

from pathlib import Path

from trust_icu.ecg_phase0_v04 import load_and_verify_model_index, load_standardized_record


def execute_fixed_resnet_phase0(
    *,
    primary_data_root: str | Path,
    index_csv: str | Path,
    index_audit_path: str | Path,
    label_manifest_path: str | Path,
    waveform_audit_path: str | Path,
    normalization_stats_path: str | Path,
    protocol_path: str | Path,
    output_root: str | Path,
    device_name: str | None = None,
    num_workers: int = 0,
):
    """Run the legacy fixed ResNet statistics/training loop with the v0.4 data boundary."""

    from trust_icu import ecg_deep_phase0 as legacy_deep

    # The legacy module owns the mature training/calibration/report logic. These two globals are
    # the only source-specific data boundary and are replaced before any Dataset is constructed.
    legacy_deep.load_and_verify_model_index = load_and_verify_model_index
    legacy_deep._load_standardized_record = load_standardized_record
    return legacy_deep.execute_fixed_resnet_phase0(
        challenge_training_root=primary_data_root,
        index_csv=index_csv,
        index_audit_path=index_audit_path,
        label_manifest_path=label_manifest_path,
        waveform_audit_path=waveform_audit_path,
        normalization_stats_path=normalization_stats_path,
        protocol_path=protocol_path,
        output_root=output_root,
        device_name=device_name,
        num_workers=num_workers,
    )
