"""Prospective protocol validation for the open-data TRUST-ECG pivot.

This module validates only public study configuration. It never downloads or reads ECG waveforms.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

_EXPECTED_EXTERNAL = {
    "georgia": 10344,
    "cpsc_2018": 6877,
    "cpsc_2018_extra": 3453,
}
_EXPECTED_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_EXPECTED_BUDGETS = [0, 50, 100, 250, 500, 1000]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_open_ecg_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    payload = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Open ECG protocol must be a YAML mapping.")
    return payload


def validate_open_ecg_protocol(path: str | Path) -> dict[str, Any]:
    """Validate the prospective open ECG study contract and return a safe summary."""

    protocol_path = Path(path).resolve()
    protocol = load_open_ecg_protocol(protocol_path)

    if protocol.get("status") != "prospective_open_data_pivot":
        raise ValueError("Open ECG protocol must remain prospective before real performance inspection.")

    resource = protocol.get("resource")
    if not isinstance(resource, dict):
        raise ValueError("Resource block is required.")
    if resource.get("name") != "PhysioNet/Computing in Cardiology Challenge 2020":
        raise ValueError("Unexpected ECG resource.")
    if str(resource.get("version")) != "1.0.2":
        raise ValueError("Challenge 2020 version must remain pinned to 1.0.2.")
    if resource.get("access_policy") != "open":
        raise ValueError("This pivot must use openly accessible data.")
    if resource.get("license") != "CC-BY-4.0":
        raise ValueError("Unexpected file license in protocol.")
    hidden = resource.get("hidden_challenge_test_set")
    if not isinstance(hidden, dict) or hidden.get("use") is not False:
        raise ValueError("The unreleased Challenge test set must not be used.")

    sources = protocol.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("Source roles are required.")
    development = sources.get("development")
    if not isinstance(development, dict) or set(development) != {"ptb_xl"}:
        raise ValueError("PTB-XL must be the sole development source in Phase 0.")
    ptb_xl = development["ptb_xl"]
    if ptb_xl.get("records") != 21837 or str(ptb_xl.get("source_version")) != "1.0.1":
        raise ValueError("PTB-XL must remain pinned to v1.0.1 with 21,837 records.")
    if ptb_xl.get("patientwise_split_required") is not True:
        raise ValueError("PTB-XL patient-wise splitting is mandatory.")
    crosswalk = ptb_xl.get("challenge_to_original_crosswalk")
    if not isinstance(crosswalk, dict):
        raise ValueError("PTB-XL Challenge/original crosswalk rules are required.")
    if crosswalk.get("required") is not True or crosswalk.get("require_every_pair_verified") is not True:
        raise ValueError("Every Challenge/PTB-XL pair must be crosswalk-verified.")
    if crosswalk.get("prohibit_unverified_filename_formula") is not True:
        raise ValueError("An unverified PTB-XL filename formula must never be trusted.")

    external = sources.get("external_primary")
    if not isinstance(external, dict) or set(external) != set(_EXPECTED_EXTERNAL):
        raise ValueError("External source set must exactly match Georgia, CPSC2018 and CPSC2018 Extra.")
    for name, expected_records in _EXPECTED_EXTERNAL.items():
        if external[name].get("records") != expected_records:
            raise ValueError(f"Unexpected record count for external source {name}.")

    task = protocol.get("prediction_task")
    if not isinstance(task, dict) or task.get("type") != "multi_label_classification":
        raise ValueError("Primary ECG task must remain multi-label classification.")
    labels = task.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("Label harmonization rules are required.")
    if labels.get("lock_before_waveform_model_training") is not True:
        raise ValueError("Label manifest must be locked before waveform model training.")
    if labels.get("prohibit_posthoc_label_addition") is not True:
        raise ValueError("Post-hoc label addition must remain prohibited.")
    if labels.get("prohibit_posthoc_label_removal_after_performance_inspection") is not True:
        raise ValueError("Post-hoc label removal after performance inspection must remain prohibited.")

    signal = protocol.get("signal_contract")
    if not isinstance(signal, dict):
        raise ValueError("Signal contract is required.")
    if signal.get("target_sampling_rate_hz") != 500 or signal.get("target_duration_seconds") != 10:
        raise ValueError("Primary representation must remain 10 seconds at 500 Hz.")
    if signal.get("leads") != _EXPECTED_LEADS:
        raise ValueError("Signal contract must contain the standard 12 leads in locked order.")
    if signal.get("physical_unit") != "mV":
        raise ValueError("Primary signal representation must use physical millivolts.")
    if signal.get("primary_filtering") != "none_beyond_physical_unit_conversion_resampling_and_windowing":
        raise ValueError("Primary ECG filtering must remain disabled before prospective amendment.")
    if signal.get("normalization_fit_folds") != [1, 2, 3, 4, 5, 6, 7]:
        raise ValueError("Normalization may only be fitted on PTB-XL folds 1-7.")
    if signal.get("padding_excluded_from_normalization_statistics") is not True:
        raise ValueError("Padding must be excluded from normalization statistics.")
    if signal.get("no_external_domain_normalization_fit") is not True:
        raise ValueError("External-domain normalization fitting must remain prohibited.")

    validation = protocol.get("internal_validation")
    if not isinstance(validation, dict):
        raise ValueError("Internal validation contract is required.")
    if validation.get("model_fit_folds") != [1, 2, 3, 4, 5, 6, 7]:
        raise ValueError("PTB-XL model-fitting folds must remain 1-7.")
    if validation.get("optimization_validation_fold") != 8:
        raise ValueError("PTB-XL fold 8 must remain optimization-only validation.")
    if validation.get("calibration_fold") != 9 or validation.get("internal_test_fold") != 10:
        raise ValueError("PTB-XL folds 9 and 10 must remain calibration and untouched test respectively.")
    if validation.get("resnet_optimization_allowed") != "early_stopping_epoch_only":
        raise ValueError("Only the ResNet stopping epoch may be optimized in Phase 0.")
    if validation.get("all_other_phase0_hyperparameters_fixed_before_training") is not True:
        raise ValueError("All other Phase 0 hyperparameters must be fixed before training.")
    if validation.get("no_refit_using_optimization_or_calibration_fold") is not True:
        raise ValueError("Optimization and calibration folds must not be folded back into model fitting.")
    if validation.get("external_data_for_model_selection") != "prohibited":
        raise ValueError("External data must not be used for model selection.")

    models = protocol.get("phase0_models")
    if not isinstance(models, dict):
        raise ValueError("Phase 0 model block is required.")
    architecture_search = models.get("architecture_search")
    if not isinstance(architecture_search, dict) or architecture_search.get("allowed") is not False:
        raise ValueError("Architecture search must remain disabled in Phase 0.")

    go_no_go = protocol.get("phase0_go_no_go")
    if not isinstance(go_no_go, dict) or go_no_go.get("require_ptbxl_crosswalk_audit") is not True:
        raise ValueError("Phase 0 must require the PTB-XL crosswalk audit.")

    phase1 = protocol.get("phase1_if_phase0_passes")
    if not isinstance(phase1, dict) or phase1.get("target_label_budgets") != _EXPECTED_BUDGETS:
        raise ValueError("Target-domain label budgets must remain prespecified.")

    novelty = protocol.get("novelty_governance")
    if not isinstance(novelty, dict) or novelty.get("novelty_claim_status") != "candidate_only":
        raise ValueError("Novelty claims must remain candidate-only until refreshed literature review.")

    return {
        "valid": True,
        "study_name": str(protocol.get("study_name")),
        "version": str(protocol.get("version")),
        "protocol_sha256": _sha256_file(protocol_path),
        "development_source": "ptb_xl",
        "development_records": 21837,
        "model_fit_folds": [1, 2, 3, 4, 5, 6, 7],
        "optimization_validation_fold": 8,
        "calibration_fold": 9,
        "internal_test_fold": 10,
        "external_sources": dict(_EXPECTED_EXTERNAL),
        "public_primary_records": 21837 + sum(_EXPECTED_EXTERNAL.values()),
        "label_budgets": list(_EXPECTED_BUDGETS),
        "architecture_search_allowed": False,
    }
