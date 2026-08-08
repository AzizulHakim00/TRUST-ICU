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
_EXPECTED_LR_FEATURES = [
    "mean",
    "std",
    "minimum",
    "maximum",
    "median",
    "q05",
    "q25",
    "q75",
    "q95",
    "rms",
    "mean_absolute_first_difference",
    "std_first_difference",
]


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

    if str(protocol.get("version")) != "0.3.0":
        raise ValueError("Open ECG protocol version must remain pinned to 0.3.0 for this stage.")
    if protocol.get("status") != "prospective_open_data_pivot":
        raise ValueError("Open ECG protocol must remain prospective before real performance inspection.")

    interpretation = protocol.get("interpretation_scope")
    if not isinstance(interpretation, dict):
        raise ValueError("Interpretation scope is required.")
    for key in (
        "certification_is_research_endpoint_not_clinical_approval",
        "prohibit_claim_of_regulatory_or_bedside_safety",
        "failure_to_meet_envelope_means_not_certified_under_this_protocol_only",
    ):
        if interpretation.get(key) is not True:
            raise ValueError(f"Research-certification interpretation safeguard must remain true: {key}")

    resource = protocol.get("resource")
    if not isinstance(resource, dict):
        raise ValueError("Resource block is required.")
    if resource.get("name") != "PhysioNet/Computing in Cardiology Challenge 2020":
        raise ValueError("Unexpected ECG resource.")
    if str(resource.get("version")) != "1.0.2":
        raise ValueError("Challenge 2020 version must remain pinned to 1.0.2.")
    if resource.get("access_policy") != "open" or resource.get("license") != "CC-BY-4.0":
        raise ValueError("This pivot must remain open-data under the pinned CC-BY-4.0 release.")
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
    if signal.get("leads") != _EXPECTED_LEADS or signal.get("physical_unit") != "mV":
        raise ValueError("Primary signal must remain standard 12-lead physical mV.")
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

    partition = protocol.get("external_partition")
    if not isinstance(partition, dict):
        raise ValueError("Independent external certification/recovery partition is required.")
    if partition.get("method") != "sha256_seeded_record_id_partition_without_labels":
        raise ValueError("External split must remain deterministic and label-blind.")
    if partition.get("seed") != 20260808:
        raise ValueError("External partition seed must remain locked.")
    if partition.get("certification_fraction") != 0.60 or partition.get("recovery_pool_fraction") != 0.40:
        raise ValueError("External partition must remain 60% certification and 40% untouched recovery pool.")
    if partition.get("label_stratification") is not False:
        raise ValueError("External partition must not use labels.")
    if partition.get("prohibit_repartition_after_label_or_performance_inspection") is not True:
        raise ValueError("External repartitioning after result inspection must remain prohibited.")

    models = protocol.get("phase0_models")
    if not isinstance(models, dict):
        raise ValueError("Phase 0 model block is required.")
    if models.get("primary_model") != "resnet1d_fixed":
        raise ValueError("The fixed 1D ResNet must remain the predeclared primary model.")
    if models.get("model_selection_between_baselines") != "none_primary_model_is_predeclared":
        raise ValueError("Phase 0 must not select between baselines from performance.")
    logistic = models.get("logistic_regression_handcrafted")
    if not isinstance(logistic, dict):
        raise ValueError("Logistic reference specification is required.")
    if logistic.get("per_lead_features") != _EXPECTED_LR_FEATURES or logistic.get("feature_count") != 144:
        raise ValueError("Handcrafted Logistic Regression feature contract must remain exactly 144 features.")
    if logistic.get("C") != 1.0 or logistic.get("class_weight") != "balanced":
        raise ValueError("Logistic Regression hyperparameters must remain fixed.")
    resnet = models.get("resnet1d_fixed")
    if not isinstance(resnet, dict):
        raise ValueError("Fixed 1D ResNet specification is required.")
    if resnet.get("input_shape") != [12, 5000]:
        raise ValueError("Fixed ResNet input shape must remain 12 x 5000.")
    if resnet.get("residual_stages", {}).get("channels") != [64, 128, 256, 512]:
        raise ValueError("Fixed ResNet channel schedule cannot drift.")
    if resnet.get("residual_stages", {}).get("blocks_per_stage") != [2, 2, 2, 2]:
        raise ValueError("Fixed ResNet stage depth cannot drift.")
    if resnet.get("batch_size") != 64 or resnet.get("max_epochs") != 50:
        raise ValueError("Fixed ResNet training budget cannot drift.")
    if resnet.get("data_augmentation") != "none_primary_analysis":
        raise ValueError("Primary ResNet analysis must not add post-hoc augmentation.")
    architecture_search = models.get("architecture_search")
    if not isinstance(architecture_search, dict) or architecture_search.get("allowed") is not False:
        raise ValueError("Architecture search must remain disabled in Phase 0.")

    calibration = protocol.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Calibration contract is required.")
    if calibration.get("method") != "independent_platt_scaling_per_label":
        raise ValueError("Calibration must remain independent per-label Platt scaling.")
    if calibration.get("fit_source") != "ptb_xl_fold_9_only":
        raise ValueError("Probability calibration may only use PTB-XL fold 9.")
    if calibration.get("external_recalibration_in_phase0") != "prohibited":
        raise ValueError("External recalibration is prohibited in Phase 0.")

    go_no_go = protocol.get("phase0_go_no_go")
    if not isinstance(go_no_go, dict):
        raise ValueError("Phase 0 research certification envelope is required.")
    if go_no_go.get("evaluation_unit") != "label_domain_pair":
        raise ValueError("Phase 0 gate must remain label-domain specific.")
    if go_no_go.get("model_used_for_primary_gate") != "resnet1d_fixed":
        raise ValueError("Only the predeclared primary ResNet may determine the Phase 0 gate.")
    if go_no_go.get("external_partition_used") != "certification":
        raise ValueError("Phase 0 must use only the certification partition.")
    if go_no_go.get("require_ptbxl_crosswalk_audit") is not True:
        raise ValueError("Phase 0 must require the PTB-XL crosswalk audit.")
    if go_no_go.get("discrimination_viability", {}).get("minimum_pr_auc_to_prevalence_ratio") != 2.0:
        raise ValueError("Discrimination viability threshold cannot drift.")
    envelope = go_no_go.get("calibration_envelope")
    if not isinstance(envelope, dict):
        raise ValueError("Calibration envelope is required.")
    if envelope.get("maximum_absolute_calibration_slope_deviation") != 0.35:
        raise ValueError("Calibration slope envelope cannot drift.")
    if envelope.get("maximum_absolute_calibration_intercept") != 0.75:
        raise ValueError("Calibration intercept envelope cannot drift.")
    if envelope.get("require_positive_brier_skill_vs_prevalence") is not True:
        raise ValueError("Positive Brier skill versus prevalence must remain required.")
    if go_no_go.get("phase1_activation", {}).get(
        "minimum_external_domains_with_at_least_one_recovery_candidate"
    ) != 2:
        raise ValueError("Phase 1 recovery must require calibration failures in at least two external domains.")

    phase1 = protocol.get("phase1_if_phase0_passes")
    if not isinstance(phase1, dict) or phase1.get("target_label_budgets") != _EXPECTED_BUDGETS:
        raise ValueError("Target-domain label budgets must remain prespecified.")
    if phase1.get("data_source") != "untouched_external_recovery_pool_only":
        raise ValueError("Phase 1 must use only the untouched external recovery pool.")
    sampling = phase1.get("sampling")
    if not isinstance(sampling, dict) or sampling.get("repeats") != 100:
        raise ValueError("Phase 1 must retain 100 repeated label-budget samples.")
    if sampling.get("draw_records_uniformly_without_replacement") is not True:
        raise ValueError("Phase 1 label acquisition must remain uniform without replacement.")
    phase1_eval = phase1.get("evaluation")
    if not isinstance(phase1_eval, dict) or phase1_eval.get("adaptation_records_excluded_from_evaluation") is not True:
        raise ValueError("Phase 1 adaptation records must be excluded from their evaluation set.")
    prohibited = set(phase1.get("prohibited", []))
    required_prohibited = {
        "target_domain_model_retraining",
        "target_domain_feature_selection",
        "target_domain_normalization_refit",
        "posthoc_threshold_search",
        "best_domain_reporting",
    }
    if not required_prohibited.issubset(prohibited):
        raise ValueError("Phase 1 prohibited target-domain operations cannot be relaxed.")

    novelty = protocol.get("novelty_governance")
    if not isinstance(novelty, dict) or novelty.get("novelty_claim_status") != "candidate_only":
        raise ValueError("Novelty claims must remain candidate-only until refreshed literature review.")
    if novelty.get("literature_collision_audit") != "docs/ecg_literature_collision_audit_2026-08-08.md":
        raise ValueError("The prospective ECG literature collision audit must remain linked.")

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
        "external_certification_fraction": 0.60,
        "external_recovery_fraction": 0.40,
        "public_primary_records": 21837 + sum(_EXPECTED_EXTERNAL.values()),
        "primary_model": "resnet1d_fixed",
        "logistic_feature_count": 144,
        "label_budgets": list(_EXPECTED_BUDGETS),
        "architecture_search_allowed": False,
    }
