"""TRUST-ICU research utilities.

The public package contains protocol validation, source-adapter audits, secure credentialed
execution, outcome-lock enforcement, canonical cohort and feature construction, locked
baselines, temporal and external Phase 0 orchestration, and aggregate feasibility logic.
Restricted patient-level data must remain outside the repository.
"""

from trust_icu.adapter_manifest import (
    AdapterDatasetReport,
    AdapterManifestReport,
    load_and_validate_adapter_manifest,
    validate_adapter_manifest,
)
from trust_icu.baseline import (
    BaselineMetrics,
    evaluate_probabilities,
    fit_catboost_baseline,
    fit_logistic_baseline,
)
from trust_icu.cohort import LandmarkSpec, assign_task_labels, build_landmark_cohort
from trust_icu.config import StudyConfig, load_config
from trust_icu.credentialed_runner import (
    CredentialedRunReport,
    ExportArtifact,
    build_dry_run_plan,
    execute_credentialed_run,
    prepare_eicu_mapping_tables,
)
from trust_icu.features import (
    FeatureContract,
    FeatureMatrixAudit,
    VariableSpec,
    build_feature_matrix,
    load_feature_contract,
)
from trust_icu.outcomes import (
    OutcomeLockReport,
    assert_task_training_allowed,
    classify_event_offset_minutes,
    evaluate_outcome_locks,
    load_outcome_contracts,
)
from trust_icu.phase0_runner import (
    Phase0BaselineReport,
    TaskPhase0Report,
    build_feature_matrix_from_extract,
    build_phase0_dry_run_plan,
    execute_phase0_baselines,
    run_task_phase0,
    temporal_patient_purged_split,
    verify_credentialed_run,
)
from trust_icu.source_validation import (
    CanonicalExtractAudit,
    audit_canonical_extract,
    load_source_adapter_contract,
    validate_source_adapter_contract,
)
from trust_icu.validation import DatasetAudit, GateDecision, evaluate_feasibility

__all__ = [
    "AdapterDatasetReport",
    "AdapterManifestReport",
    "BaselineMetrics",
    "CanonicalExtractAudit",
    "CredentialedRunReport",
    "DatasetAudit",
    "ExportArtifact",
    "FeatureContract",
    "FeatureMatrixAudit",
    "GateDecision",
    "LandmarkSpec",
    "OutcomeLockReport",
    "Phase0BaselineReport",
    "StudyConfig",
    "TaskPhase0Report",
    "VariableSpec",
    "assert_task_training_allowed",
    "assign_task_labels",
    "audit_canonical_extract",
    "build_dry_run_plan",
    "build_feature_matrix",
    "build_feature_matrix_from_extract",
    "build_landmark_cohort",
    "build_phase0_dry_run_plan",
    "classify_event_offset_minutes",
    "evaluate_feasibility",
    "evaluate_outcome_locks",
    "evaluate_probabilities",
    "execute_credentialed_run",
    "execute_phase0_baselines",
    "fit_catboost_baseline",
    "fit_logistic_baseline",
    "load_and_validate_adapter_manifest",
    "load_config",
    "load_feature_contract",
    "load_outcome_contracts",
    "load_source_adapter_contract",
    "prepare_eicu_mapping_tables",
    "run_task_phase0",
    "temporal_patient_purged_split",
    "validate_adapter_manifest",
    "validate_source_adapter_contract",
    "verify_credentialed_run",
]

__version__ = "0.7.0"
