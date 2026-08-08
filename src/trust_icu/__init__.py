"""TRUST-ICU and TRUST-ECG research utilities.

The public package contains protocol validation, source-adapter audits, secure credentialed
execution, evidence-based outcome locking, canonical cohort and feature construction, locked
baselines, temporal and external Phase 0 orchestration, aggregate feasibility logic,
publication-grade aggregate reporting, fail-closed conditional Phase 1 activation, and the
prospective open ECG transportability protocol. Restricted patient-level data must remain outside
the repository.
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
from trust_icu.ecg_data import (
    EcgHeaderAudit,
    HeaderRecord,
    LabelDecision,
    build_header_audit,
    parse_challenge_header,
    scan_headers,
    validate_ptbxl_crosswalk,
    validate_ptbxl_folds,
    write_header_audit,
)
from trust_icu.ecg_manifest import (
    EcgLabelManifest,
    build_label_manifest,
    load_and_verify_header_audit,
    load_and_verify_label_manifest,
    write_label_manifest,
)
from trust_icu.ecg_protocol import load_open_ecg_protocol, validate_open_ecg_protocol
from trust_icu.ecg_signal import (
    NormalizationStats,
    SignalSpec,
    StandardizedSignal,
    StreamingLeadStats,
    digital_to_physical_mv,
    load_mat_digital_signal,
    normalize_signal,
    parse_signal_header,
    standardize_signal,
    write_normalization_stats,
)
from trust_icu.ecg_waveform import (
    EcgWaveformAudit,
    PtbxlAssignment,
    assignment_sha256,
    build_verified_ptbxl_assignments,
    load_and_verify_waveform_audit,
    prepare_waveform_stage,
    write_ptbxl_assignment,
)
from trust_icu.features import (
    FeatureContract,
    FeatureMatrixAudit,
    VariableSpec,
    build_feature_matrix,
    load_feature_contract,
)
from trust_icu.outcome_evidence import (
    LocalOutcomeSummary,
    OutcomeEvidenceReport,
    OutcomeSummaryTask,
    build_local_outcome_summary,
    prepare_locked_runtime_context,
    validate_outcome_evidence,
)
from trust_icu.outcomes import (
    OutcomeLockReport,
    assert_task_training_allowed,
    classify_event_offset_minutes,
    evaluate_outcome_locks,
    load_outcome_contracts,
)
from trust_icu.phase0_runtime import (
    Phase0BaselineReport,
    TaskPhase0Report,
    build_feature_matrix_from_extract,
    build_phase0_dry_run_plan,
    execute_phase0_baselines,
    run_task_phase0,
    temporal_patient_purged_split,
    verify_credentialed_run,
)
from trust_icu.phase1_gate import (
    Phase1ActivationReport,
    evaluate_phase1_activation,
    load_phase1_protocol,
    validate_phase1_protocol,
)
from trust_icu.reporting import (
    ReportingBundle,
    build_reporting_dry_run_plan,
    generate_publication_bundle,
    load_and_verify_phase0_report,
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
    "EcgHeaderAudit",
    "EcgLabelManifest",
    "EcgWaveformAudit",
    "ExportArtifact",
    "FeatureContract",
    "FeatureMatrixAudit",
    "GateDecision",
    "HeaderRecord",
    "LabelDecision",
    "LandmarkSpec",
    "LocalOutcomeSummary",
    "NormalizationStats",
    "OutcomeEvidenceReport",
    "OutcomeLockReport",
    "OutcomeSummaryTask",
    "Phase0BaselineReport",
    "Phase1ActivationReport",
    "PtbxlAssignment",
    "ReportingBundle",
    "SignalSpec",
    "StandardizedSignal",
    "StreamingLeadStats",
    "StudyConfig",
    "TaskPhase0Report",
    "VariableSpec",
    "assert_task_training_allowed",
    "assign_task_labels",
    "assignment_sha256",
    "audit_canonical_extract",
    "build_dry_run_plan",
    "build_feature_matrix",
    "build_feature_matrix_from_extract",
    "build_header_audit",
    "build_label_manifest",
    "build_landmark_cohort",
    "build_local_outcome_summary",
    "build_phase0_dry_run_plan",
    "build_reporting_dry_run_plan",
    "build_verified_ptbxl_assignments",
    "classify_event_offset_minutes",
    "digital_to_physical_mv",
    "evaluate_feasibility",
    "evaluate_outcome_locks",
    "evaluate_phase1_activation",
    "evaluate_probabilities",
    "execute_credentialed_run",
    "execute_phase0_baselines",
    "fit_catboost_baseline",
    "fit_logistic_baseline",
    "generate_publication_bundle",
    "load_and_validate_adapter_manifest",
    "load_and_verify_header_audit",
    "load_and_verify_label_manifest",
    "load_and_verify_phase0_report",
    "load_and_verify_waveform_audit",
    "load_config",
    "load_feature_contract",
    "load_mat_digital_signal",
    "load_open_ecg_protocol",
    "load_outcome_contracts",
    "load_phase1_protocol",
    "load_source_adapter_contract",
    "normalize_signal",
    "parse_challenge_header",
    "parse_signal_header",
    "prepare_eicu_mapping_tables",
    "prepare_locked_runtime_context",
    "prepare_waveform_stage",
    "run_task_phase0",
    "scan_headers",
    "standardize_signal",
    "temporal_patient_purged_split",
    "validate_adapter_manifest",
    "validate_open_ecg_protocol",
    "validate_outcome_evidence",
    "validate_phase1_protocol",
    "validate_ptbxl_crosswalk",
    "validate_ptbxl_folds",
    "validate_source_adapter_contract",
    "verify_credentialed_run",
    "write_header_audit",
    "write_label_manifest",
    "write_normalization_stats",
    "write_ptbxl_assignment",
]

__version__ = "0.14.0"
