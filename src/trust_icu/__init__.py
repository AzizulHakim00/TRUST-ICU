"""TRUST-ICU research utilities.

The public package contains protocol validation, outcome-lock enforcement, canonical cohort
and feature construction, locked baselines and aggregate feasibility logic. Restricted
patient-level data must remain outside the repository.
"""

from trust_icu.baseline import (
    BaselineMetrics,
    evaluate_probabilities,
    fit_catboost_baseline,
    fit_logistic_baseline,
)
from trust_icu.cohort import LandmarkSpec, assign_task_labels, build_landmark_cohort
from trust_icu.config import StudyConfig, load_config
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
from trust_icu.validation import DatasetAudit, GateDecision, evaluate_feasibility

__all__ = [
    "BaselineMetrics",
    "DatasetAudit",
    "FeatureContract",
    "FeatureMatrixAudit",
    "GateDecision",
    "LandmarkSpec",
    "OutcomeLockReport",
    "StudyConfig",
    "VariableSpec",
    "assert_task_training_allowed",
    "assign_task_labels",
    "build_feature_matrix",
    "build_landmark_cohort",
    "classify_event_offset_minutes",
    "evaluate_feasibility",
    "evaluate_outcome_locks",
    "evaluate_probabilities",
    "fit_catboost_baseline",
    "fit_logistic_baseline",
    "load_config",
    "load_feature_contract",
    "load_outcome_contracts",
]

__version__ = "0.3.0"
