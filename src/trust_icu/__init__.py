"""TRUST-ICU research utilities.

The public package contains protocol validation, outcome-lock enforcement and aggregate
feasibility logic. Restricted patient-level data must remain outside the repository.
"""

from trust_icu.config import StudyConfig, load_config
from trust_icu.outcomes import (
    OutcomeLockReport,
    assert_task_training_allowed,
    classify_event_offset_minutes,
    evaluate_outcome_locks,
    load_outcome_contracts,
)
from trust_icu.validation import DatasetAudit, GateDecision, evaluate_feasibility

__all__ = [
    "DatasetAudit",
    "GateDecision",
    "OutcomeLockReport",
    "StudyConfig",
    "assert_task_training_allowed",
    "classify_event_offset_minutes",
    "evaluate_feasibility",
    "evaluate_outcome_locks",
    "load_config",
    "load_outcome_contracts",
]

__version__ = "0.2.0"
