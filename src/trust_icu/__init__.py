"""TRUST-ICU research utilities.

The public package contains protocol validation and aggregate feasibility logic.
Restricted patient-level data must remain outside the repository.
"""

from trust_icu.config import StudyConfig, load_config
from trust_icu.validation import DatasetAudit, GateDecision, evaluate_feasibility

__all__ = [
    "DatasetAudit",
    "GateDecision",
    "StudyConfig",
    "evaluate_feasibility",
    "load_config",
]

__version__ = "0.1.0"
