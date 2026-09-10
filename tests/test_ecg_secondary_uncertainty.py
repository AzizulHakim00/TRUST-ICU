from __future__ import annotations

import numpy as np
from trust_icu.ecg_secondary_uncertainty import (
    bootstrap_gate_uncertainty,
    parse_header_demographics,
)


def test_gate_uncertainty_is_deterministic_and_aggregate_only() -> None:
    rng = np.random.default_rng(7)
    y = np.array([0] * 140 + [1] * 60, dtype=np.int64)
    p = np.concatenate(
        [rng.uniform(0.01, 0.25, 140), rng.uniform(0.65, 0.98, 60)]
    )
    first = bootstrap_gate_uncertainty(y, p, repeats=120, seed=20260808)
    second = bootstrap_gate_uncertainty(y, p, repeats=120, seed=20260808)
    assert first == second
    assert first["repeats_requested"] == 120
    assert first["estimable_repeats"] > 0
    assert 0.0 <= first["complete_gate_satisfaction_rate"] <= 1.0
    assert first["pr_auc_to_prevalence_ratio_q025"] <= first["pr_auc_to_prevalence_ratio_q975"]
    serialized = repr(first).lower()
    assert "record_id" not in serialized
    assert "probabilities" not in serialized


def test_gate_uncertainty_reports_insufficient_primary_support() -> None:
    y = np.array([0] * 30 + [1] * 20, dtype=np.int64)
    p = np.linspace(0.1, 0.9, y.size)
    result = bootstrap_gate_uncertainty(y, p, repeats=50)
    assert result["status"] == "insufficient_support"
    assert result["estimable_repeats"] == 0


def test_parse_header_demographics_is_conservative() -> None:
    header = """A0001 12 500 5000\n# Age: 67\n# Sex: Female\n#Dx: 426783006\n"""
    result = parse_header_demographics(header)
    assert result == {"age": 67.0, "sex": "female", "age_band": "65_plus"}

    unknown = parse_header_demographics("A0002 12 500 5000\n#Age: NaN\n#Sex: Unknown\n")
    assert unknown == {"age": None, "sex": None, "age_band": None}

    implausible = parse_header_demographics("A0003 12 500 5000\n#Age: 190\n#Sex: M\n")
    assert implausible == {"age": None, "sex": "male", "age_band": None}
