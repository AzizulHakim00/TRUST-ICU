from __future__ import annotations

import csv
from pathlib import Path

import pytest

from trust_icu import ecg_ptbxl_labels
from trust_icu.ecg_ptbxl_labels import build_ptbxl_label_concordance_audit


def _write_metadata(path: Path, *, leak_patient: bool = False, drop_last_af: bool = False) -> Path:
    # Small synthetic target mapping replaces the real fixed counts only inside this unit test.
    mapping = {
        "1": {"abbreviation": "A", "scp_code": "AAA", "challenge_positive_count": 3},
        "2": {"abbreviation": "B", "scp_code": "BBB", "challenge_positive_count": 2},
    }
    ecg_ptbxl_labels.PTBXL_SCP_TO_CHALLENGE = mapping
    rows = []
    for index in range(10):
        codes = {}
        if index < 3 and not (drop_last_af and index == 2):
            codes["AAA"] = 100.0
        if index < 2:
            codes["BBB"] = 50.0
        patient = "shared" if leak_patient and index in {0, 1} else f"p{index}"
        rows.append(
            {
                "ecg_id": index + 1,
                "patient_id": patient,
                "strat_fold": index + 1,
                "scp_codes": repr(codes),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ecg_id", "patient_id", "strat_fold", "scp_codes"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_exact_label_concordance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ecg_ptbxl_labels, "PTBXL_SCP_TO_CHALLENGE", {
        "1": {"abbreviation": "A", "scp_code": "AAA", "challenge_positive_count": 3},
        "2": {"abbreviation": "B", "scp_code": "BBB", "challenge_positive_count": 2},
    })
    monkeypatch.setattr(ecg_ptbxl_labels, "_EXPECTED_PTBXL_ROWS", 10, raising=False)
    path = _write_metadata(tmp_path / "ptb.csv")
    audit = build_ptbxl_label_concordance_audit(path)
    # Production row-count invariant intentionally remains active, so the synthetic fixture is
    # concordant but not ready. This separates label semantics from release-size validation.
    assert audit.all_labels_exactly_concordant is True
    assert audit.selected_count_semantics == "scp_key_present"
    assert audit.patients_spanning_multiple_folds == 0


def test_one_label_count_mismatch_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ecg_ptbxl_labels, "PTBXL_SCP_TO_CHALLENGE", {
        "1": {"abbreviation": "A", "scp_code": "AAA", "challenge_positive_count": 3},
        "2": {"abbreviation": "B", "scp_code": "BBB", "challenge_positive_count": 2},
    })
    path = _write_metadata(tmp_path / "ptb.csv", drop_last_af=True)
    audit = build_ptbxl_label_concordance_audit(path)
    assert audit.all_labels_exactly_concordant is False
    assert "ptbxl_scp_to_challenge_label_counts_not_exactly_concordant" in audit.blockers


def test_patient_spanning_folds_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ecg_ptbxl_labels, "PTBXL_SCP_TO_CHALLENGE", {
        "1": {"abbreviation": "A", "scp_code": "AAA", "challenge_positive_count": 3},
        "2": {"abbreviation": "B", "scp_code": "BBB", "challenge_positive_count": 2},
    })
    path = _write_metadata(tmp_path / "ptb.csv", leak_patient=True)
    audit = build_ptbxl_label_concordance_audit(path)
    assert audit.patients_spanning_multiple_folds == 1
    assert "ptbxl_patient_fold_leakage_detected" in audit.blockers
