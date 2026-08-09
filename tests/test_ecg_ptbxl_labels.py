from __future__ import annotations

import csv
from pathlib import Path

import pytest

from trust_icu import ecg_ptbxl_labels
from trust_icu.ecg_ptbxl_labels import build_ptbxl_label_concordance_audit

SYNTHETIC_MAPPING = {
    "1": {"abbreviation": "A", "scp_codes": ("AAA", "AUX"), "challenge_positive_count": 3},
    "2": {"abbreviation": "B", "scp_codes": ("BBB",), "challenge_positive_count": 2},
}


def _write_metadata(path: Path, *, leak_patient: bool = False, drop_union_member: bool = False) -> Path:
    rows = []
    for index in range(10):
        codes = {}
        if index < 2:
            codes["AAA"] = 0.0
        if index == 2 and not drop_union_member:
            codes["AUX"] = 0.0
        if index < 2:
            codes["BBB"] = 0.0
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


def test_union_key_presence_label_concordance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ecg_ptbxl_labels, "PTBXL_SCP_TO_CHALLENGE", SYNTHETIC_MAPPING)
    audit = build_ptbxl_label_concordance_audit(_write_metadata(tmp_path / "ptb.csv"))
    # Production release-size invariant remains active for the 10-row fixture; label semantics are
    # nevertheless exact and demonstrate that likelihood==0 still counts by key presence.
    assert audit.all_labels_exactly_concordant is True
    assert audit.selected_count_semantics == "union_of_scp_key_presence_per_record"
    assert audit.patients_spanning_multiple_folds == 0
    assert audit.label_rows[0].scp_codes == ("AAA", "AUX")
    assert audit.label_rows[0].original_ptbxl_union_key_present_count == 3


def test_one_union_member_count_mismatch_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ecg_ptbxl_labels, "PTBXL_SCP_TO_CHALLENGE", SYNTHETIC_MAPPING)
    audit = build_ptbxl_label_concordance_audit(
        _write_metadata(tmp_path / "ptb.csv", drop_union_member=True)
    )
    assert audit.all_labels_exactly_concordant is False
    assert "ptbxl_scp_union_to_challenge_label_counts_not_exactly_concordant" in audit.blockers


def test_patient_spanning_folds_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ecg_ptbxl_labels, "PTBXL_SCP_TO_CHALLENGE", SYNTHETIC_MAPPING)
    audit = build_ptbxl_label_concordance_audit(
        _write_metadata(tmp_path / "ptb.csv", leak_patient=True)
    )
    assert audit.patients_spanning_multiple_folds == 1
    assert "ptbxl_patient_fold_leakage_detected" in audit.blockers
