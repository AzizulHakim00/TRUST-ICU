from __future__ import annotations

import csv
from pathlib import Path

from trust_icu.ecg_crosswalk import resolve_ptbxl_checksum_crosswalk
from trust_icu.ecg_data import EXPECTED_LEADS, HeaderRecord


def _challenge(record_id: str, checksum_seed: int) -> HeaderRecord:
    return HeaderRecord(
        source="ptb-xl",
        record_id=record_id,
        sampling_rate_hz=500.0,
        sample_count=5000,
        lead_names=EXPECTED_LEADS,
        age=None,
        sex=None,
        dx_codes=("426783006",),
        signal_checksums=tuple(checksum_seed + index for index in range(12)),
    )


def _write_original_header(path: Path, *, checksum_seed: int, lead_names: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{path.stem} 12 500 5000"]
    for index, lead in enumerate(lead_names):
        checksum = checksum_seed + index
        lines.append(f"{path.stem}.dat 16 1000/mV 16 0 0 {checksum} 0 {lead}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ecg_id", "patient_id", "strat_fold", "filename_hr"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_checksum_crosswalk_ignores_numeric_rank_and_normalizes_only_augmented_case(
    tmp_path: Path,
) -> None:
    original_root = tmp_path / "original"
    metadata = tmp_path / "ptbxl_database.csv"
    original_leads = (
        "I",
        "II",
        "III",
        "AVR",
        "AVL",
        "AVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    )
    _write_original_header(
        original_root / "00000" / "00001_hr.hea",
        checksum_seed=100,
        lead_names=original_leads,
    )
    _write_original_header(
        original_root / "00000" / "00002_hr.hea",
        checksum_seed=200,
        lead_names=original_leads,
    )
    _write_metadata(
        metadata,
        [
            {
                "ecg_id": "2",
                "patient_id": "20",
                "strat_fold": "9",
                "filename_hr": "00000/00002_hr",
            },
            {
                "ecg_id": "1",
                "patient_id": "10",
                "strat_fold": "3",
                "filename_hr": "00000/00001_hr",
            },
        ],
    )

    # Numeric IDs are deliberately opposite the original ecg_id order.
    challenge = [_challenge("HR00001", 200), _challenge("HR00002", 100)]
    resolved, report = resolve_ptbxl_checksum_crosswalk(
        challenge_records=challenge,
        metadata_csv=metadata,
        original_ptbxl_root=original_root,
    )

    assert report["valid"] is True
    assert report["method"] == "all_12_wfdb_checksum_signature_join"
    assert report["verified_pairs"] == 2
    assert report["augmented_lead_case_normalization_records"] == 2
    assert report["lead_order_mismatches_after_canonicalization"] == 0
    by_challenge = {row.challenge_record_id: row for row in resolved}
    assert by_challenge["HR00001"].ecg_id == 2
    assert by_challenge["HR00001"].strat_fold == 9
    assert by_challenge["HR00002"].ecg_id == 1
    assert by_challenge["HR00002"].strat_fold == 3


def test_checksum_crosswalk_fails_closed_on_duplicate_original_signature(tmp_path: Path) -> None:
    original_root = tmp_path / "original"
    metadata = tmp_path / "ptbxl_database.csv"
    leads = (
        "I",
        "II",
        "III",
        "AVR",
        "AVL",
        "AVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    )
    _write_original_header(original_root / "a" / "one.hea", checksum_seed=100, lead_names=leads)
    _write_original_header(original_root / "b" / "two.hea", checksum_seed=100, lead_names=leads)
    _write_metadata(
        metadata,
        [
            {"ecg_id": "1", "patient_id": "1", "strat_fold": "1", "filename_hr": "a/one"},
            {"ecg_id": "2", "patient_id": "2", "strat_fold": "2", "filename_hr": "b/two"},
        ],
    )

    _, report = resolve_ptbxl_checksum_crosswalk(
        challenge_records=[_challenge("HR00001", 100), _challenge("HR00002", 200)],
        metadata_csv=metadata,
        original_ptbxl_root=original_root,
    )

    assert report["valid"] is False
    assert report["duplicate_original_checksum_signature_records"] == 2
    assert report["checksum_signature_ambiguous"] >= 1


def test_checksum_crosswalk_fails_closed_on_unmatched_signature(tmp_path: Path) -> None:
    original_root = tmp_path / "original"
    metadata = tmp_path / "ptbxl_database.csv"
    leads = (
        "I",
        "II",
        "III",
        "AVR",
        "AVL",
        "AVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    )
    _write_original_header(original_root / "a" / "one.hea", checksum_seed=100, lead_names=leads)
    _write_metadata(
        metadata,
        [{"ecg_id": "1", "patient_id": "1", "strat_fold": "1", "filename_hr": "a/one"}],
    )

    _, report = resolve_ptbxl_checksum_crosswalk(
        challenge_records=[_challenge("HR00001", 999)],
        metadata_csv=metadata,
        original_ptbxl_root=original_root,
    )

    assert report["valid"] is False
    assert report["checksum_signature_unmatched"] == 1
    assert report["verified_pairs"] == 0
    assert report["metadata_rows_unmatched"] == 1


def test_checksum_crosswalk_fails_closed_on_noncanonical_lead_order(tmp_path: Path) -> None:
    original_root = tmp_path / "original"
    metadata = tmp_path / "ptbxl_database.csv"
    wrong_order = (
        "II",
        "I",
        "III",
        "AVR",
        "AVL",
        "AVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    )
    _write_original_header(
        original_root / "a" / "one.hea",
        checksum_seed=100,
        lead_names=wrong_order,
    )
    _write_metadata(
        metadata,
        [{"ecg_id": "1", "patient_id": "1", "strat_fold": "1", "filename_hr": "a/one"}],
    )

    _, report = resolve_ptbxl_checksum_crosswalk(
        challenge_records=[_challenge("HR00001", 100)],
        metadata_csv=metadata,
        original_ptbxl_root=original_root,
    )

    assert report["valid"] is False
    assert report["lead_order_mismatches_after_canonicalization"] == 1
    assert report["verified_pairs"] == 0
