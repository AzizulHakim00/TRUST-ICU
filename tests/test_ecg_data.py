from __future__ import annotations

import csv
from pathlib import Path

from trust_icu import ecg_data
from trust_icu.ecg_data import HeaderRecord, build_header_audit, parse_challenge_header

ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "schemas/challenge2020_scored_classes.csv"
LEADS = ecg_data.EXPECTED_LEADS


def _record(
    source: str,
    record_id: str,
    codes: tuple[str, ...],
    *,
    checksum_base: int = 0,
) -> HeaderRecord:
    return HeaderRecord(
        source=source,
        record_id=record_id,
        sampling_rate_hz=500.0,
        sample_count=5000,
        lead_names=LEADS,
        age="50",
        sex="Male",
        dx_codes=codes,
        signal_checksums=tuple(checksum_base + index for index in range(12)),
    )


def _metadata(path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ecg_id", "patient_id", "strat_fold", "filename_hr"],
        )
        writer.writeheader()
        for fold in range(1, 11):
            writer.writerow(
                {
                    "ecg_id": fold,
                    "patient_id": f"p{fold}",
                    "strat_fold": fold,
                    "filename_hr": f"records500/00000/{fold:05d}_hr",
                }
            )
    return path


def _write_original_header(root: Path, ecg_id: int, *, checksum_base: int) -> None:
    path = root / "records500" / "00000" / f"{ecg_id:05d}_hr.hea"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{ecg_id:05d}_hr 12 500 5000"]
    for index, lead in enumerate(LEADS):
        checksum = checksum_base + index
        lines.append(f"{ecg_id:05d}_hr.dat 16 1000/mV 16 0 0 {checksum} 0 {lead}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_challenge_header_reads_signal_contract_and_dx(tmp_path: Path) -> None:
    signal_lines = "\n".join(
        f"A0001.mat 16+24 1000/mV 16 0 0 {index} 0 {lead}"
        for index, lead in enumerate(LEADS)
    )
    header = tmp_path / "A0001.hea"
    header.write_text(
        f"A0001 12 500 5000\n{signal_lines}\n#Age: 74\n#Sex: Male\n#Dx: 426783006,164889003\n",
        encoding="utf-8",
    )
    parsed = parse_challenge_header(header, source="georgia")
    assert parsed.record_id == "A0001"
    assert parsed.sampling_rate_hz == 500
    assert parsed.duration_seconds == 10
    assert parsed.lead_names == LEADS
    assert parsed.dx_codes == ("164889003", "426783006")
    assert parsed.signal_checksums == tuple(range(12))


def test_equivalent_scored_codes_collapse_to_one_label_group(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        ecg_data,
        "EXPECTED_SOURCES",
        {"ptb-xl": 2, "georgia": 2, "cpsc_2018": 2, "cpsc_2018_extra": 2},
    )
    records = [
        _record("ptb-xl", "HR00000", ("284470004",)),
        _record("ptb-xl", "HR00001", ("63593006",)),
        _record("georgia", "g1", ("284470004",)),
        _record("georgia", "g2", ("63593006",)),
        _record("cpsc_2018", "c1", ("284470004",)),
        _record("cpsc_2018", "c2", ("63593006",)),
        _record("cpsc_2018_extra", "x1", ("426783006",)),
        _record("cpsc_2018_extra", "x2", ("426783006",)),
    ]
    audit = build_header_audit(
        records=records,
        scored_mapping_path=MAPPING,
        ptbxl_metadata_csv=_metadata(tmp_path / "ptbxl.csv"),
        require_ptbxl_crosswalk=False,
        minimum_development_positives=2,
        minimum_external_positives=2,
        minimum_external_domains=2,
    )
    pac = next(label for label in audit.labels if label.canonical_code == "284470004")
    assert pac.member_codes == ("63593006", "284470004")
    assert pac.development_positives == 2
    assert pac.external_positives["georgia"] == 2
    assert pac.external_positives["cpsc_2018"] == 2
    assert pac.eligible is True


def test_patient_spanning_ptbxl_folds_blocks_waveform_stage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        ecg_data,
        "EXPECTED_SOURCES",
        {"ptb-xl": 1, "georgia": 1, "cpsc_2018": 1, "cpsc_2018_extra": 1},
    )
    records = [
        _record("ptb-xl", "HR00000", ("426783006",)),
        _record("georgia", "g1", ("426783006",)),
        _record("cpsc_2018", "c1", ("426783006",)),
        _record("cpsc_2018_extra", "x1", ("426783006",)),
    ]
    metadata = tmp_path / "ptbxl.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ecg_id", "patient_id", "strat_fold", "filename_hr"],
        )
        writer.writeheader()
        for fold in range(1, 11):
            writer.writerow(
                {
                    "ecg_id": fold,
                    "patient_id": "same" if fold < 3 else f"p{fold}",
                    "strat_fold": fold,
                    "filename_hr": f"records500/00000/{fold:05d}_hr",
                }
            )
    audit = build_header_audit(
        records=records,
        scored_mapping_path=MAPPING,
        ptbxl_metadata_csv=metadata,
        require_ptbxl_crosswalk=False,
        minimum_development_positives=1,
        minimum_external_positives=1,
        minimum_external_domains=2,
    )
    assert audit.ptbxl_fold_integrity["valid"] is False
    assert "ptbxl_patientwise_fold_integrity_not_verified" in audit.blockers
    assert audit.ready_for_waveform_stage is False


def test_complete_synthetic_header_audit_requires_checksum_verified_crosswalk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ecg_data,
        "EXPECTED_SOURCES",
        {"ptb-xl": 10, "georgia": 1, "cpsc_2018": 1, "cpsc_2018_extra": 1},
    )
    ptb_records = [
        _record("ptb-xl", f"HR{index:05d}", ("426783006",), checksum_base=index * 100)
        for index in range(10)
    ]
    records = ptb_records + [
        _record("georgia", "g1", ("426783006",)),
        _record("cpsc_2018", "c1", ("426783006",)),
        _record("cpsc_2018_extra", "x1", ("426783006",)),
    ]
    metadata = _metadata(tmp_path / "ptbxl.csv")
    original_root = tmp_path / "ptbxl"
    for ecg_id in range(1, 11):
        _write_original_header(original_root, ecg_id, checksum_base=(ecg_id - 1) * 100)

    audit = build_header_audit(
        records=records,
        scored_mapping_path=MAPPING,
        ptbxl_metadata_csv=metadata,
        ptbxl_original_root=original_root,
        minimum_development_positives=1,
        minimum_external_positives=1,
        minimum_external_domains=2,
    )
    assert audit.ptbxl_crosswalk["valid"] is True
    assert audit.ptbxl_crosswalk["verified_pairs"] == 10
    assert audit.ready_for_waveform_stage is True
    assert "426783006" in audit.eligible_labels
    assert len(audit.manifest_sha256) == 64


def test_checksum_crosswalk_mismatch_blocks_waveform_stage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        ecg_data,
        "EXPECTED_SOURCES",
        {"ptb-xl": 10, "georgia": 1, "cpsc_2018": 1, "cpsc_2018_extra": 1},
    )
    records = [
        _record("ptb-xl", f"HR{index:05d}", ("426783006",), checksum_base=index * 100)
        for index in range(10)
    ] + [
        _record("georgia", "g1", ("426783006",)),
        _record("cpsc_2018", "c1", ("426783006",)),
        _record("cpsc_2018_extra", "x1", ("426783006",)),
    ]
    metadata = _metadata(tmp_path / "ptbxl.csv")
    original_root = tmp_path / "ptbxl"
    for ecg_id in range(1, 11):
        checksum_base = 9999 if ecg_id == 5 else (ecg_id - 1) * 100
        _write_original_header(original_root, ecg_id, checksum_base=checksum_base)

    audit = build_header_audit(
        records=records,
        scored_mapping_path=MAPPING,
        ptbxl_metadata_csv=metadata,
        ptbxl_original_root=original_root,
        minimum_development_positives=1,
        minimum_external_positives=1,
        minimum_external_domains=2,
    )
    assert audit.ptbxl_crosswalk["valid"] is False
    assert audit.ptbxl_crosswalk["checksum_mismatches"] == 1
    assert "challenge_ptbxl_crosswalk_not_verified" in audit.blockers
