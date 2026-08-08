from __future__ import annotations

import csv
from pathlib import Path

from trust_icu import ecg_data
from trust_icu.ecg_data import HeaderRecord, build_header_audit, parse_challenge_header

ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "schemas/challenge2020_scored_classes.csv"
LEADS = ecg_data.EXPECTED_LEADS


def _record(source: str, record_id: str, codes: tuple[str, ...]) -> HeaderRecord:
    return HeaderRecord(
        source=source,
        record_id=record_id,
        sampling_rate_hz=500.0,
        sample_count=5000,
        lead_names=LEADS,
        age="50",
        sex="Male",
        dx_codes=codes,
    )


def _metadata(path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "strat_fold"])
        writer.writeheader()
        for fold in range(1, 11):
            writer.writerow({"patient_id": f"p{fold}", "strat_fold": fold})
    return path


def test_parse_challenge_header_reads_signal_contract_and_dx(tmp_path: Path) -> None:
    signal_lines = "\n".join(
        f"A0001.mat 16+24 1000/mV 16 0 0 0 0 {lead}" for lead in LEADS
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


def test_equivalent_scored_codes_collapse_to_one_label_group(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        ecg_data,
        "EXPECTED_SOURCES",
        {"ptb-xl": 2, "georgia": 2, "cpsc_2018": 2, "cpsc_2018_extra": 2},
    )
    records = [
        _record("ptb-xl", "p1", ("284470004",)),
        _record("ptb-xl", "p2", ("63593006",)),
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
        _record("ptb-xl", "p1", ("426783006",)),
        _record("georgia", "g1", ("426783006",)),
        _record("cpsc_2018", "c1", ("426783006",)),
        _record("cpsc_2018_extra", "x1", ("426783006",)),
    ]
    metadata = tmp_path / "ptbxl.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "strat_fold"])
        writer.writeheader()
        for fold in range(1, 11):
            writer.writerow({"patient_id": "same" if fold < 3 else f"p{fold}", "strat_fold": fold})
    audit = build_header_audit(
        records=records,
        scored_mapping_path=MAPPING,
        ptbxl_metadata_csv=metadata,
        minimum_development_positives=1,
        minimum_external_positives=1,
        minimum_external_domains=2,
    )
    assert audit.ptbxl_fold_integrity["valid"] is False
    assert "ptbxl_patientwise_fold_integrity_not_verified" in audit.blockers
    assert audit.ready_for_waveform_stage is False


def test_complete_synthetic_header_audit_can_pass(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        ecg_data,
        "EXPECTED_SOURCES",
        {"ptb-xl": 1, "georgia": 1, "cpsc_2018": 1, "cpsc_2018_extra": 1},
    )
    records = [
        _record("ptb-xl", "p1", ("426783006",)),
        _record("georgia", "g1", ("426783006",)),
        _record("cpsc_2018", "c1", ("426783006",)),
        _record("cpsc_2018_extra", "x1", ("426783006",)),
    ]
    audit = build_header_audit(
        records=records,
        scored_mapping_path=MAPPING,
        ptbxl_metadata_csv=_metadata(tmp_path / "ptbxl.csv"),
        minimum_development_positives=1,
        minimum_external_positives=1,
        minimum_external_domains=2,
    )
    assert audit.ready_for_waveform_stage is True
    assert "426783006" in audit.eligible_labels
    assert len(audit.manifest_sha256) == 64
