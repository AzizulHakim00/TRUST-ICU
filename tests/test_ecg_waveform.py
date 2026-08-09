from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from trust_icu import ecg_waveform
from trust_icu.ecg_data import EXPECTED_LEADS
from trust_icu.ecg_manifest import build_label_manifest, write_label_manifest
from trust_icu.ecg_ptbxl_labels import PTBXL_SCP_TO_CHALLENGE
from trust_icu.ecg_waveform import load_and_verify_waveform_audit, prepare_waveform_stage

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "schemas/open_ecg_protocol.yaml"
MAPPING = ROOT / "schemas/challenge2020_scored_classes.csv"


def _canonical_hash(payload: dict, key: str) -> str:
    material = copy.deepcopy(payload)
    material[key] = ""
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _signal_lines(stem: str, checksums: tuple[int, ...], *, suffix: str, uppercase_augmented: bool = False) -> list[str]:
    leads = list(EXPECTED_LEADS)
    if uppercase_augmented:
        leads[3:6] = ["AVR", "AVL", "AVF"]
    return [
        f"{stem}.{suffix} 16 1000/mV 16 0 0 {checksum} 0 {lead}"
        for lead, checksum in zip(leads, checksums, strict=True)
    ]


def _write_external_record(
    source_root: Path,
    record_id: str,
    *,
    sampling_rate: int,
    samples: int,
    value_offset: float,
) -> None:
    source_root.mkdir(parents=True, exist_ok=True)
    checksums = tuple(1000 + index for index in range(12))
    lines = [f"{record_id} 12 {sampling_rate} {samples}"] + _signal_lines(
        record_id, checksums, suffix="mat"
    )
    lines += ["#Age: 50", "#Sex: Male", "#Dx: 426783006"]
    (source_root / f"{record_id}.hea").write_text("\n".join(lines) + "\n", encoding="utf-8")
    time = np.linspace(0.0, 1.0, samples, endpoint=False)
    physical_mv = np.stack(
        [value_offset + lead_index * 0.01 + time for lead_index in range(12)], axis=0
    )
    savemat(source_root / f"{record_id}.mat", {"val": np.rint(physical_mv * 1000).astype(np.int32)})


def _write_original_ptb_record(root: Path, ecg_id: int) -> str:
    relative_stem = f"records500/00000/{ecg_id:05d}_hr"
    stem_path = root / relative_stem
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    checksums = tuple(ecg_id * 100 + index for index in range(12))
    lines = [f"{ecg_id:05d}_hr 12 500 5000"] + _signal_lines(
        f"{ecg_id:05d}_hr", checksums, suffix="dat", uppercase_augmented=True
    )
    stem_path.with_suffix(".hea").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stem_path.with_suffix(".dat").write_bytes(f"synthetic-{ecg_id}".encode())
    return relative_stem


def _write_metadata(path: Path, ptb_root: Path, count: int) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ecg_id", "patient_id", "strat_fold", "filename_hr", "scp_codes"],
        )
        writer.writeheader()
        for index in range(count):
            ecg_id = index + 1
            writer.writerow(
                {
                    "ecg_id": ecg_id,
                    "patient_id": f"patient_{ecg_id}",
                    "strat_fold": (index % 10) + 1,
                    "filename_hr": _write_original_ptb_record(ptb_root, ecg_id),
                    "scp_codes": "{'SR': 0.0, 'NORM': 0.0}",
                }
            )
    return path


def _write_header_audit(path: Path) -> Path:
    labels = []
    for code, spec in PTBXL_SCP_TO_CHALLENGE.items():
        labels.append(
            {
                "canonical_code": code,
                "abbreviation": spec["abbreviation"],
                "member_codes": [code],
                "development_positives": spec["challenge_positive_count"],
                "external_positives": {
                    "georgia": 200,
                    "cpsc_2018": 200,
                    "cpsc_2018_extra": 200,
                },
                "external_domains_meeting_threshold": 3,
                "eligible": True,
            }
        )
    payload = {
        "protocol_version": "0.4.0",
        "ready_for_waveform_stage": True,
        "ptbxl_crosswalk": {"required": False, "valid": True},
        "eligible_labels": list(PTBXL_SCP_TO_CHALLENGE),
        "labels": labels,
        "manifest_sha256": "",
    }
    payload["manifest_sha256"] = _canonical_hash(payload, "manifest_sha256")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_concordance(path: Path) -> Path:
    rows = []
    for code, spec in PTBXL_SCP_TO_CHALLENGE.items():
        rows.append(
            {
                "canonical_code": code,
                "abbreviation": spec["abbreviation"],
                "scp_codes": list(spec["scp_codes"]),
                "challenge_positive_count": spec["challenge_positive_count"],
                "original_ptbxl_union_key_present_count": spec["challenge_positive_count"],
                "exact_union_key_present_match": True,
            }
        )
    payload = {
        "audit_version": "0.2.0",
        "ptbxl_rows": 21837,
        "unique_ecg_ids": 21837,
        "unique_patients": 18885,
        "folds_present": list(range(1, 11)),
        "patients_spanning_multiple_folds": 0,
        "label_rows": rows,
        "selected_count_semantics": "union_of_scp_key_presence_per_record",
        "all_labels_exactly_concordant": True,
        "ready_for_original_ptbxl_development": True,
        "blockers": [],
        "audit_sha256": "",
    }
    payload["audit_sha256"] = _canonical_hash(payload, "audit_sha256")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _mock_wfdb_loader(record_stem: str | Path):
    stem = Path(record_stem)
    ecg_id = int(stem.name.split("_", 1)[0])
    time = np.linspace(0.0, 1.0, 5000, endpoint=False)
    signal = np.stack(
        [ecg_id + lead_index * 0.01 + time for lead_index in range(12)], axis=0
    )
    leads = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
    return signal, leads, 500.0


def _build_fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path, Path]:
    expected = {"ptb-xl": 10, "georgia": 1, "cpsc_2018": 1, "cpsc_2018_extra": 1}
    monkeypatch.setattr(ecg_waveform, "EXPECTED_SOURCES", expected)
    monkeypatch.setattr(ecg_waveform, "load_wfdb_physical_signal", _mock_wfdb_loader)
    data_root = tmp_path / "primary"
    ptb_root = data_root / "ptb-xl"
    metadata = _write_metadata(ptb_root / "ptbxl_database.csv", ptb_root, 10)
    _write_external_record(data_root / "georgia", "G00001", sampling_rate=500, samples=5000, value_offset=20.0)
    _write_external_record(data_root / "cpsc_2018", "A00001", sampling_rate=500, samples=6000, value_offset=30.0)
    _write_external_record(data_root / "cpsc_2018_extra", "Q00001", sampling_rate=250, samples=2000, value_offset=40.0)

    header_audit = _write_header_audit(tmp_path / "header_audit.json")
    concordance = _write_concordance(tmp_path / "concordance.json")
    manifest = build_label_manifest(
        header_audit_path=header_audit,
        ptbxl_label_concordance_path=concordance,
        protocol_path=PROTOCOL,
        scored_mapping_path=MAPPING,
    )
    label_manifest = tmp_path / "labels.json"
    write_label_manifest(manifest, label_manifest)
    return data_root, metadata, header_audit, label_manifest


def test_direct_ptbxl_waveform_stage_is_ready(tmp_path: Path, monkeypatch) -> None:
    data_root, metadata, header_audit, labels = _build_fixture(tmp_path, monkeypatch)
    output = tmp_path / "out"
    audit = prepare_waveform_stage(
        primary_data_root=data_root,
        ptbxl_metadata_csv=metadata,
        header_audit_path=header_audit,
        label_manifest_path=labels,
        output_root=output,
    )
    assert audit.ready_for_model_stage is True
    assert audit.development_source == "original_ptbxl_v1_0_1"
    assert audit.challenge_ptbxl_model_input is False
    assert audit.source_record_counts == {
        "ptb-xl": 10,
        "georgia": 1,
        "cpsc_2018": 1,
        "cpsc_2018_extra": 1,
    }
    assert audit.source_waveform_format["ptb-xl"] == "wfdb_dat_original_ptbxl_v1_0_1"
    assert audit.normalization_fit_folds == (1, 2, 3, 4, 5, 6, 7)
    assert audit.normalization_stats_sha256
    assert (output / "ptbxl_verified_assignment.csv").is_file()
    assert (output / "open_ecg_normalization_stats.json").is_file()
    loaded = load_and_verify_waveform_audit(output / "open_ecg_waveform_audit.json")
    assert loaded["audit_sha256"] == audit.audit_sha256


def test_missing_external_waveform_blocks(tmp_path: Path, monkeypatch) -> None:
    data_root, metadata, header_audit, labels = _build_fixture(tmp_path, monkeypatch)
    (data_root / "georgia" / "G00001.mat").unlink()
    audit = prepare_waveform_stage(
        primary_data_root=data_root,
        ptbxl_metadata_csv=metadata,
        header_audit_path=header_audit,
        label_manifest_path=labels,
        output_root=tmp_path / "out",
    )
    assert audit.ready_for_model_stage is False
    assert "invalid_or_missing_waveform_records" in audit.blockers


def test_waveform_audit_tamper_is_rejected(tmp_path: Path, monkeypatch) -> None:
    data_root, metadata, header_audit, labels = _build_fixture(tmp_path, monkeypatch)
    output = tmp_path / "out"
    prepare_waveform_stage(
        primary_data_root=data_root,
        ptbxl_metadata_csv=metadata,
        header_audit_path=header_audit,
        label_manifest_path=labels,
        output_root=output,
    )
    audit_path = output / "open_ecg_waveform_audit.json"
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["source_record_counts"]["ptb-xl"] += 1
    audit_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_and_verify_waveform_audit(audit_path)
    except ValueError as exc:
        assert "SHA-256 verification failed" in str(exc)
    else:
        raise AssertionError("Tampered waveform audit was accepted")
