from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from trust_icu import ecg_waveform
from trust_icu.ecg_data import EXPECTED_LEADS
from trust_icu.ecg_manifest import build_label_manifest, write_label_manifest
from trust_icu.ecg_waveform import load_and_verify_waveform_audit, prepare_waveform_stage

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "schemas/open_ecg_protocol.yaml"
MAPPING = ROOT / "schemas/challenge2020_scored_classes.csv"


def _canonical_hash(payload: dict, key: str) -> str:
    material = copy.deepcopy(payload)
    material[key] = ""
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _signal_lines(stem: str, checksums: tuple[int, ...]) -> list[str]:
    return [
        f"{stem}.mat 16+24 1000/mV 16 0 0 {checksum} 0 {lead}"
        for lead, checksum in zip(EXPECTED_LEADS, checksums, strict=True)
    ]


def _write_challenge_record(
    source_root: Path,
    record_id: str,
    *,
    sampling_rate: int,
    samples: int,
    value_offset: float,
    checksums: tuple[int, ...],
) -> None:
    source_root.mkdir(parents=True, exist_ok=True)
    header = source_root / f"{record_id}.hea"
    mat = source_root / f"{record_id}.mat"
    lines = [f"{record_id} 12 {sampling_rate} {samples}"] + _signal_lines(record_id, checksums)
    lines += ["#Age: 50", "#Sex: Male", "#Dx: 426783006"]
    header.write_text("\n".join(lines) + "\n", encoding="utf-8")
    time = np.linspace(0.0, 1.0, samples, endpoint=False)
    physical_mv = np.stack(
        [value_offset + lead_index * 0.01 + time for lead_index in range(12)],
        axis=0,
    )
    digital = np.rint(physical_mv * 1000.0).astype(np.int32)
    savemat(mat, {"val": digital})


def _write_original_header(
    root: Path,
    ecg_id: int,
    checksums: tuple[int, ...],
) -> str:
    relative_stem = f"records500/00000/{ecg_id:05d}_hr"
    path = root / f"{relative_stem}.hea"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{ecg_id:05d}_hr 12 500 5000"] + [
        f"{ecg_id:05d}_hr.dat 16 1000/mV 16 0 0 {checksum} 0 {lead}"
        for lead, checksum in zip(EXPECTED_LEADS, checksums, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return relative_stem


def _write_metadata(path: Path, original_root: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ecg_id", "patient_id", "strat_fold", "filename_hr"],
        )
        writer.writeheader()
        for ecg_id in range(1, 11):
            checksums = tuple(ecg_id * 100 + index for index in range(12))
            filename_hr = _write_original_header(original_root, ecg_id, checksums)
            writer.writerow(
                {
                    "ecg_id": ecg_id,
                    "patient_id": f"patient_{ecg_id}",
                    "strat_fold": ecg_id,
                    "filename_hr": filename_hr,
                }
            )
    return path


def _write_header_audit(path: Path) -> Path:
    payload = {
        "protocol_version": "0.2.0",
        "ready_for_waveform_stage": True,
        "ptbxl_crosswalk": {"valid": True},
        "eligible_labels": ["426783006"],
        "labels": [
            {
                "canonical_code": "426783006",
                "abbreviation": "NSR",
                "member_codes": ["426783006"],
                "development_positives": 1000,
                "external_positives": {
                    "georgia": 200,
                    "cpsc_2018": 200,
                    "cpsc_2018_extra": 200,
                },
                "external_domains_meeting_threshold": 3,
                "eligible": True,
            }
        ],
        "manifest_sha256": "",
    }
    payload["manifest_sha256"] = _canonical_hash(payload, "manifest_sha256")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _build_fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path, Path, Path]:
    expected = {
        "ptb-xl": 10,
        "georgia": 1,
        "cpsc_2018": 1,
        "cpsc_2018_extra": 1,
    }
    monkeypatch.setattr(ecg_waveform, "EXPECTED_SOURCES", expected)
    training = tmp_path / "training"
    original = tmp_path / "ptbxl_original"
    metadata = _write_metadata(tmp_path / "ptbxl_database.csv", original)

    for index in range(10):
        ecg_id = index + 1
        checksums = tuple(ecg_id * 100 + lead for lead in range(12))
        _write_challenge_record(
            training / "ptb-xl",
            f"HR{index:05d}",
            sampling_rate=500,
            samples=5000,
            value_offset=float(index),
            checksums=checksums,
        )
    _write_challenge_record(
        training / "georgia",
        "G00001",
        sampling_rate=500,
        samples=5000,
        value_offset=20.0,
        checksums=tuple(2000 + lead for lead in range(12)),
    )
    _write_challenge_record(
        training / "cpsc_2018",
        "A00001",
        sampling_rate=500,
        samples=6000,
        value_offset=30.0,
        checksums=tuple(3000 + lead for lead in range(12)),
    )
    _write_challenge_record(
        training / "cpsc_2018_extra",
        "Q00001",
        sampling_rate=250,
        samples=2000,
        value_offset=40.0,
        checksums=tuple(4000 + lead for lead in range(12)),
    )

    header_audit = _write_header_audit(tmp_path / "header_audit.json")
    label_payload = build_label_manifest(
        header_audit_path=header_audit,
        protocol_path=PROTOCOL,
        scored_mapping_path=MAPPING,
    )
    label_manifest = tmp_path / "labels.json"
    write_label_manifest(label_payload, label_manifest)
    return training, metadata, original, header_audit, label_manifest


def test_waveform_stage_hashes_data_and_fits_training_only_stats(tmp_path: Path, monkeypatch) -> None:
    training, metadata, original, header_audit, label_manifest = _build_fixture(
        tmp_path,
        monkeypatch,
    )
    output = tmp_path / "outputs"
    audit = prepare_waveform_stage(
        challenge_training_root=training,
        ptbxl_metadata_csv=metadata,
        ptbxl_original_root=original,
        header_audit_path=header_audit,
        label_manifest_path=label_manifest,
        output_root=output,
    )
    assert audit.ready_for_model_stage is True
    assert audit.blockers == ()
    assert audit.normalization_fit_folds == (1, 2, 3, 4, 5, 6, 7)
    assert audit.source_resampled_records["cpsc_2018_extra"] == 1
    assert audit.source_cropped_records["cpsc_2018"] == 1
    assert audit.source_padded_records["cpsc_2018_extra"] == 1
    assert all(len(value) == 64 for value in audit.source_header_corpus_sha256.values())
    assert all(len(value) == 64 for value in audit.source_waveform_corpus_sha256.values())
    assert len(audit.normalization_stats_sha256) == 64
    assert len(audit.ptbxl_assignment_sha256) == 64
    assignment = (output / "ptbxl_verified_assignment.csv").read_text(encoding="utf-8")
    assert "patient_id" not in assignment
    verified = load_and_verify_waveform_audit(output / "open_ecg_waveform_audit.json")
    assert verified["audit_sha256"] == audit.audit_sha256


def test_waveform_mutation_changes_corpus_hash(tmp_path: Path, monkeypatch) -> None:
    training, metadata, original, header_audit, label_manifest = _build_fixture(
        tmp_path,
        monkeypatch,
    )
    first = prepare_waveform_stage(
        challenge_training_root=training,
        ptbxl_metadata_csv=metadata,
        ptbxl_original_root=original,
        header_audit_path=header_audit,
        label_manifest_path=label_manifest,
        output_root=tmp_path / "first",
    )
    mat_path = training / "georgia" / "G00001.mat"
    payload = loadmat(mat_path)
    payload["val"][0, 0] += 1
    savemat(mat_path, {"val": payload["val"]})
    second = prepare_waveform_stage(
        challenge_training_root=training,
        ptbxl_metadata_csv=metadata,
        ptbxl_original_root=original,
        header_audit_path=header_audit,
        label_manifest_path=label_manifest,
        output_root=tmp_path / "second",
    )
    assert first.source_waveform_corpus_sha256["georgia"] != second.source_waveform_corpus_sha256["georgia"]


def test_missing_waveform_blocks_instead_of_silently_dropping_record(tmp_path: Path, monkeypatch) -> None:
    training, metadata, original, header_audit, label_manifest = _build_fixture(
        tmp_path,
        monkeypatch,
    )
    (training / "georgia" / "G00001.mat").unlink()
    audit = prepare_waveform_stage(
        challenge_training_root=training,
        ptbxl_metadata_csv=metadata,
        ptbxl_original_root=original,
        header_audit_path=header_audit,
        label_manifest_path=label_manifest,
        output_root=tmp_path / "outputs",
    )
    assert audit.ready_for_model_stage is False
    assert audit.source_invalid_records["georgia"] == 1
    assert "invalid_or_missing_waveform_records" in audit.blockers
