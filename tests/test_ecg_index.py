from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from trust_icu import ecg_index
from trust_icu.ecg_data import EXPECTED_LEADS
from trust_icu.ecg_index import build_model_index, load_and_verify_index_audit, write_model_index
from trust_icu.ecg_waveform import PtbxlAssignment, assignment_sha256


def _canonical_hash(payload: dict, key: str) -> str:
    material = dict(payload)
    material[key] = ""
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_external_record(source_root: Path, record_id: str, dx: str = "426783006") -> None:
    source_root.mkdir(parents=True, exist_ok=True)
    lines = [f"{record_id} 12 500 5000"]
    lines.extend(
        f"{record_id}.mat 16+24 1000/mV 16 0 0 {index + 1} 0 {lead}"
        for index, lead in enumerate(EXPECTED_LEADS)
    )
    lines.extend(["#Age: 50", "#Sex: Male", f"#Dx: {dx}"])
    (source_root / f"{record_id}.hea").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (source_root / f"{record_id}.mat").write_bytes((record_id + "-waveform").encode())


def _write_ptb_record(root: Path, ecg_id: int) -> PtbxlAssignment:
    stem = f"{ecg_id:05d}_hr"
    relative_base = Path("ptb-xl") / "records500" / "00000" / stem
    header = root / relative_base.with_suffix(".hea")
    waveform = root / relative_base.with_suffix(".dat")
    header.parent.mkdir(parents=True, exist_ok=True)
    leads = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
    lines = [f"{stem} 12 500 5000"]
    lines.extend(
        f"{stem}.dat 16 1000/mV 16 0 0 {ecg_id * 100 + index} 0 {lead}"
        for index, lead in enumerate(leads)
    )
    header.write_text("\n".join(lines) + "\n", encoding="utf-8")
    waveform.write_bytes(f"ptb-{ecg_id}-waveform".encode())
    folds = [1, 8, 9, 10]
    return PtbxlAssignment(
        record_id=stem,
        ecg_id=ecg_id,
        strat_fold=folds[ecg_id - 1],
        relative_header_path=relative_base.with_suffix(".hea").as_posix(),
        relative_waveform_path=relative_base.with_suffix(".dat").as_posix(),
    )


def _write_assignment(path: Path, rows: tuple[PtbxlAssignment, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_id",
                "ecg_id",
                "strat_fold",
                "relative_header_path",
                "relative_waveform_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "record_id": row.record_id,
                    "ecg_id": row.ecg_id,
                    "strat_fold": row.strat_fold,
                    "relative_header_path": row.relative_header_path,
                    "relative_waveform_path": row.relative_waveform_path,
                }
            )


def _write_metadata(path: Path, rows: tuple[PtbxlAssignment, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ecg_id", "scp_codes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"ecg_id": row.ecg_id, "scp_codes": "{'SR': 0.0}"})


def _hash_files(source_root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.relative_to(source_root).as_posix()):
        relative = path.relative_to(source_root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _build_fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path, Path, Path]:
    expected = {"ptb-xl": 4, "georgia": 20, "cpsc_2018": 20, "cpsc_2018_extra": 20}
    monkeypatch.setattr(ecg_index, "EXPECTED_SOURCES", expected)
    root = tmp_path / "primary"
    assignments = tuple(_write_ptb_record(root, index + 1) for index in range(4))
    assignment_path = tmp_path / "ptbxl_verified_assignment.csv"
    _write_assignment(assignment_path, assignments)
    metadata_path = tmp_path / "ptbxl_database.csv"
    _write_metadata(metadata_path, assignments)

    prefixes = {"georgia": "G", "cpsc_2018": "A", "cpsc_2018_extra": "Q"}
    for source, prefix in prefixes.items():
        for index in range(20):
            _write_external_record(root / source, f"{prefix}{index:05d}")

    manifest = {
        "manifest_version": "0.2.0",
        "study": "TRUST-ECG",
        "status": "locked_before_waveform_model_training",
        "protocol_version": "0.4.0",
        "development_source": "original_ptbxl_v1_0_1",
        "external_source": "challenge2020_v1_0_2_georgia_cpsc",
        "challenge_ptbxl_model_input": False,
        "label_count_semantics": "union_of_scp_key_presence_per_record",
        "labels": [
            {
                "canonical_code": "426783006",
                "abbreviation": "NSR",
                "ptbxl_scp_codes": ["SR", "NORM"],
                "challenge_member_codes": ["426783006"],
            }
        ],
        "label_count": 1,
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest, "manifest_sha256")
    manifest_path = tmp_path / "labels.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    header_hashes: dict[str, str] = {}
    waveform_hashes: dict[str, str] = {}
    ptb_root = root / "ptb-xl"
    ptb_headers = [root / row.relative_header_path for row in assignments]
    ptb_waveforms = [root / row.relative_waveform_path for row in assignments]
    header_hashes["ptb-xl"] = _hash_files(ptb_root, ptb_headers)
    waveform_hashes["ptb-xl"] = _hash_files(ptb_root, ptb_waveforms)
    for source in prefixes:
        source_root = root / source
        headers = sorted(source_root.glob("*.hea"))
        waveforms = [path.with_suffix(".mat") for path in headers]
        header_hashes[source] = _hash_files(source_root, headers)
        waveform_hashes[source] = _hash_files(source_root, waveforms)

    waveform_audit = {
        "audit_version": "0.2.0",
        "development_source": "original_ptbxl_v1_0_1",
        "challenge_ptbxl_model_input": False,
        "label_manifest_sha256": manifest["manifest_sha256"],
        "ptbxl_assignment_sha256": assignment_sha256(assignments),
        "source_header_corpus_sha256": header_hashes,
        "source_waveform_corpus_sha256": waveform_hashes,
        "source_waveform_format": {
            "ptb-xl": "wfdb_dat_original_ptbxl_v1_0_1",
            "georgia": "challenge_mat_v4",
            "cpsc_2018": "challenge_mat_v4",
            "cpsc_2018_extra": "challenge_mat_v4",
        },
        "ready_for_model_stage": True,
        "audit_sha256": "",
    }
    waveform_audit["audit_sha256"] = _canonical_hash(waveform_audit, "audit_sha256")
    waveform_path = tmp_path / "waveform_audit.json"
    waveform_path.write_text(json.dumps(waveform_audit), encoding="utf-8")
    return root, metadata_path, waveform_path, manifest_path, assignment_path


def test_model_index_binds_source_specific_labels_roles_and_hashes(tmp_path: Path, monkeypatch) -> None:
    root, metadata, waveform, manifest, assignment = _build_fixture(tmp_path, monkeypatch)
    rows, audit = build_model_index(
        primary_data_root=root,
        ptbxl_metadata_path=metadata,
        waveform_audit_path=waveform,
        label_manifest_path=manifest,
        ptbxl_assignment_path=assignment,
    )
    assert audit.ready_for_baseline_execution is True
    assert audit.development_source == "original_ptbxl_v1_0_1"
    assert audit.challenge_ptbxl_model_input is False
    assert audit.corpus_hashes_verified is True
    assert audit.total_rows == 64
    assert audit.role_rows["model_fit"] == 1
    assert audit.role_rows["optimization_validation"] == 1
    assert audit.role_rows["calibration"] == 1
    assert audit.role_rows["internal_test"] == 1
    assert audit.role_rows["external_certification"] > 0
    assert audit.role_rows["external_recovery_pool"] > 0
    assert all(row.labels == (1,) for row in rows)
    assert all(
        row.waveform_format == "wfdb_dat_original_ptbxl_v1_0_1"
        for row in rows
        if row.source == "ptb-xl"
    )

    output = tmp_path / "outputs"
    write_model_index(
        rows,
        audit,
        index_output=output / "open_ecg_model_index.csv",
        audit_output=output / "open_ecg_model_index_audit.json",
    )
    verified = load_and_verify_index_audit(output / "open_ecg_model_index_audit.json")
    assert verified["index_sha256"] == audit.index_sha256


def test_external_mat_mutation_blocks_model_index(tmp_path: Path, monkeypatch) -> None:
    root, metadata, waveform, manifest, assignment = _build_fixture(tmp_path, monkeypatch)
    target = root / "georgia" / "G00000.mat"
    target.write_bytes(target.read_bytes() + b"tamper")
    _, audit = build_model_index(
        primary_data_root=root,
        ptbxl_metadata_path=metadata,
        waveform_audit_path=waveform,
        label_manifest_path=manifest,
        ptbxl_assignment_path=assignment,
    )
    assert audit.ready_for_baseline_execution is False
    assert "corpus_hash_changed_after_waveform_audit" in audit.blockers


def test_development_dat_mutation_blocks_model_index(tmp_path: Path, monkeypatch) -> None:
    root, metadata, waveform, manifest, assignment = _build_fixture(tmp_path, monkeypatch)
    target = root / "ptb-xl" / "records500" / "00000" / "00001_hr.dat"
    target.write_bytes(target.read_bytes() + b"tamper")
    _, audit = build_model_index(
        primary_data_root=root,
        ptbxl_metadata_path=metadata,
        waveform_audit_path=waveform,
        label_manifest_path=manifest,
        ptbxl_assignment_path=assignment,
    )
    assert audit.ready_for_baseline_execution is False
    assert "corpus_hash_changed_after_waveform_audit" in audit.blockers


def test_tampered_index_audit_is_rejected(tmp_path: Path, monkeypatch) -> None:
    root, metadata, waveform, manifest, assignment = _build_fixture(tmp_path, monkeypatch)
    rows, audit = build_model_index(
        primary_data_root=root,
        ptbxl_metadata_path=metadata,
        waveform_audit_path=waveform,
        label_manifest_path=manifest,
        ptbxl_assignment_path=assignment,
    )
    output = tmp_path / "outputs"
    write_model_index(
        rows,
        audit,
        index_output=output / "open_ecg_model_index.csv",
        audit_output=output / "open_ecg_model_index_audit.json",
    )
    audit_path = output / "open_ecg_model_index_audit.json"
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["total_rows"] += 1
    audit_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_and_verify_index_audit(audit_path)
