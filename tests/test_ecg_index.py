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


def _write_record(source_root: Path, record_id: str, dx: str = "426783006") -> None:
    source_root.mkdir(parents=True, exist_ok=True)
    lines = [f"{record_id} 12 500 5000"]
    lines.extend(
        f"{record_id}.mat 16+24 1000/mV 16 0 0 {index + 1} 0 {lead}"
        for index, lead in enumerate(EXPECTED_LEADS)
    )
    lines.extend(["#Age: 50", "#Sex: Male", f"#Dx: {dx}"])
    (source_root / f"{record_id}.hea").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (source_root / f"{record_id}.mat").write_bytes((record_id + "-waveform").encode())


def _write_assignment(path: Path, ptb_ids: list[str]) -> tuple[PtbxlAssignment, ...]:
    folds = [1, 8, 9, 10]
    rows = tuple(
        PtbxlAssignment(challenge_record_id=record_id, ecg_id=index + 1, strat_fold=folds[index])
        for index, record_id in enumerate(ptb_ids)
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["challenge_record_id", "ecg_id", "strat_fold"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "challenge_record_id": row.challenge_record_id,
                    "ecg_id": row.ecg_id,
                    "strat_fold": row.strat_fold,
                }
            )
    return rows


def _corpus_hash(source_root: Path, header_paths: list[Path], suffix: str) -> str:
    digest = hashlib.sha256()
    for header in header_paths:
        path = header if suffix == ".hea" else header.with_suffix(suffix)
        relative = path.relative_to(source_root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _build_fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path, Path]:
    expected = {"ptb-xl": 4, "georgia": 20, "cpsc_2018": 20, "cpsc_2018_extra": 20}
    monkeypatch.setattr(ecg_index, "EXPECTED_SOURCES", expected)
    training = tmp_path / "training"
    ptb_ids = [f"HR{index:05d}" for index in range(4)]
    for record_id in ptb_ids:
        _write_record(training / "ptb-xl", record_id)
    prefixes = {"georgia": "G", "cpsc_2018": "A", "cpsc_2018_extra": "Q"}
    for source, prefix in prefixes.items():
        for index in range(20):
            _write_record(training / source, f"{prefix}{index:05d}")

    assignment_path = tmp_path / "ptbxl_verified_assignment.csv"
    assignments = _write_assignment(assignment_path, ptb_ids)

    manifest = {
        "manifest_version": "0.1.0",
        "study": "TRUST-ECG",
        "status": "locked_before_waveform_model_training",
        "labels": [
            {
                "canonical_code": "426783006",
                "abbreviation": "NSR",
                "member_codes": ["426783006"],
            }
        ],
        "label_count": 1,
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest, "manifest_sha256")
    manifest_path = tmp_path / "labels.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    header_hashes = {}
    waveform_hashes = {}
    for source in expected:
        source_root = training / source
        headers = sorted(
            source_root.glob("*.hea"),
            key=lambda path: (ecg_index._numeric_record_id(path.stem), str(path)),
        )
        header_hashes[source] = _corpus_hash(source_root, headers, ".hea")
        waveform_hashes[source] = _corpus_hash(source_root, headers, ".mat")

    waveform_audit = {
        "label_manifest_sha256": manifest["manifest_sha256"],
        "ptbxl_assignment_sha256": assignment_sha256(assignments),
        "source_header_corpus_sha256": header_hashes,
        "source_waveform_corpus_sha256": waveform_hashes,
        "ready_for_model_stage": True,
        "audit_sha256": "",
    }
    waveform_audit["audit_sha256"] = _canonical_hash(waveform_audit, "audit_sha256")
    waveform_path = tmp_path / "waveform_audit.json"
    waveform_path.write_text(json.dumps(waveform_audit), encoding="utf-8")
    return training, waveform_path, manifest_path, assignment_path


def test_model_index_binds_roles_labels_and_corpus_hashes(tmp_path: Path, monkeypatch) -> None:
    training, waveform, manifest, assignment = _build_fixture(tmp_path, monkeypatch)
    rows, audit = build_model_index(
        challenge_training_root=training,
        waveform_audit_path=waveform,
        label_manifest_path=manifest,
        ptbxl_assignment_path=assignment,
    )
    assert audit.ready_for_baseline_execution is True
    assert audit.corpus_hashes_verified is True
    assert audit.total_rows == 64
    assert audit.role_rows["model_fit"] == 1
    assert audit.role_rows["optimization_validation"] == 1
    assert audit.role_rows["calibration"] == 1
    assert audit.role_rows["internal_test"] == 1
    assert audit.role_rows["external_certification"] > 0
    assert audit.role_rows["external_recovery_pool"] > 0
    assert all(row.labels == (1,) for row in rows)

    output = tmp_path / "outputs"
    write_model_index(
        rows,
        audit,
        index_output=output / "open_ecg_model_index.csv",
        audit_output=output / "open_ecg_model_index_audit.json",
    )
    verified = load_and_verify_index_audit(output / "open_ecg_model_index_audit.json")
    assert verified["index_sha256"] == audit.index_sha256


def test_waveform_mutation_blocks_model_index(tmp_path: Path, monkeypatch) -> None:
    training, waveform, manifest, assignment = _build_fixture(tmp_path, monkeypatch)
    target = training / "georgia" / "G00000.mat"
    target.write_bytes(target.read_bytes() + b"tamper")
    _, audit = build_model_index(
        challenge_training_root=training,
        waveform_audit_path=waveform,
        label_manifest_path=manifest,
        ptbxl_assignment_path=assignment,
    )
    assert audit.ready_for_baseline_execution is False
    assert "corpus_hash_changed_after_waveform_audit" in audit.blockers


def test_tampered_index_audit_is_rejected(tmp_path: Path, monkeypatch) -> None:
    training, waveform, manifest, assignment = _build_fixture(tmp_path, monkeypatch)
    rows, audit = build_model_index(
        challenge_training_root=training,
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
