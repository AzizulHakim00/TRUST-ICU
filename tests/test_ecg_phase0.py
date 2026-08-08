from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trust_icu.ecg_index import EcgIndexRow, model_index_sha256
from trust_icu.ecg_phase0 import (
    build_phase0_dry_run_plan,
    execute_logistic_reference_phase0,
    load_and_verify_model_index,
    load_and_verify_normalization_stats,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "schemas/open_ecg_protocol.yaml"


def _hash_payload(payload: dict, key: str) -> str:
    material = dict(payload)
    material[key] = ""
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fixture_index(tmp_path: Path) -> tuple[Path, Path, Path, list[EcgIndexRow]]:
    rows: list[EcgIndexRow] = []
    counter = 1

    def add(source: str, role: str, labels: list[int]) -> None:
        nonlocal counter
        for label in labels:
            record_id = f"R{counter:05d}"
            rows.append(
                EcgIndexRow(
                    source=source,
                    record_id=record_id,
                    relative_header_path=f"{source}/{record_id}.hea",
                    relative_mat_path=f"{source}/{record_id}.mat",
                    role=role,  # type: ignore[arg-type]
                    strat_fold={
                        "model_fit": 1,
                        "optimization_validation": 8,
                        "calibration": 9,
                        "internal_test": 10,
                    }.get(role),
                    labels=(label,),
                )
            )
            counter += 1

    add("ptb-xl", "model_fit", [0, 1, 0, 1, 0, 1])
    add("ptb-xl", "optimization_validation", [0, 1])
    add("ptb-xl", "calibration", [0, 1, 0, 1])
    add("ptb-xl", "internal_test", [0, 1, 0, 1])
    for source in ("georgia", "cpsc_2018", "cpsc_2018_extra"):
        add(source, "external_certification", [0, 1, 0, 1])
        add(source, "external_recovery_pool", [0, 1])

    label_codes = ("426783006",)
    index_path = tmp_path / "open_ecg_model_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "record_id",
                "relative_header_path",
                "relative_mat_path",
                "role",
                "strat_fold",
                "label_426783006",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source": row.source,
                    "record_id": row.record_id,
                    "relative_header_path": row.relative_header_path,
                    "relative_mat_path": row.relative_mat_path,
                    "role": row.role,
                    "strat_fold": "" if row.strat_fold is None else row.strat_fold,
                    "label_426783006": row.labels[0],
                }
            )

    manifest = {
        "manifest_version": "0.1.0",
        "study": "TRUST-ECG",
        "status": "locked_before_waveform_model_training",
        "labels": [
            {
                "canonical_code": "426783006",
                "member_codes": ["426783006"],
            }
        ],
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = _hash_payload(manifest, "manifest_sha256")
    manifest_path = tmp_path / "open_ecg_label_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    source_counts = Counter(row.source for row in rows)
    role_counts = Counter(row.role for row in rows)
    audit = {
        "audit_version": "0.1.0",
        "waveform_audit_sha256": "a" * 64,
        "label_manifest_sha256": manifest["manifest_sha256"],
        "label_codes": list(label_codes),
        "total_rows": len(rows),
        "source_rows": dict(sorted(source_counts.items())),
        "role_rows": dict(sorted(role_counts.items())),
        "source_role_rows": {},
        "source_label_positives": {},
        "rows_with_no_locked_positive_label": 0,
        "duplicate_source_record_ids": 0,
        "corpus_hashes_verified": True,
        "index_sha256": model_index_sha256(rows, label_codes),
        "ready_for_baseline_execution": True,
        "blockers": [],
        "audit_sha256": "",
    }
    audit["audit_sha256"] = _hash_payload(audit, "audit_sha256")
    audit_path = tmp_path / "open_ecg_model_index_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return index_path, audit_path, manifest_path, rows


def test_phase0_dry_run_keeps_recovery_pool_prohibited() -> None:
    plan = build_phase0_dry_run_plan(PROTOCOL)
    assert plan["primary_gate_model"] == "resnet1d_fixed_only"
    assert plan["external_recovery_pool_access"] == "prohibited"


def test_model_index_loader_rejects_tampered_label(tmp_path: Path) -> None:
    index_path, audit_path, _, _ = _fixture_index(tmp_path)
    rows, _ = load_and_verify_model_index(index_csv=index_path, index_audit_path=audit_path)
    assert any(row.role == "external_recovery_pool" for row in rows)

    text = index_path.read_text(encoding="utf-8")
    index_path.write_text(text.replace(",external_recovery_pool,,0", ",external_recovery_pool,,1", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_and_verify_model_index(index_csv=index_path, index_audit_path=audit_path)


def test_normalization_stats_require_folds_1_to_7(tmp_path: Path) -> None:
    payload = {
        "means_mv": [0.0] * 12,
        "stds_mv": [1.0] * 12,
        "valid_sample_counts": [100] * 12,
        "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "fit_folds": [1, 2, 3, 4, 5, 6, 7],
        "stats_sha256": "",
    }
    payload["stats_sha256"] = _hash_payload(payload, "stats_sha256")
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    stats = load_and_verify_normalization_stats(path)
    assert stats.fit_folds == (1, 2, 3, 4, 5, 6, 7)

    payload["fit_folds"] = [1, 2, 3, 4, 5, 6, 7, 8]
    payload["stats_sha256"] = _hash_payload(payload, "stats_sha256")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="folds 1-7"):
        load_and_verify_normalization_stats(path)


def test_logistic_phase0_never_loads_recovery_records(tmp_path: Path, monkeypatch) -> None:
    index_path, audit_path, manifest_path, rows = _fixture_index(tmp_path)
    recovery_ids = {row.record_id for row in rows if row.role == "external_recovery_pool"}
    loaded_ids: list[str] = []

    def fake_load(_root: Path, row: EcgIndexRow):
        if row.record_id in recovery_ids:
            raise AssertionError("Recovery-pool record was accessed in Phase 0")
        loaded_ids.append(row.record_id)
        number = int(row.record_id[1:])
        time = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        waveform = np.stack(
            [
                0.01 * lead + (1.0 + 0.05 * number) * np.sin(time + lead * 0.03) + number * 0.001
                for lead in range(12)
            ],
            axis=0,
        ).astype(np.float32)
        return SimpleNamespace(
            waveform_mv=waveform,
            valid_mask=np.ones(waveform.shape[1], dtype=bool),
        )

    monkeypatch.setattr("trust_icu.ecg_phase0._load_standardized_record", fake_load)
    report = execute_logistic_reference_phase0(
        challenge_training_root=tmp_path,
        index_csv=index_path,
        index_audit_path=audit_path,
        label_manifest_path=manifest_path,
        protocol_path=PROTOCOL,
    )
    assert report.model_name == "logistic_regression_handcrafted"
    assert report.primary_gate_eligible is False
    assert report.external_recovery_pool_used is False
    assert not recovery_ids.intersection(loaded_ids)
    assert report.internal_test.n == 4
    assert set(report.external_certification) == {"georgia", "cpsc_2018", "cpsc_2018_extra"}
    assert len(report.report_sha256) == 64
