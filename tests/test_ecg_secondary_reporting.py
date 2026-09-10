from __future__ import annotations

import hashlib
import json
from pathlib import Path

from trust_icu.ecg_secondary_reporting import (
    build_secondary_aggregate_payload,
    verify_embedded_report_hash,
    write_secondary_aggregate_package,
)


def _with_report_hash(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["report_sha256"] = ""
    material = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    result["report_sha256"] = hashlib.sha256(material).hexdigest()
    return result


def _metric(*, ratio: float = 3.0, intercept: float = 0.0) -> dict[str, object]:
    return {
        "n": 200,
        "positives": 100,
        "negatives": 100,
        "prevalence": 0.5,
        "pr_auc": 0.8,
        "pr_auc_to_prevalence_ratio": ratio,
        "roc_auc": 0.9,
        "brier": 0.1,
        "brier_skill_vs_prevalence": 0.2,
        "calibration_slope": 1.0,
        "calibration_intercept": intercept,
    }


def _phase0() -> dict[str, object]:
    return _with_report_hash(
        {
            "study": "TRUST-ECG",
            "protocol_version": "0.4.0",
            "protocol_sha256": "protocol-hash",
            "model_sha256": "model-hash",
            "external_certification": {
                "georgia": {
                    "A": {"status": "certified", "metrics": _metric()},
                    "B": {
                        "status": "calibration_recovery_candidate",
                        "metrics": _metric(intercept=1.1),
                    },
                    "C": {
                        "status": "discrimination_failure",
                        "metrics": _metric(ratio=1.0),
                    },
                    "D": {"status": "insufficient_support", "metrics": None},
                }
            },
        }
    )


def _phase1() -> dict[str, object]:
    return _with_report_hash(
        {
            "phase1_plan": {"repeats": 100},
            "pair_results": {
                "georgia/B": {
                    "source": "georgia",
                    "label_code": "B",
                    "budgets": {
                        "50": {
                            "methods": {
                                "platt_recalibration": {
                                    "estimable_repeats": 90,
                                    "nonestimable_repeats": 10,
                                    "nonestimable_reasons": {
                                        "single_class_adaptation_sample": 10
                                    },
                                    "recovery_envelope_met_count": 81,
                                    "recovery_envelope_met_rate_among_estimable": 0.9,
                                }
                            }
                        }
                    },
                }
            },
        }
    )


def test_embedded_hash_verification_fails_closed() -> None:
    report = _phase0()
    assert verify_embedded_report_hash(report) is True
    tampered = json.loads(json.dumps(report))
    tampered["model_sha256"] = "tampered"
    assert verify_embedded_report_hash(tampered) is False


def test_secondary_payload_is_aggregate_only_and_reproduces_official_counts() -> None:
    payload = build_secondary_aggregate_payload(_phase0(), _phase1())
    assert payload["primary_provenance"]["phase0_report_sha256"] == _phase0()["report_sha256"]
    assert payload["official_status_counts"] == {
        "certified": 1,
        "calibration_recovery_candidate": 1,
        "discrimination_failure": 1,
        "insufficient_support": 1,
    }
    assert len(payload["envelope_sensitivity"]) == 72
    assert len(payload["phase1_estimability"]) == 1
    serialized = json.dumps(payload, sort_keys=True).lower()
    for forbidden in ('"record_id"', '"patient_id"', '"predictions"', '"logits"', '"waveforms"'):
        assert forbidden not in serialized


def test_secondary_package_writes_manifest_tracked_aggregate_files(tmp_path: Path) -> None:
    summary = write_secondary_aggregate_package(_phase0(), _phase1(), tmp_path)
    expected = {
        "secondary_summary.json",
        "envelope_sensitivity.csv",
        "framework_ablation.csv",
        "phase1_estimability.csv",
        "secondary_sha256_manifest.json",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    manifest = json.loads((tmp_path / "secondary_sha256_manifest.json").read_text())
    assert set(manifest) == expected - {"secondary_sha256_manifest.json"}
    for name, digest in manifest.items():
        observed = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert observed == digest
    assert summary["manifest_entries"] == 4
