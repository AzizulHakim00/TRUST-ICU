from trust_icu.ecg_secondary_reporting import (
    build_secondary_provenance,
    validate_public_secondary_payload,
    validate_secondary_output_root,
)


def test_privacy_validator_rejects_record_identifiers():
    try:
        validate_public_secondary_payload({"record_id": "A0001"})
    except ValueError as exc:
        assert "record" in str(exc).lower()
    else:
        raise AssertionError("record-level identifier was not rejected")


def test_privacy_validator_rejects_record_level_probabilities():
    try:
        validate_public_secondary_payload({"record_level_probabilities": [0.1, 0.9]})
    except ValueError as exc:
        assert "probab" in str(exc).lower()
    else:
        raise AssertionError("record-level probabilities were not rejected")


def test_privacy_validator_accepts_aggregate_counts_and_intervals():
    validate_public_secondary_payload(
        {
            "certified": 9,
            "calibration_recovery_candidate": 9,
            "gate_satisfaction_rate": 0.88,
            "metric_intervals": {"pr_auc": {"q025": 0.7, "q975": 0.9}},
        }
    )


def test_output_root_cannot_be_inside_primary_results(tmp_path):
    primary = tmp_path / "primary_results"
    try:
        validate_secondary_output_root(primary / "secondary", primary_roots=[primary])
    except ValueError as exc:
        assert "primary" in str(exc).lower()
    else:
        raise AssertionError("secondary output was allowed inside primary results")


def test_secondary_provenance_binds_primary_hashes():
    phase0 = {
        "study": "TRUST-ECG",
        "protocol_version": "0.4.0",
        "protocol_sha256": "a" * 64,
        "report_sha256": "b" * 64,
        "model_sha256": "c" * 64,
        "model_index_sha256": "d" * 64,
        "label_manifest_sha256": "e" * 64,
    }
    phase1 = {
        "study": "TRUST-ECG",
        "protocol_sha256": "a" * 64,
        "phase0_report_sha256": "b" * 64,
        "phase0_model_sha256": "c" * 64,
        "report_sha256": "f" * 64,
    }
    provenance = build_secondary_provenance(phase0, phase1)
    assert provenance["primary_phase0_report_sha256"] == "b" * 64
    assert provenance["primary_model_sha256"] == "c" * 64
    assert provenance["primary_phase1_report_sha256"] == "f" * 64
