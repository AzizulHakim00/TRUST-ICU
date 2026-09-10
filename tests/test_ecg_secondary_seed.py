from __future__ import annotations

import copy

import pytest
from trust_icu.ecg_secondary_seed import build_replication_protocol


def _protocol() -> dict[str, object]:
    return {
        "version": "0.4.0",
        "external_partition": {"seed": 20260808, "certification_fraction": 0.6},
        "phase0_models": {
            "primary_model": "resnet1d_fixed",
            "resnet1d_fixed": {
                "random_seed": 20260808,
                "batch_size": 64,
                "max_epochs": 50,
            },
        },
        "calibration": {"fit_source": "ptb_xl_fold_9_only"},
    }


def test_replication_protocol_changes_only_resnet_seed() -> None:
    primary = _protocol()
    original = copy.deepcopy(primary)
    replication, audit = build_replication_protocol(primary, 20260809)
    assert primary == original
    assert replication["phase0_models"]["resnet1d_fixed"]["random_seed"] == 20260809  # type: ignore[index]
    assert replication["external_partition"]["seed"] == 20260808  # type: ignore[index]
    restored = copy.deepcopy(replication)
    restored["phase0_models"]["resnet1d_fixed"]["random_seed"] = 20260808  # type: ignore[index]
    assert restored == original
    assert audit["primary_seed"] == 20260808
    assert audit["replication_seed"] == 20260809
    assert audit["only_scientific_mutation"] == "phase0_models.resnet1d_fixed.random_seed"
    assert audit["external_partition_seed_unchanged"] is True


def test_replication_protocol_allows_only_two_predeclared_secondary_seeds() -> None:
    for seed in (20260809, 20260810):
        replication, _ = build_replication_protocol(_protocol(), seed)
        assert replication["phase0_models"]["resnet1d_fixed"]["random_seed"] == seed  # type: ignore[index]

    for disallowed in (20260808, 20260811, 1):
        with pytest.raises(ValueError, match="20260809.*20260810"):
            build_replication_protocol(_protocol(), disallowed)


def test_replication_protocol_fails_if_primary_seed_already_drifted() -> None:
    primary = _protocol()
    primary["phase0_models"]["resnet1d_fixed"]["random_seed"] = 9  # type: ignore[index]
    with pytest.raises(ValueError, match="primary seed"):
        build_replication_protocol(primary, 20260809)
