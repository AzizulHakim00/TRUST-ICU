"""Protocol-safe helpers for the two TRUST-ECG secondary ResNet seeds."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

PRIMARY_SEED = 20260808
ALLOWED_REPLICATION_SEEDS = (20260809, 20260810)
_MUTATION_PATH = "phase0_models.resnet1d_fixed.random_seed"


def _nested_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Protocol lacks required mapping: {key}")
    return value


def build_replication_protocol(
    primary_protocol: Mapping[str, Any],
    replication_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deep-copy the frozen protocol and change only the ResNet initialization seed."""

    seed = int(replication_seed)
    if seed not in ALLOWED_REPLICATION_SEEDS:
        raise ValueError("Replication seed must be exactly 20260809 or 20260810.")

    phase0_models = _nested_mapping(primary_protocol, "phase0_models")
    resnet = _nested_mapping(phase0_models, "resnet1d_fixed")
    observed_primary_seed = int(resnet.get("random_seed", -1))
    if observed_primary_seed != PRIMARY_SEED:
        raise ValueError(
            "Source protocol primary seed has drifted; expected 20260808 before replication."
        )

    external_partition = _nested_mapping(primary_protocol, "external_partition")
    observed_partition_seed = int(external_partition.get("seed", -1))
    if observed_partition_seed != PRIMARY_SEED:
        raise ValueError(
            "External partition seed has drifted; expected frozen value 20260808."
        )

    replication = copy.deepcopy(dict(primary_protocol))
    replication["phase0_models"]["resnet1d_fixed"]["random_seed"] = seed

    restored = copy.deepcopy(replication)
    restored["phase0_models"]["resnet1d_fixed"]["random_seed"] = PRIMARY_SEED
    if restored != dict(primary_protocol):
        raise RuntimeError("Replication protocol changed fields outside the permitted seed path.")
    if int(replication["external_partition"]["seed"]) != PRIMARY_SEED:
        raise RuntimeError("Replication protocol changed the frozen external partition seed.")

    audit = {
        "primary_seed": PRIMARY_SEED,
        "replication_seed": seed,
        "allowed_replication_seeds": list(ALLOWED_REPLICATION_SEEDS),
        "only_scientific_mutation": _MUTATION_PATH,
        "external_partition_seed_unchanged": True,
        "primary_protocol_unmodified": dict(primary_protocol) == restored,
    }
    return replication, audit
