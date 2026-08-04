"""Audit local canonical source-adapter exports and emit aggregate-only JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from trust_icu.features import load_feature_contract
from trust_icu.source_validation import (
    audit_canonical_extract,
    load_source_adapter_contract,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return pd.read_csv(path)
    if name.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet input requires pyarrow or fastparquet in the credentialed environment."
            ) from exc
    raise ValueError(f"Unsupported table format: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["mimic_iv_3_1", "eicu_crd_2_0"],
    )
    parser.add_argument("--stays", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument(
        "--source-contract",
        type=Path,
        default=Path("schemas/source_adapter_contract.yaml"),
    )
    parser.add_argument(
        "--feature-contract",
        type=Path,
        default=Path("schemas/phase0_features.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    for path in (args.stays, args.events, args.observations):
        if not path.is_file():
            raise FileNotFoundError(path)

    audit = audit_canonical_extract(
        _read_table(args.stays),
        _read_table(args.events),
        _read_table(args.observations),
        dataset=args.dataset,
        source_contract=load_source_adapter_contract(args.source_contract),
        feature_contract=load_feature_contract(args.feature_contract),
    )
    payload = {
        "data_classification": "credentialed_local_extract",
        "contains_row_level_examples": False,
        "inputs": {
            "stays": {"name": args.stays.name, "sha256": _sha256(args.stays)},
            "events": {"name": args.events.name, "sha256": _sha256(args.events)},
            "observations": {
                "name": args.observations.name,
                "sha256": _sha256(args.observations),
            },
        },
        "audit": audit.to_dict(),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.require_ready and not audit.ready_for_cohort_build:
        sys.exit(2)


if __name__ == "__main__":
    main()
