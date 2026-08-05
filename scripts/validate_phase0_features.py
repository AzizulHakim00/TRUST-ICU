"""Validate the public Phase 0 feature contract without reading clinical data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trust_icu.features import load_feature_contract

DEFAULT_CONTRACT = Path(__file__).resolve().parents[1] / "schemas" / "phase0_features.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    contract = load_feature_contract(args.contract)
    print(
        json.dumps(
            {
                "contract_version": contract.version,
                "observation_window_minutes": [
                    contract.observation_start_minutes,
                    contract.observation_end_minutes,
                ],
                "variable_count": len(contract.variables),
                "variables": list(contract.variable_names),
                "status": "valid_public_feature_contract",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
