"""Validate the public source-adapter contract without reading clinical data."""

from __future__ import annotations

import json
from pathlib import Path

from trust_icu.source_validation import (
    load_source_adapter_contract,
    mapping_sha256,
)


def main() -> None:
    path = Path("schemas/source_adapter_contract.yaml")
    raw = load_source_adapter_contract(path)
    report = {
        "contract": str(path),
        "contract_version": raw["contract_version"],
        "sha256": mapping_sha256(raw),
        "allowed_datasets": raw["allowed_datasets"],
        "allowed_tasks": raw["allowed_tasks"],
        "tables": sorted(raw["tables"]),
        "valid": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
