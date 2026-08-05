"""Validate public source-adapter files and execution order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trust_icu.adapter_manifest import load_and_validate_adapter_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("schemas/source_adapter_manifest.yaml"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    report = load_and_validate_adapter_manifest(args.manifest, args.repo_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if not report.valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
