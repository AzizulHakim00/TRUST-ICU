#!/usr/bin/env python3
"""Generate aggregate manuscript tables, figures, and reporting traceability from Phase 0."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from trust_icu.reporting import build_reporting_dry_run_plan, generate_publication_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--phase0-report", type=Path)
    output_default = os.environ.get("TRUST_ICU_REPORTING_OUTPUT_ROOT")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(output_default) if output_default else None,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps(build_reporting_dry_run_plan(args.repo_root), indent=2, sort_keys=True))
        return 0

    missing = [
        name
        for name, value in (
            ("--phase0-report", args.phase0_report),
            ("--output-root or TRUST_ICU_REPORTING_OUTPUT_ROOT", args.output_root),
        )
        if value is None
    ]
    if missing:
        parser.error("Missing required execution arguments: " + ", ".join(missing))

    bundle = generate_publication_bundle(
        repo_root=args.repo_root,
        phase0_report=args.phase0_report,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": "publication_bundle_generated",
                "output_root": bundle.output_root,
                "source_report_sha256": bundle.source_report_sha256,
                "reporting_contract_sha256": bundle.reporting_contract_sha256,
                "manifest_sha256": bundle.manifest_sha256,
                "files": list(bundle.files),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
