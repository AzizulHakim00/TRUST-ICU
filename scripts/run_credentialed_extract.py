#!/usr/bin/env python3
"""Run a secure TRUST-ICU canonical extraction inside a credentialed environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from trust_icu.credentialed_runner import (
    build_dry_run_plan,
    execute_credentialed_run,
    prepare_eicu_mapping_tables,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("mimic_iv_3_1", "eicu_crd_2_0"),
        default="mimic_iv_3_1",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("TRUST_ICU_POSTGRES_DSN", ""),
        help="PostgreSQL DSN. Prefer TRUST_ICU_POSTGRES_DSN to avoid shell history.",
    )
    output_default = os.environ.get("TRUST_ICU_OUTPUT_ROOT")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(output_default) if output_default else None,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-eicu-mappings", action="store_true")
    parser.add_argument("--allow-reviewed-eicu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(
            json.dumps(
                build_dry_run_plan(args.repo_root, args.dataset),
                indent=2,
                sort_keys=True,
            )
        )
        return

    if not args.dsn:
        raise SystemExit(
            "TRUST_ICU_POSTGRES_DSN is not set. Credentials are never read from repository files."
        )

    if args.prepare_eicu_mappings:
        if args.dataset != "eicu_crd_2_0":
            raise SystemExit("--prepare-eicu-mappings is valid only for eicu_crd_2_0.")
        prepare_eicu_mapping_tables(repo_root=args.repo_root, dsn=args.dsn)
        print("Created/verified empty local eICU mapping tables. No mappings were approved.")
        return

    if args.output_root is None:
        raise SystemExit("TRUST_ICU_OUTPUT_ROOT is not set.")
    report = execute_credentialed_run(
        repo_root=args.repo_root,
        dataset=args.dataset,
        dsn=args.dsn,
        output_root=args.output_root,
        allow_reviewed_eicu=args.allow_reviewed_eicu,
        overwrite=args.overwrite,
    )
    summary = {
        "dataset": report.dataset,
        "completed_at_utc": report.completed_at_utc,
        "ready_for_cohort_build": report.audit["ready_for_cohort_build"],
        "stay_rows": report.audit["stay_rows"],
        "event_rows": report.audit["event_rows"],
        "observation_rows": report.audit["observation_rows"],
        "report_sha256": report.report_sha256,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
