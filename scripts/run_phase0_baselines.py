#!/usr/bin/env python3
"""Run the locked TRUST-ICU Phase 0 temporal and external baselines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from trust_icu.phase0_runner import build_phase0_dry_run_plan, execute_phase0_baselines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--mimic-run-dir", type=Path)
    parser.add_argument("--eicu-run-dir", type=Path)
    output_default = os.environ.get("TRUST_ICU_PHASE0_OUTPUT_ROOT")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(output_default) if output_default else None,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("logistic_regression", "catboost"),
        default=["logistic_regression", "catboost"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps(build_phase0_dry_run_plan(args.repo_root), indent=2, sort_keys=True))
        return 0

    missing = [
        name
        for name, value in (
            ("--mimic-run-dir", args.mimic_run_dir),
            ("--eicu-run-dir", args.eicu_run_dir),
            ("--output-root or TRUST_ICU_PHASE0_OUTPUT_ROOT", args.output_root),
        )
        if value is None
    ]
    if missing:
        parser.error("Missing required execution arguments: " + ", ".join(missing))

    report = execute_phase0_baselines(
        repo_root=args.repo_root,
        mimic_run_dir=args.mimic_run_dir,
        eicu_run_dir=args.eicu_run_dir,
        output_root=args.output_root,
        overwrite=args.overwrite,
        models=tuple(args.models),
    )
    print(
        json.dumps(
            {
                "study": report.study,
                "phase": report.phase,
                "all_tasks_continue": report.all_tasks_continue,
                "report_sha256": report.report_sha256,
                "tasks": [
                    {
                        "task": task.task,
                        "selected_model": task.selected_model,
                        "continue_to_architecture_development": task.feasibility_decision.continue_to_architecture_development,
                    }
                    for task in report.tasks
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
