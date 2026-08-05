#!/usr/bin/env python3
"""Build a hashed aggregate outcome-validation summary from a canonical run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trust_icu.outcome_evidence import build_local_outcome_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mimic_iv_3_1", "eicu_crd_2_0"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_local_outcome_summary(
        run_dir=args.run_dir,
        dataset=args.dataset,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "dataset": report.dataset,
                "summary_sha256": report.summary_sha256,
                "tasks": [
                    {
                        "task": item.task,
                        "eligible_stays": item.eligible_stays,
                        "prediction_window_event_stays": item.prediction_window_event_stays,
                        "incident_hospitals": item.incident_hospitals,
                        "invalid_intervals": item.invalid_intervals,
                    }
                    for item in report.tasks
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
