#!/usr/bin/env python3
"""Validate outcome contracts and emit a public metadata-only lock report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trust_icu.outcomes import evaluate_outcome_locks, load_outcome_contracts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACTS = ROOT / "schemas" / "outcome_contracts.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path. Keep patient-level paths outside the repository.",
    )
    parser.add_argument(
        "--require-locked",
        action="store_true",
        help="Return a non-zero exit code unless every task is fully locked.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = load_outcome_contracts(args.contracts)
    reports = [report.to_dict() for report in evaluate_outcome_locks(raw)]
    payload = {
        "contracts": str(args.contracts),
        "all_tasks_locked": all(report["ready_for_model_training"] for report in reports),
        "reports": reports,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote outcome lock report: {args.output}")

    if args.require_locked and not payload["all_tasks_locked"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
