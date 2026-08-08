#!/usr/bin/env python3
"""Validate the conditional Phase 1 plan and optionally evaluate activation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trust_icu.phase1_gate import evaluate_phase1_activation, validate_phase1_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="schemas/phase1_conditional_protocol.yaml",
        help="Path to the prospective Phase 1 protocol.",
    )
    parser.add_argument(
        "--phase0-report",
        help="Optional verified phase0_go_no_go.json used only to evaluate activation.",
    )
    parser.add_argument(
        "--require-active",
        action="store_true",
        help="Exit with code 2 unless at least one Phase 0 task is eligible for Phase 1.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_path = Path(args.protocol)
    validation = validate_phase1_protocol(protocol_path)
    activation = evaluate_phase1_activation(
        protocol_path=protocol_path,
        phase0_report_path=args.phase0_report,
    )
    payload = {
        "protocol": validation,
        "activation": activation.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_active and not activation.architecture_or_method_development_allowed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
