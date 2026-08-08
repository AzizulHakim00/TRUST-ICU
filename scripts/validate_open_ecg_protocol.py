#!/usr/bin/env python3
"""Validate the prospective open ECG transportability protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trust_icu.ecg_protocol import validate_open_ecg_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="schemas/open_ecg_protocol.yaml",
        help="Path to the prospective open ECG protocol.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_open_ecg_protocol(Path(args.protocol))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
