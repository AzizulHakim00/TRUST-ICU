#!/usr/bin/env python3
"""Materialize one allowed TRUST-ECG secondary-seed protocol copy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from trust_icu.ecg_protocol import load_open_ecg_protocol, validate_open_ecg_protocol
from trust_icu.ecg_secondary_seed import build_replication_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="schemas/open_ecg_protocol.yaml")
    parser.add_argument("--seed", type=int, required=True, choices=(20260809, 20260810))
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    primary_path = Path(args.protocol).expanduser().resolve()
    primary_validation = validate_open_ecg_protocol(primary_path)
    primary = load_open_ecg_protocol(primary_path)
    replication, audit = build_replication_protocol(primary, args.seed)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(replication, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    replication_validation = validate_open_ecg_protocol(output)

    audit_output = Path(args.audit_output).expanduser().resolve()
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    final_audit = {
        **audit,
        "primary_protocol_sha256": _sha256(primary_path),
        "replication_protocol_sha256": _sha256(output),
        "primary_validation": primary_validation,
        "replication_validation": replication_validation,
    }
    audit_output.write_text(
        json.dumps(final_audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
