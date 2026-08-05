#!/usr/bin/env python3
"""Check TRUST-ICU access paths without reading or exposing patient data."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

from trust_icu.config import load_config

ROOT_VARIABLES = {
    "MIMIC-IV v3.1": "TRUST_ICU_MIMIC_ROOT",
    "eICU-CRD v2.0": "TRUST_ICU_EICU_ROOT",
    "AmsterdamUMCdb v1.5.0": "TRUST_ICU_AMSTERDAM_ROOT",
    "HiRID v1.1.1": "TRUST_ICU_HIRID_ROOT",
    "Output root": "TRUST_ICU_OUTPUT_ROOT",
}


def path_status(variable: str) -> dict[str, object]:
    raw = os.getenv(variable)
    if not raw:
        return {"environment_variable": variable, "configured": False, "exists": False}
    path = Path(raw).expanduser().resolve()
    return {
        "environment_variable": variable,
        "configured": True,
        "exists": path.exists(),
        "is_directory": path.is_dir(),
        "path": str(path),
    }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    config = load_config(repository_root / "configs" / "feasibility.yaml")
    checks = {name: path_status(variable) for name, variable in ROOT_VARIABLES.items()}

    output_raw = os.getenv("TRUST_ICU_OUTPUT_ROOT")
    output_inside_repository = False
    if output_raw:
        output_path = Path(output_raw).expanduser().resolve()
        output_inside_repository = repository_root == output_path or repository_root in output_path.parents

    report = {
        "study": config.name,
        "phase": config.phase,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "repository_root": str(repository_root),
        "paths": checks,
        "output_inside_public_repository": output_inside_repository,
        "ready_for_phase_0": (
            checks["MIMIC-IV v3.1"]["exists"]
            and checks["eICU-CRD v2.0"]["exists"]
            and checks["Output root"]["exists"]
            and not output_inside_repository
        ),
    }
    print(json.dumps(report, indent=2))

    if output_inside_repository:
        print("ERROR: TRUST_ICU_OUTPUT_ROOT must be outside the public repository.", file=sys.stderr)
        return 2
    return 0 if report["ready_for_phase_0"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
