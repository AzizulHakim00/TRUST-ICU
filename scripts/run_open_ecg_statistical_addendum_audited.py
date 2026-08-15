#!/usr/bin/env python3
"""Run the TRUST-ECG addendum with audited Phase-0 probability reconstruction."""

from __future__ import annotations

import runpy
from pathlib import Path

from trust_icu import ecg_statistical_models as models
from trust_icu.ecg_statistical_reconstruction import (
    apply_phase0_calibration_payload,
    validate_resnet_report_metrics_audited,
)


def main() -> None:
    # The original addendum remains unchanged. Patch only the two reconstruction
    # hooks before it imports its function references.
    models._global_calibrated_probabilities = apply_phase0_calibration_payload
    models.validate_resnet_report_metrics = validate_resnet_report_metrics_audited
    script = Path(__file__).with_name("run_open_ecg_statistical_addendum.py")
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
