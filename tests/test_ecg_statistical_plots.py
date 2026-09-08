from __future__ import annotations

from trust_icu.ecg_statistical_plots import plot_internal_calibration


def test_internal_calibration_tolerates_roundoff_in_wilson_bounds(tmp_path):
    rows = [
        {
            "scope": "internal_fold10",
            "source": "ptb-xl",
            "model": model_name,
            "label_code": "59118001",
            "label_name": "RBBB",
            "bin": 1,
            "n": 3210,
            "mean_predicted_probability": 0.001,
            "observed_prevalence": 0.0,
            "observed_wilson_low": 1.0842021724855044e-19,
            "observed_wilson_high": 0.001195285725794244,
            "minimum_probability": 0.0,
            "maximum_probability": 0.002,
        }
        for model_name in ("fixed_resnet", "logistic_reference")
    ]

    plot_internal_calibration(rows, tmp_path, ("59118001",))

    assert (tmp_path / "internal_calibration.png").is_file()
    assert (tmp_path / "internal_calibration.pdf").is_file()
