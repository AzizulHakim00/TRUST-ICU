"""Aggregate tables and publication plots for the TRUST-ECG addendum."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from trust_icu.ecg_statistical_core import equal_frequency_calibration_bins
from trust_icu.ecg_statistical_models import EXTERNAL_SOURCES, LABEL_NAMES

plt.switch_backend("Agg")


def calibration_records(
    *,
    internal_targets: np.ndarray,
    resnet_internal: np.ndarray,
    logistic_internal: np.ndarray,
    external_rows: list[Any],
    external_targets: np.ndarray,
    resnet_external: np.ndarray,
    logistic_external: np.ndarray,
    label_codes: tuple[str, ...],
    candidate_pairs: set[tuple[str, str]],
    bins: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model_name, probabilities in (
        ("fixed_resnet", resnet_internal),
        ("logistic_reference", logistic_internal),
    ):
        for label_index, code in enumerate(label_codes):
            for row in equal_frequency_calibration_bins(
                internal_targets[:, label_index],
                probabilities[:, label_index],
                bins=bins,
            ):
                output.append(
                    {
                        "scope": "internal_fold10",
                        "source": "ptb-xl",
                        "model": model_name,
                        "label_code": code,
                        "label_name": LABEL_NAMES[code],
                        **row,
                    }
                )
    for source in EXTERNAL_SOURCES:
        indices = np.asarray(
            [index for index, row in enumerate(external_rows) if row.source == source],
            dtype=np.int64,
        )
        for label_index, code in enumerate(label_codes):
            if (source, code) not in candidate_pairs:
                continue
            for model_name, probabilities in (
                ("fixed_resnet", resnet_external),
                ("logistic_reference", logistic_external),
            ):
                for row in equal_frequency_calibration_bins(
                    external_targets[indices, label_index],
                    probabilities[indices, label_index],
                    bins=bins,
                ):
                    output.append(
                        {
                            "scope": "external_certification_candidate",
                            "source": source,
                            "model": model_name,
                            "label_code": code,
                            "label_name": LABEL_NAMES[code],
                            **row,
                        }
                    )
    return output


def plot_internal_calibration(
    rows: list[dict[str, Any]],
    output_root: Path,
    label_codes: tuple[str, ...],
) -> None:
    selected = [row for row in rows if row["scope"] == "internal_fold10"]
    figure, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
    axes_flat = axes.ravel()
    for axis, code in zip(axes_flat, label_codes, strict=False):
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Ideal")
        for model_name, marker in (("fixed_resnet", "o"), ("logistic_reference", "s")):
            subset = [
                row
                for row in selected
                if row["label_code"] == code and row["model"] == model_name
            ]
            subset.sort(key=lambda row: int(row["bin"]))
            x = [float(row["mean_predicted_probability"]) for row in subset]
            y = [float(row["observed_prevalence"]) for row in subset]
            low = [float(row["observed_wilson_low"]) for row in subset]
            high = [float(row["observed_wilson_high"]) for row in subset]
            errors = np.vstack([np.asarray(y) - np.asarray(low), np.asarray(high) - np.asarray(y)])
            axis.errorbar(
                x,
                y,
                yerr=errors,
                marker=marker,
                linewidth=1.3,
                capsize=2,
                label=model_name,
            )
        axis.set_title(LABEL_NAMES[code])
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed prevalence")
        axis.grid(alpha=0.25)
    if len(label_codes) < len(axes_flat):
        axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.suptitle("Internal fold-10 calibration: fixed ResNet versus logistic reference")
    figure.savefig(output_root / "internal_calibration.png", dpi=300)
    figure.savefig(output_root / "internal_calibration.pdf")
    plt.close(figure)


def plot_external_candidate_calibration(
    rows: list[dict[str, Any]],
    output_root: Path,
    candidate_pairs: list[tuple[str, str]],
) -> None:
    selected = [row for row in rows if row["scope"] == "external_certification_candidate"]
    figure, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
    axes_flat = axes.ravel()
    for axis, (source, code) in zip(axes_flat, candidate_pairs, strict=False):
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Ideal")
        for model_name, marker in (("fixed_resnet", "o"), ("logistic_reference", "s")):
            subset = [
                row
                for row in selected
                if row["source"] == source
                and row["label_code"] == code
                and row["model"] == model_name
            ]
            subset.sort(key=lambda row: int(row["bin"]))
            axis.plot(
                [float(row["mean_predicted_probability"]) for row in subset],
                [float(row["observed_prevalence"]) for row in subset],
                marker=marker,
                linewidth=1.3,
                label=model_name,
            )
        axis.set_title(f"{source}: {LABEL_NAMES[code]}")
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed prevalence")
        axis.grid(alpha=0.25)
    for axis in axes_flat[len(candidate_pairs) :]:
        axis.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.suptitle("External calibration-recovery candidates before local recalibration")
    figure.savefig(output_root / "external_candidate_calibration.png", dpi=300)
    figure.savefig(output_root / "external_candidate_calibration.pdf")
    plt.close(figure)


def plot_internal_pr_auc_differences(
    rows: list[dict[str, Any]],
    output_root: Path,
) -> None:
    selected = [row for row in rows if row["metric"] == "pr_auc"]
    medians = np.asarray([float(row["paired_improvement_median"]) for row in selected])
    lows = np.asarray([float(row["paired_improvement_q025"]) for row in selected])
    highs = np.asarray([float(row["paired_improvement_q975"]) for row in selected])
    positions = np.arange(len(selected))
    figure, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    axis.axvline(0.0, linestyle="--", linewidth=1)
    axis.errorbar(
        medians,
        positions,
        xerr=np.vstack([medians - lows, highs - medians]),
        fmt="o",
        capsize=3,
    )
    axis.set_yticks(positions, [str(row["label_name"]) for row in selected])
    axis.set_xlabel("Paired PR-AUC difference (ResNet − Logistic)")
    axis.set_title("Internal fold-10 paired bootstrap differences")
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(output_root / "internal_paired_pr_auc_differences.png", dpi=300)
    figure.savefig(output_root / "internal_paired_pr_auc_differences.pdf")
    plt.close(figure)


def plot_phase1_recovery(phase1_report: dict[str, Any], output_root: Path) -> None:
    pairs = list(phase1_report["pair_results"])
    figure, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
    axes_flat = axes.ravel()
    for axis, pair_key in zip(axes_flat, pairs, strict=False):
        pair = phase1_report["pair_results"][pair_key]
        budgets = sorted(int(value) for value in pair["budgets"])
        for method, marker in (
            ("frozen_no_update", "o"),
            ("intercept_only_recalibration", "s"),
            ("platt_recalibration", "^"),
        ):
            valid_budgets: list[int] = []
            rates: list[float] = []
            for budget in budgets:
                summary = pair["budgets"][str(budget)]["methods"][method]
                rate = summary["recovery_envelope_met_rate_among_estimable"]
                if rate is not None:
                    valid_budgets.append(budget)
                    rates.append(float(rate))
            axis.plot(valid_budgets, rates, marker=marker, linewidth=1.3, label=method)
        axis.set_title(f"{pair['source']}: {LABEL_NAMES[pair['label_code']]}")
        axis.set_xscale("symlog", linthresh=50)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xlabel("Target labels")
        axis.set_ylabel("Recovery-envelope success rate")
        axis.grid(alpha=0.25)
    for axis in axes_flat[len(pairs) :]:
        axis.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.suptitle("Matched Phase-1 probability-recovery experiment")
    figure.savefig(output_root / "phase1_recovery_curves.png", dpi=300)
    figure.savefig(output_root / "phase1_recovery_curves.pdf")
    plt.close(figure)


def write_summary(
    *,
    output_root: Path,
    model_results: dict[str, Any],
    phase1_rows: list[dict[str, Any]],
    bootstrap_repeats: int,
) -> None:
    macro = model_results["internal"]["macro"]
    pr_delta = macro["paired_improvement"]["pr_auc"]
    brier_delta = macro["paired_improvement"]["brier"]
    significant_phase1 = sum(
        row["q_value_bh"] is not None
        and float(row["q_value_bh"]) < 0.05
        and row["outcome"] == "recovery_envelope_success"
        for row in phase1_rows
    )
    lines = [
        "# TRUST-ECG aggregate-only statistical addendum",
        "",
        f"- Bootstrap replicates: **{bootstrap_repeats}**",
        "- Bootstrap unit: paired record resampling in memory; no record-level output persisted.",
        "- Candidate model: fixed ResNet; reference: locked handcrafted Logistic Regression.",
        "",
        "## Internal paired comparison",
        "",
        (
            "- Macro PR-AUC improvement (ResNet − Logistic): "
            f"median **{float(pr_delta['median']):.4f}**, "
            f"95% interval [{float(pr_delta['q025']):.4f}, {float(pr_delta['q975']):.4f}]."
        ),
        (
            "- Macro Brier improvement (Logistic − ResNet; positive favors ResNet): "
            f"median **{float(brier_delta['median']):.4f}**, "
            f"95% interval [{float(brier_delta['q025']):.4f}, "
            f"{float(brier_delta['q975']):.4f}]."
        ),
        "",
        "## Phase-1 matched comparisons",
        "",
        (
            "- Recovery-envelope comparisons significant after global BH correction: "
            f"**{significant_phase1}**."
        ),
        "",
        "## Privacy",
        "",
        "Only aggregate intervals, paired differences, corrected significance values, calibration-bin counts, and figures are written. Raw predictions, logits, identifiers, waveforms, sampled indices, and checkpoints are not included.",
        "",
    ]
    (output_root / "statistical_addendum_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
