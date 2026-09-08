"""Dependency-light aggregate reporting helpers for TRUST-ECG."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
