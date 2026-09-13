#!/usr/bin/env python3
"""Run model-dependent TRUST-ECG secondary analyses on predeclared secondary seeds.

The historical 20260808 primary result remains frozen and report-only. This runner
never substitutes a secondary model for the historical primary checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np

from trust_icu.ecg_data import EXPECTED_LEADS
from trust_icu.ecg_statistical_core import write_csv

HISTORICAL_PRIMARY = {
    "seed": 20260808,
    "model_sha256": "1dc5a0ebbd7126ac8cd214c02e1faae456b5a8f0dd05f90f20877d7e8b749358",
    "report_sha256": "9287e6be55ac86885f688fed86929732e953e5f4dc6f728098b368e7d0333ad9",
    "epochs_completed": 33,
    "best_epoch": 26,
}

SEED_EXPECTATIONS = {
    20260809: {
        "model_sha256": "7a91bb6c0701ce5aa8d2ba66c4821782288b7c347d2050cd6646db111037eb19",
        "report_sha256": "af4e92f90695df6104d03aa004ab991c566ac290987e54c386bf5b5ee407b299",
        "protocol_sha256": "79a7785af998d9c32ae0fba44b2eab05fa0a62f6b5f27dff24f27dcec8f00cbb",
        "epochs_completed": 31,
        "best_epoch": 24,
    },
    20260810: {
        "model_sha256": "0cb57b6bcc7c124666d10def9c83cadc0260846392d81102174d14e6d426ce95",
        "report_sha256": "31a281e8b77504038826acb5ad4be6b2e000aac7335582041f20120606836493",
        "protocol_sha256": "bb7655ab3e70d55961cc78b275581b100b7d82c1e36dab954d4da40872fa9144",
        "epochs_completed": 38,
        "best_epoch": 31,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("audit", "full_xai", "reproducibility_xai", "sampled_robustness"),
    )
    parser.add_argument("--analysis-seed", required=True, type=int, choices=tuple(SEED_EXPECTATIONS))
    parser.add_argument("--phase0-report", required=True)
    parser.add_argument("--primary-data-root", required=True)
    parser.add_argument("--model-index", required=True)
    parser.add_argument("--model-index-audit", required=True)
    parser.add_argument("--label-manifest", required=True)
    parser.add_argument("--normalization-stats", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--global-calibration", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--robustness-sample-cap", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _load_legacy_runner() -> ModuleType:
    path = Path(__file__).with_name("run_open_ecg_secondary_waveform.py")
    spec = importlib.util.spec_from_file_location("trust_ecg_secondary_waveform_legacy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        mode="xai",
        phase0_report=args.phase0_report,
        primary_data_root=args.primary_data_root,
        model_index=args.model_index,
        model_index_audit=args.model_index_audit,
        label_manifest=args.label_manifest,
        normalization_stats=args.normalization_stats,
        checkpoint=args.checkpoint,
        global_calibration=args.global_calibration,
        protocol=args.protocol,
        output_root=args.output_root,
        bootstrap_repeats=args.bootstrap_repeats,
        device=args.device,
    )


def _verify_analysis_seed(state: dict[str, Any], seed: int) -> dict[str, Any]:
    expected = SEED_EXPECTATIONS[seed]
    report = state["report"]
    observed = {
        "model_sha256": str(report["model_sha256"]),
        "report_sha256": str(report["report_sha256"]),
        "protocol_sha256": str(report["protocol_sha256"]),
    }
    for key, want in expected.items():
        if key in observed and observed[key] != want:
            raise RuntimeError(
                f"Secondary seed {seed} identity mismatch for {key}: "
                f"expected {want}, observed {observed[key]}"
            )
    return expected


def _analysis_provenance(state: dict[str, Any], seed: int) -> dict[str, Any]:
    report = state["report"]
    expected = SEED_EXPECTATIONS[seed]
    return {
        "analysis_seed": seed,
        "analysis_model_role": "predeclared_secondary_resnet_sensitivity_model",
        "model_sha256": str(report["model_sha256"]),
        "phase0_report_sha256": str(report["report_sha256"]),
        "protocol_sha256": str(report["protocol_sha256"]),
        "model_index_sha256": str(report["model_index_sha256"]),
        "label_manifest_sha256": str(report["label_manifest_sha256"]),
        "normalization_stats_sha256": str(state["stats"].stats_sha256),
        "epochs_completed": expected["epochs_completed"],
        "best_epoch": expected["best_epoch"],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(output_root: Path, *, state: dict[str, Any], seed: int, mode: str) -> None:
    files: dict[str, str] = {}
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name != "secondary_sha256_manifest.json":
            files[path.name] = _sha256(path)
    payload = {
        "study": "TRUST-ECG",
        "mode": mode,
        "secondary_evidence_only": True,
        "historical_primary_unchanged": True,
        "historical_primary_checkpoint_available": False,
        "primary_substitution": False,
        "historical_primary": HISTORICAL_PRIMARY,
        "analysis_model_provenance": _analysis_provenance(state, seed),
        "files": files,
    }
    (output_root / "secondary_sha256_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sample_state_for_robustness(
    legacy: ModuleType,
    state: dict[str, Any],
    *,
    seed: int,
    cap: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    if cap < 64:
        raise ValueError("robustness sample cap must be at least 64 per source")
    groups = legacy._groups(state["rows"])
    selected_rows: list[Any] = []
    counts: dict[str, int] = {}
    for source_index, (source, rows) in enumerate(groups.items()):
        labels = np.asarray([row.labels for row in rows], dtype=np.int64)
        indices = legacy.deterministic_coverage_indices(
            labels,
            max_samples=min(cap, len(rows)),
            seed=seed + 1000 + source_index,
        )
        chosen = [rows[int(index)] for index in indices]
        selected_rows.extend(chosen)
        counts[source] = len(chosen)
    sampled = dict(state)
    sampled["rows"] = selected_rows
    return sampled, counts


def _run_reproducibility_xai(
    legacy: ModuleType,
    state: dict[str, Any],
    output_root: Path,
    *,
    seed: int,
) -> None:
    groups = legacy._groups(state["rows"])
    label_codes = state["label_codes"]
    rows_out: list[dict[str, Any]] = []
    agreement: list[dict[str, Any]] = []

    for source_index, (source, source_rows) in enumerate(groups.items()):
        target_matrix = np.asarray([row.labels for row in source_rows], dtype=np.int64)
        selected64 = legacy.deterministic_coverage_indices(
            target_matrix,
            max_samples=min(64, len(source_rows)),
            seed=seed + source_index,
        )
        rows64 = [source_rows[int(index)] for index in selected64]
        tensor64 = legacy._normalized_tensor(state, rows64)
        base = legacy._predict_tensor(state, tensor64)
        lead_vectors: dict[str, list[float]] = {code: [] for code in label_codes}

        for lead_index, lead_name in enumerate(EXPECTED_LEADS):
            perturbed = legacy._predict_tensor(state, legacy.occlude_lead(tensor64, lead_index))
            drops = base - perturbed
            for label_index, code in enumerate(label_codes):
                mean_abs = float(np.mean(np.abs(drops[:, label_index])))
                lead_vectors[code].append(mean_abs)
                rows_out.append(
                    {
                        "source": source,
                        "label_code": code,
                        "label_name": legacy.LABEL_NAMES[code],
                        "method": "lead_occlusion",
                        "lead": lead_name,
                        "sample_count": int(tensor64.shape[0]),
                        "mean_probability_drop": float(np.mean(drops[:, label_index])),
                        "mean_absolute_probability_change": mean_abs,
                    }
                )

        selected8 = legacy.deterministic_coverage_indices(
            target_matrix,
            max_samples=min(8, len(source_rows)),
            seed=seed + source_index,
        )
        rows8 = [source_rows[int(index)] for index in selected8]
        tensor8 = legacy._normalized_tensor(state, rows8).to(dtype=legacy.torch.float32)
        for label_index, code in enumerate(label_codes):
            ig = legacy.integrated_gradients(
                state["model"], tensor8, target_index=label_index, steps=24
            ).detach().cpu().numpy()
            ig_leads = np.mean(np.abs(ig), axis=(0, 2)).tolist()
            for lead_name, value in zip(EXPECTED_LEADS, ig_leads, strict=True):
                rows_out.append(
                    {
                        "source": source,
                        "label_code": code,
                        "label_name": legacy.LABEL_NAMES[code],
                        "method": "integrated_gradients",
                        "lead": lead_name,
                        "sample_count": int(tensor8.shape[0]),
                        "mean_probability_drop": None,
                        "mean_absolute_probability_change": float(value),
                    }
                )
            agreement.append(
                {
                    "source": source,
                    "label_code": code,
                    "label_name": legacy.LABEL_NAMES[code],
                    "spearman_ig_vs_lead_occlusion": legacy._finite_spearman(
                        [float(value) for value in ig_leads],
                        lead_vectors[code],
                    ),
                }
            )

    write_csv(output_root / "xai_lead_importance.csv", rows_out)
    write_csv(output_root / "xai_method_agreement.csv", agreement)
    (output_root / "xai_resnet_summary.json").write_text(
        json.dumps(
            {
                "analysis_seed": seed,
                "profile": "reproducibility",
                "methods": ["integrated_gradients", "lead_occlusion"],
                "occlusion_sample_cap_per_source": 64,
                "integrated_gradients_sample_cap_per_source": 8,
                "integrated_gradients_steps": 24,
                "causal_interpretation_claimed": False,
                "surrogate_model_used": False,
                "secondary_evidence_only": True,
                "historical_primary_unchanged": True,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.device != "cpu":
        raise SystemExit("Secondary TRUST-ECG execution is locked to CPU.")
    if args.mode in {"audit", "full_xai", "sampled_robustness"} and args.analysis_seed != 20260809:
        raise SystemExit(f"{args.mode} is predeclared on seed 20260809 only.")
    if args.mode == "reproducibility_xai" and args.analysis_seed != 20260810:
        raise SystemExit("reproducibility_xai is predeclared on seed 20260810 only.")
    if args.bootstrap_repeats < 100:
        raise SystemExit("--bootstrap-repeats must be at least 100.")

    legacy = _load_legacy_runner()
    state = legacy._load_state(_legacy_args(args))
    _verify_analysis_seed(state, args.analysis_seed)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "audit":
        legacy._run_audit(state, output_root, repeats=args.bootstrap_repeats)
    elif args.mode == "full_xai":
        legacy._run_xai(state, output_root)
        summary_path = output_root / "xai_resnet_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "analysis_seed": args.analysis_seed,
                "profile": "full_secondary_xai",
                "historical_primary_unchanged": True,
                "primary_substitution": False,
            }
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    elif args.mode == "reproducibility_xai":
        _run_reproducibility_xai(
            legacy, state, output_root, seed=args.analysis_seed
        )
    elif args.mode == "sampled_robustness":
        sampled_state, counts = _sample_state_for_robustness(
            legacy,
            state,
            seed=args.analysis_seed,
            cap=args.robustness_sample_cap,
        )
        legacy._run_robustness(sampled_state, output_root)
        summary_path = output_root / "robustness_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.pop("primary_model_frozen", None)
        summary.update(
            {
                "analysis_seed": args.analysis_seed,
                "analysis_model_frozen": True,
                "profile": "deterministic_bounded_secondary_stress_test",
                "sample_cap_per_source": args.robustness_sample_cap,
                "sample_counts": counts,
                "historical_primary_unchanged": True,
                "primary_substitution": False,
            }
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    else:  # pragma: no cover
        raise RuntimeError(args.mode)

    _write_manifest(
        output_root,
        state=state,
        seed=args.analysis_seed,
        mode=args.mode,
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "analysis_seed": args.analysis_seed,
                "output_root": str(output_root),
                "historical_primary_unchanged": True,
                "primary_substitution": False,
                "secondary_evidence_only": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
