#!/usr/bin/env python3
"""Run real-data secondary TRUST-ECG audit, XAI, or robustness analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

from trust_icu import ecg_statistical_models as models
from trust_icu.ecg_baseline import (
    certify_label_domain_pair,
    evaluate_binary_probabilities,
)
from trust_icu.ecg_data import EXPECTED_LEADS
from trust_icu.ecg_manifest import load_and_verify_label_manifest
from trust_icu.ecg_phase0_v04 import (
    load_and_verify_model_index,
    load_and_verify_normalization_stats,
    load_standardized_record,
)
from trust_icu.ecg_phase1 import load_and_verify_phase0_report
from trust_icu.ecg_protocol import load_open_ecg_protocol
from trust_icu.ecg_secondary_overlap import (
    compact_morphology_fingerprint,
    exact_signal_sha256,
)
from trust_icu.ecg_secondary_robustness import robustness_suite
from trust_icu.ecg_secondary_uncertainty import (
    bootstrap_gate_uncertainty,
    parse_header_demographics,
)
from trust_icu.ecg_secondary_xai import (
    deterministic_coverage_indices,
    grad_cam_1d,
    integrated_gradients,
    occlude_lead,
    occlude_time_window,
)
from trust_icu.ecg_signal import normalize_signal
from trust_icu.ecg_statistical_core import write_csv
from trust_icu.ecg_statistical_reconstruction import apply_phase0_calibration_payload

PRIMARY_SEED = 20260808
EXTERNAL_SOURCES = ("georgia", "cpsc_2018", "cpsc_2018_extra")
LABEL_NAMES = {
    "59118001": "RBBB",
    "164889003": "AF",
    "164909002": "LBBB",
    "270492004": "IAVB",
    "284470004": "PAC",
    "426783006": "NSR",
    "427084000": "STach",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "xai", "robustness"), required=True)
    parser.add_argument("--phase0-report", required=True)
    parser.add_argument("--primary-data-root", required=True)
    parser.add_argument("--model-index", required=True)
    parser.add_argument("--model-index-audit", required=True)
    parser.add_argument("--label-manifest", required=True)
    parser.add_argument("--normalization-stats", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--global-calibration", required=True)
    parser.add_argument("--protocol", default="schemas/open_ecg_protocol.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(output_root: Path, *, primary: dict[str, str], mode: str) -> None:
    files: dict[str, str] = {}
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name != "secondary_sha256_manifest.json":
            files[path.name] = _sha256(path)
    _json_write(
        output_root / "secondary_sha256_manifest.json",
        {
            "study": "TRUST-ECG",
            "mode": mode,
            "secondary_evidence_only": True,
            "primary_analysis_unchanged": True,
            "primary_provenance": primary,
            "files": files,
        },
    )


def _load_state(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cpu":
        raise SystemExit("Secondary TRUST-ECG execution is locked to CPU.")
    protocol_path = Path(args.protocol).expanduser().resolve()
    report_path = Path(args.phase0_report).expanduser().resolve()
    report = load_and_verify_phase0_report(report_path, protocol_path)
    protocol = load_open_ecg_protocol(protocol_path)
    manifest = load_and_verify_label_manifest(args.label_manifest)
    rows, index_audit = load_and_verify_model_index(
        index_csv=args.model_index,
        index_audit_path=args.model_index_audit,
    )
    stats = load_and_verify_normalization_stats(args.normalization_stats)
    label_codes = tuple(str(code) for code in report["label_codes"])
    if tuple(str(code) for code in index_audit["label_codes"]) != label_codes:
        raise RuntimeError("Model-index label order differs from the primary report.")
    if str(index_audit["index_sha256"]) != str(report["model_index_sha256"]):
        raise RuntimeError("Model-index SHA differs from the primary report.")
    if str(manifest["manifest_sha256"]) != str(report["label_manifest_sha256"]):
        raise RuntimeError("Label-manifest SHA differs from the primary report.")

    device = torch.device("cpu")
    model, calibration = models.load_verified_resnet(
        checkpoint_path=Path(args.checkpoint).expanduser().resolve(),
        calibration_path=Path(args.global_calibration).expanduser().resolve(),
        report=report,
        normalization_stats=stats,
        label_codes=label_codes,
        device=device,
    )
    models._global_calibrated_probabilities = apply_phase0_calibration_payload
    return {
        "protocol": protocol,
        "report": report,
        "manifest": manifest,
        "rows": rows,
        "index_audit": index_audit,
        "stats": stats,
        "label_codes": label_codes,
        "device": device,
        "model": model,
        "calibration": calibration,
        "data_root": Path(args.primary_data_root).expanduser().resolve(),
        "checkpoint_path": Path(args.checkpoint).expanduser().resolve(),
        "protocol_path": protocol_path,
        "report_path": report_path,
    }


def _provenance(state: dict[str, Any]) -> dict[str, str]:
    report = state["report"]
    return {
        "protocol_sha256": str(report["protocol_sha256"]),
        "phase0_report_sha256": str(report["report_sha256"]),
        "model_sha256": str(report["model_sha256"]),
        "model_index_sha256": str(report["model_index_sha256"]),
        "label_manifest_sha256": str(report["label_manifest_sha256"]),
        "normalization_stats_sha256": str(state["stats"].stats_sha256),
    }


def _groups(rows: list[Any]) -> dict[str, list[Any]]:
    groups = {
        "ptb-xl": [row for row in rows if row.role == "internal_test"],
    }
    for source in EXTERNAL_SOURCES:
        groups[source] = [
            row for row in rows
            if row.source == source and row.role == "external_certification"
        ]
    if any(not value for value in groups.values()):
        raise RuntimeError("A required internal/external secondary evaluation group is empty.")
    return groups


def _clean_group_probabilities(
    state: dict[str, Any],
    rows: list[Any],
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = int(state["protocol"]["phase0_models"]["resnet1d_fixed"]["batch_size"])
    return models.collect_resnet_probabilities(
        model=state["model"],
        rows=rows,
        data_root=state["data_root"],
        normalization_stats=state["stats"],
        calibration_payload=state["calibration"],
        label_codes=state["label_codes"],
        batch_size=batch_size,
        device=state["device"],
        num_workers=0,
    )


def _normalized_tensor(state: dict[str, Any], rows: list[Any]) -> torch.Tensor:
    parts: list[np.ndarray] = []
    for row in rows:
        standardized = load_standardized_record(state["data_root"], row)
        parts.append(
            normalize_signal(
                standardized.waveform_mv,
                standardized.valid_mask,
                state["stats"],
            )
        )
    return torch.from_numpy(np.stack(parts).astype(np.float32, copy=False))


def _predict_tensor(state: dict[str, Any], tensor: torch.Tensor, *, batch_size: int = 16) -> np.ndarray:
    logits: list[np.ndarray] = []
    model = state["model"]
    model.eval()
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            outputs = model(tensor[start : start + batch_size].to(dtype=torch.float32))
            logits.append(outputs.detach().cpu().numpy())
    raw = np.vstack(logits).astype(np.float64, copy=False)
    return apply_phase0_calibration_payload(raw, state["calibration"], state["label_codes"])


def _safe_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any] | None:
    if targets.ndim != 1 or probabilities.shape != targets.shape:
        raise ValueError("Metric inputs are not aligned.")
    positives = int(targets.sum())
    negatives = int(targets.size - positives)
    if positives == 0 or negatives == 0:
        return None
    return asdict(evaluate_binary_probabilities(targets, probabilities))


def _run_certification_uncertainty(
    state: dict[str, Any],
    clean: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(EXTERNAL_SOURCES):
        probabilities, targets = clean[source]
        for label_index, code in enumerate(state["label_codes"]):
            result = bootstrap_gate_uncertainty(
                targets[:, label_index],
                probabilities[:, label_index],
                repeats=repeats,
                seed=PRIMARY_SEED + source_index * 100 + label_index,
            )
            rows.append(
                {
                    "source": source,
                    "label_code": code,
                    "label_name": LABEL_NAMES[code],
                    "official_status": state["report"]["external_certification"][source][code]["status"],
                    **result,
                }
            )
    return rows


def _header_metadata(data_root: Path, row: Any) -> dict[str, float | str | None]:
    header_path = (data_root / Path(row.relative_header_path)).resolve()
    if data_root not in header_path.parents or not header_path.is_file():
        return {"age": None, "sex": None, "age_band": None}
    return parse_header_demographics(header_path.read_text(encoding="utf-8", errors="replace"))


def _subgroup_rows(
    state: dict[str, Any],
    groups: dict[str, list[Any]],
    clean: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for source, source_rows in groups.items():
        metadata = [_header_metadata(state["data_root"], row) for row in source_rows]
        for field in ("sex", "age_band"):
            known = [item[field] is not None for item in metadata]
            coverage_rows.append(
                {
                    "source": source,
                    "field": field,
                    "records": len(metadata),
                    "known_records": int(sum(known)),
                    "coverage_fraction": float(sum(known) / len(metadata)),
                }
            )
            values = sorted({str(item[field]) for item in metadata if item[field] is not None})
            probabilities, targets = clean[source]
            for value in values:
                mask = np.asarray([item[field] == value for item in metadata], dtype=bool)
                for label_index, code in enumerate(state["label_codes"]):
                    y = targets[mask, label_index]
                    p = probabilities[mask, label_index]
                    positives = int(y.sum())
                    negatives = int(y.size - positives)
                    if positives < 50 or negatives < 50:
                        result_rows.append(
                            {
                                "source": source,
                                "subgroup_field": field,
                                "subgroup": value,
                                "label_code": code,
                                "label_name": LABEL_NAMES[code],
                                "status": "not_estimable_support_below_50_50",
                                "n": int(y.size),
                                "positives": positives,
                                "negatives": negatives,
                            }
                        )
                        continue
                    metrics = asdict(evaluate_binary_probabilities(y, p))
                    result_rows.append(
                        {
                            "source": source,
                            "subgroup_field": field,
                            "subgroup": value,
                            "label_code": code,
                            "label_name": LABEL_NAMES[code],
                            "status": "estimable",
                            **metrics,
                        }
                    )
    return result_rows, coverage_rows


def _overlap_source(state: dict[str, Any], source: str) -> dict[str, Any]:
    cert_rows = [
        row for row in state["rows"]
        if row.source == source and row.role == "external_certification"
    ]
    recovery_rows = [
        row for row in state["rows"]
        if row.source == source and row.role == "external_recovery_pool"
    ]
    if not cert_rows or not recovery_rows:
        raise RuntimeError(f"Overlap audit partitions are empty for {source}.")

    cert_hashes: set[str] = set()
    cert_fp: list[np.ndarray] = []
    for row in cert_rows:
        signal = load_standardized_record(state["data_root"], row).waveform_mv
        cert_hashes.add(exact_signal_sha256(signal))
        cert_fp.append(compact_morphology_fingerprint(signal))
    cert_matrix = np.vstack(cert_fp)

    exact = 0
    maxima: list[float] = []
    batch: list[np.ndarray] = []
    for row in recovery_rows:
        signal = load_standardized_record(state["data_root"], row).waveform_mv
        exact += int(exact_signal_sha256(signal) in cert_hashes)
        batch.append(compact_morphology_fingerprint(signal))
        if len(batch) >= 128:
            similarity = np.vstack(batch) @ cert_matrix.T
            maxima.extend(np.max(similarity, axis=1).tolist())
            batch.clear()
    if batch:
        similarity = np.vstack(batch) @ cert_matrix.T
        maxima.extend(np.max(similarity, axis=1).tolist())

    values = np.asarray(maxima, dtype=np.float64)
    threshold = 0.995
    return {
        "source": source,
        "certification_records": len(cert_rows),
        "recovery_records": len(recovery_rows),
        "exact_cross_partition_duplicates": int(exact),
        "near_duplicate_threshold": threshold,
        "near_duplicate_candidates": int(np.sum(values >= threshold)),
        "maximum_near_duplicate_similarity": float(np.max(values)),
        "q95_near_duplicate_similarity": float(np.quantile(values, 0.95)),
        "q99_near_duplicate_similarity": float(np.quantile(values, 0.99)),
        "patient_independence_claimed": False,
    }


def _compute_profile(state: dict[str, Any], sample_rows: list[Any]) -> dict[str, Any]:
    model = state["model"]
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    batch = _normalized_tensor(state, sample_rows[: min(16, len(sample_rows))])
    with torch.no_grad():
        for _ in range(2):
            model(batch)
        durations: list[float] = []
        for _ in range(8):
            started = time.perf_counter()
            model(batch)
            durations.append(time.perf_counter() - started)
    median = float(np.median(np.asarray(durations)))
    n = int(batch.shape[0])
    return {
        "parameter_count": parameters,
        "trainable_parameter_count": trainable,
        "serialized_checkpoint_bytes": int(state["checkpoint_path"].stat().st_size),
        "cpu_inference_batch_records": n,
        "cpu_inference_median_batch_seconds": median,
        "cpu_inference_records_per_second": float(n / median),
        "latency_scope": "representative_cpu_runner_secondary_measurement_not_hardware_benchmark",
    }


def _run_audit(state: dict[str, Any], output_root: Path, *, repeats: int) -> None:
    groups = _groups(state["rows"])
    clean = {source: _clean_group_probabilities(state, rows) for source, rows in groups.items()}
    uncertainty = _run_certification_uncertainty(state, clean, repeats=repeats)
    subgroups, coverage = _subgroup_rows(state, groups, clean)
    overlap = [_overlap_source(state, source) for source in EXTERNAL_SOURCES]
    compute = _compute_profile(state, groups["ptb-xl"])

    write_csv(output_root / "certification_uncertainty.csv", uncertainty)
    write_csv(output_root / "subgroup_results.csv", subgroups)
    write_csv(output_root / "subgroup_coverage.csv", coverage)
    _json_write(output_root / "duplicate_audit.json", {"sources": overlap})
    _json_write(output_root / "compute_profile.json", compute)
    _json_write(
        output_root / "audit_summary.json",
        {
            "bootstrap_repeats": repeats,
            "certification_pairs": len(uncertainty),
            "subgroup_rows": len(subgroups),
            "overlap_sources": len(overlap),
            "primary_analysis_unchanged": True,
            "secondary_evidence_only": True,
        },
    )


def _temporal_bins(values: np.ndarray, bins: int = 10) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    edges = np.linspace(0, array.shape[-1], bins + 1, dtype=int)
    return [float(np.mean(array[..., start:stop])) for start, stop in zip(edges[:-1], edges[1:], strict=True)]


def _finite_spearman(a: list[float], b: list[float]) -> float | None:
    result = spearmanr(np.asarray(a), np.asarray(b)).statistic
    return None if not np.isfinite(result) else float(result)


def _run_xai(state: dict[str, Any], output_root: Path) -> None:
    groups = _groups(state["rows"])
    label_codes = state["label_codes"]
    lead_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    gradcam_rows: list[dict[str, Any]] = []
    source_vectors: dict[str, dict[str, dict[str, list[float]]]] = {}

    for source_index, (source, source_rows) in enumerate(groups.items()):
        target_matrix = np.asarray([row.labels for row in source_rows], dtype=np.int64)
        selected = deterministic_coverage_indices(
            target_matrix,
            max_samples=min(64, len(source_rows)),
            seed=PRIMARY_SEED + source_index,
        )
        rows64 = [source_rows[int(index)] for index in selected]
        tensor64 = _normalized_tensor(state, rows64)
        base = _predict_tensor(state, tensor64)
        source_vectors[source] = {}

        lead_importance_by_label: dict[str, list[float]] = {code: [] for code in label_codes}
        for lead_index, lead_name in enumerate(EXPECTED_LEADS):
            perturbed = _predict_tensor(state, occlude_lead(tensor64, lead_index))
            drops = base - perturbed
            for label_index, code in enumerate(label_codes):
                mean_drop = float(np.mean(drops[:, label_index]))
                mean_abs = float(np.mean(np.abs(drops[:, label_index])))
                lead_importance_by_label[code].append(mean_abs)
                lead_rows.append(
                    {
                        "source": source,
                        "label_code": code,
                        "label_name": LABEL_NAMES[code],
                        "method": "lead_occlusion",
                        "lead": lead_name,
                        "sample_count": int(tensor64.shape[0]),
                        "mean_probability_drop": mean_drop,
                        "mean_absolute_probability_change": mean_abs,
                    }
                )

        for group_name, indices in (
            ("limb", list(range(0, 6))),
            ("precordial", list(range(6, 12))),
        ):
            grouped = tensor64.clone()
            grouped[:, indices, :] = 0.0
            perturbed = _predict_tensor(state, grouped)
            drops = base - perturbed
            for label_index, code in enumerate(label_codes):
                lead_rows.append(
                    {
                        "source": source,
                        "label_code": code,
                        "label_name": LABEL_NAMES[code],
                        "method": "lead_group_occlusion",
                        "lead": group_name,
                        "sample_count": int(tensor64.shape[0]),
                        "mean_probability_drop": float(np.mean(drops[:, label_index])),
                        "mean_absolute_probability_change": float(np.mean(np.abs(drops[:, label_index]))),
                    }
                )

        window_size = tensor64.shape[-1] // 10
        for window in range(10):
            start = window * window_size
            stop = tensor64.shape[-1] if window == 9 else (window + 1) * window_size
            perturbed = _predict_tensor(state, occlude_time_window(tensor64, start, stop))
            drops = base - perturbed
            for label_index, code in enumerate(label_codes):
                temporal_rows.append(
                    {
                        "source": source,
                        "label_code": code,
                        "label_name": LABEL_NAMES[code],
                        "method": "temporal_occlusion",
                        "window": window + 1,
                        "start_seconds": float(start / 500.0),
                        "stop_seconds": float(stop / 500.0),
                        "sample_count": int(tensor64.shape[0]),
                        "mean_probability_drop": float(np.mean(drops[:, label_index])),
                        "mean_absolute_probability_change": float(np.mean(np.abs(drops[:, label_index]))),
                    }
                )

        selected8 = deterministic_coverage_indices(
            target_matrix,
            max_samples=min(8, len(source_rows)),
            seed=PRIMARY_SEED + source_index,
        )
        rows8 = [source_rows[int(index)] for index in selected8]
        tensor8 = _normalized_tensor(state, rows8).to(dtype=torch.float32)

        for label_index, code in enumerate(label_codes):
            ig = integrated_gradients(
                state["model"],
                tensor8,
                target_index=label_index,
                steps=24,
            ).detach().cpu().numpy()
            ig_abs = np.abs(ig)
            ig_leads = np.mean(ig_abs, axis=(0, 2)).tolist()
            source_vectors[source].setdefault(code, {})["integrated_gradients"] = [
                float(value) for value in ig_leads
            ]
            source_vectors[source][code]["lead_occlusion"] = lead_importance_by_label[code]
            for lead_name, value in zip(EXPECTED_LEADS, ig_leads, strict=True):
                lead_rows.append(
                    {
                        "source": source,
                        "label_code": code,
                        "label_name": LABEL_NAMES[code],
                        "method": "integrated_gradients",
                        "lead": lead_name,
                        "sample_count": int(tensor8.shape[0]),
                        "mean_probability_drop": None,
                        "mean_absolute_probability_change": float(value),
                    }
                )
            ig_time = np.mean(ig_abs, axis=(0, 1))
            for window, value in enumerate(_temporal_bins(ig_time), start=1):
                temporal_rows.append(
                    {
                        "source": source,
                        "label_code": code,
                        "label_name": LABEL_NAMES[code],
                        "method": "integrated_gradients",
                        "window": window,
                        "start_seconds": float(window - 1),
                        "stop_seconds": float(window),
                        "sample_count": int(tensor8.shape[0]),
                        "mean_probability_drop": None,
                        "mean_absolute_probability_change": value,
                    }
                )

            cam = grad_cam_1d(
                state["model"],
                tensor8,
                target_index=label_index,
            ).detach().cpu().numpy()
            cam_mean = np.mean(cam, axis=0)
            for window, value in enumerate(_temporal_bins(cam_mean), start=1):
                gradcam_rows.append(
                    {
                        "source": source,
                        "label_code": code,
                        "label_name": LABEL_NAMES[code],
                        "window": window,
                        "start_seconds": float(window - 1),
                        "stop_seconds": float(window),
                        "sample_count": int(tensor8.shape[0]),
                        "mean_gradcam_activation": value,
                    }
                )

    stability: list[dict[str, Any]] = []
    for source in EXTERNAL_SOURCES:
        for code in label_codes:
            for method in ("integrated_gradients", "lead_occlusion"):
                internal = source_vectors["ptb-xl"][code][method]
                external = source_vectors[source][code][method]
                stability.append(
                    {
                        "source": source,
                        "label_code": code,
                        "label_name": LABEL_NAMES[code],
                        "method": method,
                        "spearman_lead_rank_vs_internal": _finite_spearman(internal, external),
                    }
                )

    write_csv(output_root / "xai_lead_importance.csv", lead_rows)
    write_csv(output_root / "xai_temporal_importance.csv", temporal_rows)
    write_csv(output_root / "xai_gradcam_temporal.csv", gradcam_rows)
    write_csv(output_root / "xai_attribution_stability.csv", stability)
    _json_write(
        output_root / "xai_resnet_summary.json",
        {
            "methods": [
                "integrated_gradients",
                "grad_cam_1d",
                "lead_occlusion",
                "lead_group_occlusion",
                "temporal_occlusion",
            ],
            "occlusion_sample_cap_per_source": 64,
            "ig_gradcam_sample_cap_per_source": 8,
            "integrated_gradients_steps": 24,
            "temporal_windows": 10,
            "causal_interpretation_claimed": False,
            "surrogate_model_used": False,
            "secondary_evidence_only": True,
        },
    )
    _plot_xai_heatmap(lead_rows, output_root)


def _plot_xai_heatmap(rows: list[dict[str, Any]], output_root: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row for row in rows
        if row["method"] == "lead_occlusion" and row["lead"] in EXPECTED_LEADS
    ]
    order = ["ptb-xl", *EXTERNAL_SOURCES]
    matrix: list[list[float]] = []
    ylabels: list[str] = []
    for source in order:
        for code, label in LABEL_NAMES.items():
            values = [
                float(row["mean_absolute_probability_change"])
                for row in selected
                if row["source"] == source and row["label_code"] == code
            ]
            if len(values) == 12:
                matrix.append(values)
                ylabels.append(f"{source}:{label}")
    fig, ax = plt.subplots(figsize=(9.2, 9.0))
    image = ax.imshow(np.asarray(matrix), aspect="auto")
    ax.set_xticks(range(12), EXPECTED_LEADS, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=7)
    ax.set_title("TRUST-ECG lead-occlusion sensitivity (secondary)")
    fig.colorbar(image, ax=ax, label="Mean absolute probability change")
    fig.tight_layout()
    fig.savefig(output_root / "xai_lead_importance_heatmap.png", dpi=300)
    fig.savefig(output_root / "xai_lead_importance_heatmap.pdf")
    plt.close(fig)


def _record_noise_seed(row: Any) -> int:
    material = f"{PRIMARY_SEED}|{row.source}|{row.record_id}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**32)


def _run_robustness(state: dict[str, Any], output_root: Path) -> None:
    groups = _groups(state["rows"])
    label_codes = state["label_codes"]
    perturbation_names = [
        "clean",
        "gaussian_noise_snr20db",
        "gain_0p9x",
        "gain_1p1x",
        "baseline_wander_0p05mv_0p33hz",
        "time_shift_minus_100ms",
        "time_shift_plus_100ms",
        "limb_lead_dropout",
        "precordial_lead_dropout",
    ]
    rows_out: list[dict[str, Any]] = []

    for source, source_rows in groups.items():
        probabilities: dict[str, list[np.ndarray]] = {name: [] for name in perturbation_names}
        targets_parts: list[np.ndarray] = []
        batch_size = 16
        for start in range(0, len(source_rows), batch_size):
            batch_rows = source_rows[start : start + batch_size]
            tensors: dict[str, list[np.ndarray]] = {name: [] for name in perturbation_names}
            for row in batch_rows:
                standardized = load_standardized_record(state["data_root"], row)
                physical = standardized.waveform_mv
                valid = standardized.valid_mask
                tensors["clean"].append(normalize_signal(physical, valid, state["stats"]))
                suite = robustness_suite(physical, seed=_record_noise_seed(row))
                for name, perturbed in suite.items():
                    tensors[name].append(normalize_signal(perturbed, valid, state["stats"]))
            targets_parts.append(np.asarray([row.labels for row in batch_rows], dtype=np.int64))
            for name in perturbation_names:
                tensor = torch.from_numpy(np.stack(tensors[name]).astype(np.float32, copy=False))
                probabilities[name].append(_predict_tensor(state, tensor, batch_size=batch_size))
        targets = np.vstack(targets_parts)
        stacked = {name: np.vstack(parts) for name, parts in probabilities.items()}

        clean_metrics_by_code: dict[str, dict[str, Any] | None] = {}
        for label_index, code in enumerate(label_codes):
            clean_metrics_by_code[code] = _safe_metrics(
                targets[:, label_index],
                stacked["clean"][:, label_index],
            )

        for name in perturbation_names:
            for label_index, code in enumerate(label_codes):
                metrics = _safe_metrics(targets[:, label_index], stacked[name][:, label_index])
                if metrics is None:
                    rows_out.append(
                        {
                            "source": source,
                            "perturbation": name,
                            "label_code": code,
                            "label_name": LABEL_NAMES[code],
                            "status": "not_estimable_single_class",
                        }
                    )
                    continue
                clean_metrics = clean_metrics_by_code[code]
                assert clean_metrics is not None
                certification_status = None
                if source != "ptb-xl":
                    certification_status = certify_label_domain_pair(
                        targets[:, label_index],
                        stacked[name][:, label_index],
                    ).status
                rows_out.append(
                    {
                        "source": source,
                        "perturbation": name,
                        "label_code": code,
                        "label_name": LABEL_NAMES[code],
                        "status": "estimable",
                        "certification_status": certification_status,
                        **metrics,
                        "delta_pr_auc_vs_clean": float(metrics["pr_auc"] - clean_metrics["pr_auc"]),
                        "delta_roc_auc_vs_clean": float(metrics["roc_auc"] - clean_metrics["roc_auc"]),
                        "delta_brier_vs_clean": float(metrics["brier"] - clean_metrics["brier"]),
                        "delta_calibration_slope_vs_clean": float(
                            metrics["calibration_slope"] - clean_metrics["calibration_slope"]
                        ),
                        "delta_calibration_intercept_vs_clean": float(
                            metrics["calibration_intercept"] - clean_metrics["calibration_intercept"]
                        ),
                    }
                )
        print(f"robustness_complete source={source} records={len(source_rows)}", flush=True)

    write_csv(output_root / "robustness.csv", rows_out)
    _json_write(
        output_root / "robustness_summary.json",
        {
            "perturbations": perturbation_names[1:],
            "primary_model_frozen": True,
            "retraining": False,
            "recalibration": False,
            "secondary_evidence_only": True,
        },
    )
    _plot_robustness_heatmap(rows_out, output_root)


def _plot_robustness_heatmap(rows: list[dict[str, Any]], output_root: Path) -> None:
    import matplotlib.pyplot as plt

    perturbations = [
        "gaussian_noise_snr20db",
        "gain_0p9x",
        "gain_1p1x",
        "baseline_wander_0p05mv_0p33hz",
        "time_shift_minus_100ms",
        "time_shift_plus_100ms",
        "limb_lead_dropout",
        "precordial_lead_dropout",
    ]
    sources = ["ptb-xl", *EXTERNAL_SOURCES]
    matrix: list[list[float]] = []
    for source in sources:
        row_values: list[float] = []
        for perturbation in perturbations:
            values = [
                float(item["delta_pr_auc_vs_clean"])
                for item in rows
                if item.get("source") == source
                and item.get("perturbation") == perturbation
                and item.get("status") == "estimable"
            ]
            row_values.append(float(np.mean(values)) if values else 0.0)
        matrix.append(row_values)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    image = ax.imshow(np.asarray(matrix), aspect="auto")
    ax.set_xticks(range(len(perturbations)), perturbations, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(sources)), sources)
    ax.set_title("Macro PR-AUC change under fixed signal perturbations (secondary)")
    fig.colorbar(image, ax=ax, label="Mean ΔPR-AUC vs clean")
    fig.tight_layout()
    fig.savefig(output_root / "robustness_pr_auc_heatmap.png", dpi=300)
    fig.savefig(output_root / "robustness_pr_auc_heatmap.pdf")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.bootstrap_repeats < 100:
        raise SystemExit("--bootstrap-repeats must be at least 100.")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state = _load_state(args)
    primary = _provenance(state)

    if args.mode == "audit":
        _run_audit(state, output_root, repeats=args.bootstrap_repeats)
    elif args.mode == "xai":
        _run_xai(state, output_root)
    elif args.mode == "robustness":
        _run_robustness(state, output_root)
    else:  # pragma: no cover
        raise RuntimeError(args.mode)

    _write_manifest(output_root, primary=primary, mode=args.mode)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "output_root": str(output_root),
                "primary_analysis_unchanged": True,
                "secondary_evidence_only": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
