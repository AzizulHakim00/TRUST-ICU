#!/usr/bin/env python3
"""Run the locked TRUST-ECG ResNet across resumable CPU execution segments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from trust_icu import ecg_deep_phase0 as deep
from trust_icu.ecg_baseline import apply_platt_calibrators, fit_platt_calibrators
from trust_icu.ecg_manifest import load_and_verify_label_manifest
from trust_icu.ecg_phase0 import (
    _external_certification,
    _finalize_report,
    _metrics_for_multilabel,
    _sha256_file,
)
from trust_icu.ecg_phase0_v04 import (
    load_and_verify_model_index,
    load_standardized_record,
    write_phase0_report,
)
from trust_icu.ecg_protocol import load_open_ecg_protocol, validate_open_ecg_protocol
from trust_icu.ecg_resnet import (
    FixedResNet1D,
    ResNet1DContract,
    compute_positive_class_weights,
    macro_pr_auc_from_logits,
    set_torch_determinism,
)
from trust_icu.ecg_waveform import load_and_verify_waveform_audit

try:
    import torch
    from torch import nn
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("PyTorch is required for resumable TRUST-ECG execution.") from exc

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "schemas/open_ecg_protocol.yaml"
RESUME_VERSION = "0.1.0"
PHASE_TRAIN = "train"
PHASE_OPTIMIZATION = "optimization_inference"
PHASE_CALIBRATION = "calibration_inference"
PHASE_INTERNAL = "internal_inference"
PHASE_EXTERNAL = "external_inference"
PHASE_COMPLETE = "complete"
INFERENCE_PHASES = {
    PHASE_OPTIMIZATION,
    PHASE_CALIBRATION,
    PHASE_INTERNAL,
    PHASE_EXTERNAL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-data-root", required=True)
    parser.add_argument("--model-index", required=True)
    parser.add_argument("--model-index-audit", required=True)
    parser.add_argument("--label-manifest", required=True)
    parser.add_argument("--waveform-audit", required=True)
    parser.add_argument("--normalization-stats", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume-checkpoint", required=True)
    parser.add_argument("--status-output", required=True)
    parser.add_argument("--segment-seconds", type=int, default=14400)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _status_payload(state: dict[str, Any], checkpoint_path: Path, *, complete: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "resume_version": RESUME_VERSION,
        "complete": complete,
        "phase": state["phase"],
        "epoch": int(state["epoch"]),
        "batch_cursor": int(state["batch_cursor"]),
        "epochs_completed": len(state["epoch_history"]),
        "best_epoch": int(state["best_epoch"]),
        "best_fold8_macro_pr_auc": (
            None if not np.isfinite(float(state["best_metric"])) else float(state["best_metric"])
        ),
        "epochs_without_improvement": int(state["epochs_without_improvement"]),
        "runtime_torch_version": str(torch.__version__),
    }
    if checkpoint_path.is_file():
        payload["resume_checkpoint_sha256"] = _file_sha256(checkpoint_path)
    return payload


def _identity_payload(
    *,
    protocol_file: Path,
    protocol_summary: dict[str, Any],
    index_audit: dict[str, Any],
    manifest: dict[str, Any],
    waveform_audit: dict[str, Any],
    normalization_stats: Any,
    contract_hash: str,
    label_codes: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "study": "TRUST-ECG",
        "protocol_version": str(protocol_summary["version"]),
        "protocol_sha256": _sha256_file(protocol_file),
        "model_index_audit_sha256": str(index_audit["audit_sha256"]),
        "model_index_sha256": str(index_audit["index_sha256"]),
        "label_manifest_sha256": str(manifest["manifest_sha256"]),
        "waveform_audit_sha256": str(waveform_audit["audit_sha256"]),
        "normalization_stats_sha256": normalization_stats.stats_sha256,
        "resnet_contract_sha256": contract_hash,
        "label_codes": label_codes,
        "torch_version": str(torch.__version__),
        "device_type": "cpu",
    }


def _new_permutation(
    sample_count: int,
    generator_state: torch.Tensor | None,
    seed: int,
) -> tuple[list[int], torch.Tensor]:
    generator = torch.Generator()
    if generator_state is None:
        generator.manual_seed(seed)
    else:
        generator.set_state(generator_state)
    permutation = torch.randperm(sample_count, generator=generator).tolist()
    return permutation, generator.get_state()


def _ordered_loader(
    dataset: Any,
    indices: list[int] | range,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=indices,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def _initialize_state(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    fit_count: int,
    seed: int,
    identity: dict[str, Any],
) -> dict[str, Any]:
    permutation, generator_state = _new_permutation(fit_count, None, seed)
    return {
        "resume_version": RESUME_VERSION,
        "identity": identity,
        "phase": PHASE_TRAIN,
        "epoch": 1,
        "batch_cursor": 0,
        "epoch_permutation": permutation,
        "shuffle_generator_state": generator_state,
        "epoch_loss_sum": 0.0,
        "epoch_sample_count": 0,
        "model_state": _cpu_state_dict(model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "best_metric": float("-inf"),
        "best_epoch": 0,
        "best_state": None,
        "epochs_without_improvement": 0,
        "epoch_history": [],
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "active_inference": None,
        "inference_outputs": {},
    }


def _load_state(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    identity: dict[str, Any],
) -> dict[str, Any]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("resume_version") != RESUME_VERSION:
        raise ValueError("Unsupported TRUST-ECG resumable checkpoint version.")
    if state.get("identity") != identity:
        raise ValueError("Resumable checkpoint identity differs from the locked study state.")
    model.load_state_dict(state["model_state"])
    model.to(device)
    optimizer.load_state_dict(state["optimizer_state"])
    _optimizer_to_device(optimizer, device)
    torch.set_rng_state(state["torch_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    random.setstate(state["python_rng_state"])
    return state


def _capture_runtime_state(
    state: dict[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    state["model_state"] = _cpu_state_dict(model.state_dict())
    state["optimizer_state"] = optimizer.state_dict()
    state["torch_rng_state"] = torch.get_rng_state()
    state["numpy_rng_state"] = np.random.get_state()
    state["python_rng_state"] = random.getstate()


def _start_inference(state: dict[str, Any], phase: str) -> None:
    state["phase"] = phase
    state["batch_cursor"] = 0
    state["active_inference"] = {
        "phase": phase,
        "cursor": 0,
        "logits": [],
        "targets": [],
    }


def _advance_after_optimization(
    state: dict[str, Any],
    *,
    model: nn.Module,
    fit_count: int,
    seed: int,
    max_epochs: int,
    patience: int,
    minimum_delta: float,
) -> None:
    logits, targets = state["inference_outputs"].pop(PHASE_OPTIMIZATION)
    metric = macro_pr_auc_from_logits(targets, logits)
    training_loss = float(state["epoch_loss_sum"]) / max(int(state["epoch_sample_count"]), 1)
    epoch = int(state["epoch"])
    state["epoch_history"].append(
        {
            "epoch": epoch,
            "training_loss": training_loss,
            "fold8_macro_pr_auc": float(metric),
        }
    )
    if metric > float(state["best_metric"]) + minimum_delta:
        state["best_metric"] = float(metric)
        state["best_epoch"] = epoch
        state["best_state"] = _cpu_state_dict(model.state_dict())
        state["epochs_without_improvement"] = 0
    else:
        state["epochs_without_improvement"] = int(state["epochs_without_improvement"]) + 1

    should_stop = int(state["epochs_without_improvement"]) >= patience or epoch >= max_epochs
    if should_stop:
        if state["best_state"] is None or int(state["best_epoch"]) <= 0:
            raise RuntimeError("Fixed ResNet training produced no valid early-stopping checkpoint.")
        model.load_state_dict(state["best_state"])
        _start_inference(state, PHASE_CALIBRATION)
        return

    next_epoch = epoch + 1
    permutation, next_generator_state = _new_permutation(
        fit_count,
        state["shuffle_generator_state"],
        seed,
    )
    state["phase"] = PHASE_TRAIN
    state["epoch"] = next_epoch
    state["batch_cursor"] = 0
    state["epoch_permutation"] = permutation
    state["shuffle_generator_state"] = next_generator_state
    state["epoch_loss_sum"] = 0.0
    state["epoch_sample_count"] = 0
    state["active_inference"] = None


def _run_training_batches(
    state: dict[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dataset: Any,
    device: torch.device,
    batch_size: int,
    gradient_clip: float,
    num_workers: int,
    deadline: float,
) -> bool:
    model.train()
    permutation = list(state["epoch_permutation"])
    batch_cursor = int(state["batch_cursor"])
    offset = batch_cursor * batch_size
    remaining = permutation[offset:]
    loader = _ordered_loader(
        dataset,
        remaining,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    for inputs, labels in loader:
        if time.monotonic() >= deadline:
            return False
        inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=False)
        labels = labels.to(device=device, dtype=torch.float32, non_blocking=False)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss encountered during fixed ResNet training.")
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()
        batch_n = int(inputs.shape[0])
        state["epoch_loss_sum"] = float(state["epoch_loss_sum"]) + (
            float(loss.detach().cpu()) * batch_n
        )
        state["epoch_sample_count"] = int(state["epoch_sample_count"]) + batch_n
        state["batch_cursor"] = int(state["batch_cursor"]) + 1
    _start_inference(state, PHASE_OPTIMIZATION)
    return True


def _run_inference_batches(
    state: dict[str, Any],
    *,
    model: nn.Module,
    dataset: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    deadline: float,
) -> bool:
    active = state.get("active_inference")
    if not isinstance(active, dict) or active.get("phase") != state["phase"]:
        raise RuntimeError("Resumable inference state is inconsistent.")
    cursor = int(active["cursor"])
    loader = _ordered_loader(
        dataset,
        range(cursor, len(dataset)),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            if time.monotonic() >= deadline:
                return False
            outputs = model(inputs.to(device=device, dtype=torch.float32, non_blocking=False))
            logits = outputs.detach().cpu().numpy().astype(np.float64, copy=False)
            targets = labels.numpy().astype(np.int64, copy=False)
            active["logits"].append(logits)
            active["targets"].append(targets)
            active["cursor"] = int(active["cursor"]) + int(targets.shape[0])
            state["batch_cursor"] = int(active["cursor"])
    if not active["logits"]:
        raise RuntimeError("No ECG batches were produced for resumable inference.")
    state["inference_outputs"][state["phase"]] = (
        np.vstack(active["logits"]).astype(np.float64, copy=False),
        np.vstack(active["targets"]).astype(np.int64, copy=False),
    )
    state["active_inference"] = None
    state["batch_cursor"] = 0
    return True


def _finalize_outputs(
    state: dict[str, Any],
    *,
    output_dir: Path,
    protocol_file: Path,
    protocol_summary: dict[str, Any],
    rows: list[Any],
    index_audit: dict[str, Any],
    manifest: dict[str, Any],
    normalization_stats: Any,
    label_codes: tuple[str, ...],
    seed: int,
    contract_hash: str,
) -> None:
    if state["best_state"] is None or int(state["best_epoch"]) <= 0:
        raise RuntimeError("Cannot finalize without a valid best ResNet checkpoint.")
    calibration_logits, calibration_targets = state["inference_outputs"][PHASE_CALIBRATION]
    internal_logits, internal_targets = state["inference_outputs"][PHASE_INTERNAL]
    external_logits, external_targets = state["inference_outputs"][PHASE_EXTERNAL]
    external_rows = [row for row in rows if row.role == "external_certification"]
    expected_external_targets = np.asarray([row.labels for row in external_rows], dtype=np.int64)
    if not np.array_equal(external_targets, expected_external_targets):
        raise RuntimeError("External inference order changed relative to the locked model index.")

    calibrator = fit_platt_calibrators(
        calibration_logits,
        calibration_targets,
        label_codes=label_codes,
    )
    internal_probabilities = apply_platt_calibrators(calibrator, internal_logits)
    external_probabilities = apply_platt_calibrators(calibrator, external_logits)
    internal_metrics = _metrics_for_multilabel(
        internal_targets,
        internal_probabilities,
        label_codes,
    )
    external = _external_certification(
        rows=external_rows,
        probabilities=external_probabilities,
        label_codes=label_codes,
    )

    best_state = state["best_state"]
    state_hash = deep._state_dict_sha256(best_state)
    calibration_payload = deep._calibrator_payload(calibrator, label_codes)
    model_hash = deep._combined_model_hash(
        state_hash,
        str(calibration_payload["sha256"]),
        contract_hash,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "state_dict": best_state,
        "label_codes": label_codes,
        "protocol_sha256": _sha256_file(protocol_file),
        "model_index_sha256": str(index_audit["index_sha256"]),
        "label_manifest_sha256": str(manifest["manifest_sha256"]),
        "normalization_stats_sha256": normalization_stats.stats_sha256,
        "resnet_contract_sha256": contract_hash,
        "best_epoch": int(state["best_epoch"]),
        "best_fold8_macro_pr_auc": float(state["best_metric"]),
        "random_seed": seed,
        "state_dict_sha256": state_hash,
        "combined_model_sha256": model_hash,
    }
    torch.save(checkpoint_payload, output_dir / "open_ecg_resnet_checkpoint.pt")
    (output_dir / "open_ecg_resnet_calibration.json").write_text(
        json.dumps(calibration_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    history_payload = {
        "best_epoch": int(state["best_epoch"]),
        "best_fold8_macro_pr_auc": float(state["best_metric"]),
        "epochs_completed": len(state["epoch_history"]),
        "history": state["epoch_history"],
        "execution_mode": "batch_resumable_cpu_exact_state",
    }
    (output_dir / "open_ecg_resnet_training_history.json").write_text(
        json.dumps(history_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    role_counts = Counter(row.role for row in rows)
    payload: dict[str, Any] = {
        "report_version": "0.1.0",
        "study": "TRUST-ECG",
        "model_name": "resnet1d_fixed",
        "model_role": "predeclared_primary_waveform_baseline_not_novel_architecture",
        "primary_gate_eligible": True,
        "protocol_version": str(protocol_summary["version"]),
        "protocol_sha256": _sha256_file(protocol_file),
        "model_index_audit_sha256": str(index_audit["audit_sha256"]),
        "model_index_sha256": str(index_audit["index_sha256"]),
        "label_manifest_sha256": str(manifest["manifest_sha256"]),
        "model_sha256": model_hash,
        "label_codes": label_codes,
        "role_rows": dict(sorted(role_counts.items())),
        "internal_test": internal_metrics,
        "external_certification": external,
        "external_recovery_pool_used": False,
        "calibration_fit_role": "ptb_xl_fold_9_only",
        "optimization_role_used": True,
        "report_sha256": "",
    }
    report = _finalize_report(payload)
    write_phase0_report(report, output_dir / "open_ecg_phase0_resnet_report.json")


def main() -> int:
    args = parse_args()
    if args.segment_seconds < 600:
        raise SystemExit("--segment-seconds must be at least 600 seconds.")
    if args.num_workers < 0:
        raise SystemExit("--num-workers cannot be negative.")
    if args.device != "cpu":
        raise SystemExit("The hosted resumable runner is locked to CPU execution.")

    started = time.monotonic()
    deadline = started + args.segment_seconds
    protocol_file = Path(args.protocol).expanduser().resolve()
    output_dir = Path(args.output_root).expanduser().resolve()
    checkpoint_path = Path(args.resume_checkpoint).expanduser().resolve()
    status_path = Path(args.status_output).expanduser().resolve()

    deep.load_and_verify_model_index = load_and_verify_model_index
    deep._load_standardized_record = load_standardized_record

    protocol_summary = validate_open_ecg_protocol(protocol_file)
    protocol = load_open_ecg_protocol(protocol_file)
    manifest = load_and_verify_label_manifest(args.label_manifest)
    rows, index_audit = load_and_verify_model_index(
        index_csv=args.model_index,
        index_audit_path=args.model_index_audit,
    )
    waveform_audit = load_and_verify_waveform_audit(args.waveform_audit)
    normalization_stats = deep.load_and_verify_normalization_stats(args.normalization_stats)

    if str(index_audit["waveform_audit_sha256"]) != str(waveform_audit["audit_sha256"]):
        raise ValueError("Model-index audit and waveform audit are from different study states.")
    if str(index_audit["label_manifest_sha256"]) != str(manifest["manifest_sha256"]):
        raise ValueError("Model-index audit and label manifest are from different study states.")
    if str(waveform_audit["normalization_stats_sha256"]) != normalization_stats.stats_sha256:
        raise ValueError("Normalization statistics do not match the waveform audit.")

    label_codes = tuple(str(code) for code in index_audit["label_codes"])
    if tuple(str(item["canonical_code"]) for item in manifest["labels"]) != label_codes:
        raise ValueError("Locked label order differs between model index and label manifest.")
    if protocol["phase0_models"]["primary_model"] != "resnet1d_fixed":
        raise RuntimeError("Protocol drift: fixed ResNet must remain the primary Phase 0 model.")

    fit_rows = [row for row in rows if row.role == "model_fit"]
    optimization_rows = [row for row in rows if row.role == "optimization_validation"]
    calibration_rows = [row for row in rows if row.role == "calibration"]
    internal_rows = [row for row in rows if row.role == "internal_test"]
    external_rows = [row for row in rows if row.role == "external_certification"]
    recovery_rows = [row for row in rows if row.role == "external_recovery_pool"]
    if not all((fit_rows, optimization_rows, calibration_rows, internal_rows, external_rows, recovery_rows)):
        raise RuntimeError("All prospectively locked ECG statistical roles must be non-empty.")

    config = protocol["phase0_models"]["resnet1d_fixed"]
    batch_size = int(config["batch_size"])
    max_epochs = int(config["max_epochs"])
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    patience = int(config["early_stopping"]["patience_epochs"])
    minimum_delta = float(config["early_stopping"]["minimum_delta"])
    gradient_clip = float(config["gradient_clip_norm"])
    seed = int(config["random_seed"])

    set_torch_determinism(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")
    root = Path(args.primary_data_root).expanduser().resolve()
    contract = ResNet1DContract(seed=seed)
    contract_hash = contract.sha256()
    identity = _identity_payload(
        protocol_file=protocol_file,
        protocol_summary=protocol_summary,
        index_audit=index_audit,
        manifest=manifest,
        waveform_audit=waveform_audit,
        normalization_stats=normalization_stats,
        contract_hash=contract_hash,
        label_codes=label_codes,
    )

    datasets = {
        PHASE_TRAIN: deep._IndexedWaveformDataset(root, fit_rows, normalization_stats),
        PHASE_OPTIMIZATION: deep._IndexedWaveformDataset(
            root,
            optimization_rows,
            normalization_stats,
        ),
        PHASE_CALIBRATION: deep._IndexedWaveformDataset(
            root,
            calibration_rows,
            normalization_stats,
        ),
        PHASE_INTERNAL: deep._IndexedWaveformDataset(root, internal_rows, normalization_stats),
        PHASE_EXTERNAL: deep._IndexedWaveformDataset(root, external_rows, normalization_stats),
    }

    y_fit = np.asarray([row.labels for row in fit_rows], dtype=np.int64)
    positive_weights = compute_positive_class_weights(y_fit).to(device)
    model = FixedResNet1D(n_labels=len(label_codes), contract=contract).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    if checkpoint_path.is_file():
        state = _load_state(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
            identity=identity,
        )
    else:
        state = _initialize_state(
            model=model,
            optimizer=optimizer,
            fit_count=len(fit_rows),
            seed=seed,
            identity=identity,
        )

    while time.monotonic() < deadline and state["phase"] != PHASE_COMPLETE:
        phase = str(state["phase"])
        if phase == PHASE_TRAIN:
            finished = _run_training_batches(
                state,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                dataset=datasets[PHASE_TRAIN],
                device=device,
                batch_size=batch_size,
                gradient_clip=gradient_clip,
                num_workers=args.num_workers,
                deadline=deadline,
            )
            if not finished:
                break
            continue

        if phase in INFERENCE_PHASES:
            finished = _run_inference_batches(
                state,
                model=model,
                dataset=datasets[phase],
                device=device,
                batch_size=batch_size,
                num_workers=args.num_workers,
                deadline=deadline,
            )
            if not finished:
                break
            if phase == PHASE_OPTIMIZATION:
                _advance_after_optimization(
                    state,
                    model=model,
                    fit_count=len(fit_rows),
                    seed=seed,
                    max_epochs=max_epochs,
                    patience=patience,
                    minimum_delta=minimum_delta,
                )
            elif phase == PHASE_CALIBRATION:
                _start_inference(state, PHASE_INTERNAL)
            elif phase == PHASE_INTERNAL:
                _start_inference(state, PHASE_EXTERNAL)
            elif phase == PHASE_EXTERNAL:
                _finalize_outputs(
                    state,
                    output_dir=output_dir,
                    protocol_file=protocol_file,
                    protocol_summary=protocol_summary,
                    rows=rows,
                    index_audit=index_audit,
                    manifest=manifest,
                    normalization_stats=normalization_stats,
                    label_codes=label_codes,
                    seed=seed,
                    contract_hash=contract_hash,
                )
                state["phase"] = PHASE_COMPLETE
            continue

        raise RuntimeError(f"Unknown resumable execution phase: {phase}")

    complete = state["phase"] == PHASE_COMPLETE
    _capture_runtime_state(state, model=model, optimizer=optimizer)
    _atomic_torch_save(checkpoint_path, state)
    status = _status_payload(state, checkpoint_path, complete=complete)
    status["segment_elapsed_seconds"] = float(time.monotonic() - started)
    if complete:
        report = json.loads((output_dir / "open_ecg_phase0_resnet_report.json").read_text())
        history = json.loads((output_dir / "open_ecg_resnet_training_history.json").read_text())
        calibration = json.loads((output_dir / "open_ecg_resnet_calibration.json").read_text())
        status.update(
            {
                "report_sha256": report["report_sha256"],
                "model_sha256": report["model_sha256"],
                "calibration_sha256": calibration["sha256"],
                "internal_macro_pr_auc": report["internal_test"]["macro_pr_auc"],
                "best_epoch": history["best_epoch"],
                "best_fold8_macro_pr_auc": history["best_fold8_macro_pr_auc"],
            }
        )
    _atomic_json(status_path, status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
