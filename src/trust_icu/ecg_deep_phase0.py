"""Optional PyTorch execution path for the fixed TRUST-ECG primary ResNet baseline."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from trust_icu.ecg_baseline import apply_platt_calibrators, fit_platt_calibrators
from trust_icu.ecg_index import EcgIndexRow
from trust_icu.ecg_manifest import load_and_verify_label_manifest
from trust_icu.ecg_phase0 import (
    EcgPhase0Report,
    _external_certification,
    _finalize_report,
    _load_standardized_record,
    _metrics_for_multilabel,
    _sha256_file,
    load_and_verify_model_index,
    load_and_verify_normalization_stats,
)
from trust_icu.ecg_protocol import load_open_ecg_protocol, validate_open_ecg_protocol
from trust_icu.ecg_signal import normalize_signal
from trust_icu.ecg_waveform import load_and_verify_waveform_audit

try:
    import torch
    from torch import nn
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - optional dependency path
    raise RuntimeError(
        'PyTorch is required for TRUST-ECG ResNet execution. Install with `pip install -e ".[ecg-deep]"`.'
    ) from exc

from trust_icu.ecg_resnet import (
    FixedResNet1D,
    ResNet1DContract,
    compute_positive_class_weights,
    macro_pr_auc_from_logits,
    set_torch_determinism,
)


class _IndexedWaveformDataset(Dataset):
    def __init__(self, data_root: Path, rows: list[EcgIndexRow], normalization_stats: Any) -> None:
        self.data_root = data_root
        self.rows = rows
        self.normalization_stats = normalization_stats

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        standardized = _load_standardized_record(self.data_root, row)
        normalized = normalize_signal(
            standardized.waveform_mv,
            standardized.valid_mask,
            self.normalization_stats,
        )
        return (
            torch.from_numpy(normalized),
            torch.tensor(row.labels, dtype=torch.float32),
        )


def _loader(
    data_root: Path,
    rows: list[EcgIndexRow],
    normalization_stats: Any,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    if not rows:
        raise RuntimeError("Cannot build an ECG DataLoader for an empty statistical role.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        _IndexedWaveformDataset(data_root, rows, normalization_stats),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def _collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device=device, dtype=torch.float32, non_blocking=True))
            logits.append(outputs.detach().cpu().numpy())
            targets.append(labels.numpy().astype(np.int64, copy=False))
    if not logits:
        raise RuntimeError("No ECG batches were produced for evaluation.")
    return np.vstack(logits).astype(np.float64, copy=False), np.vstack(targets).astype(np.int64, copy=False)


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(np.asarray(tensor.numpy()).tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _calibrator_payload(calibrator: Any, label_codes: tuple[str, ...]) -> dict[str, Any]:
    parameters: dict[str, dict[str, float]] = {}
    for code, estimator in zip(label_codes, calibrator.estimators, strict=True):
        parameters[code] = {
            "coefficient": float(estimator.coef_[0, 0]),
            "intercept": float(estimator.intercept_[0]),
        }
    payload: dict[str, Any] = {
        "method": "independent_platt_scaling_per_label",
        "fit_role": "ptb_xl_fold_9_only",
        "label_codes": list(label_codes),
        "parameters": parameters,
        "sha256": "",
    }
    material = dict(payload)
    material["sha256"] = ""
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _combined_model_hash(state_hash: str, calibration_hash: str, contract_hash: str) -> str:
    return hashlib.sha256(f"{state_hash}|{calibration_hash}|{contract_hash}".encode()).hexdigest()


def execute_fixed_resnet_phase0(
    *,
    challenge_training_root: str | Path,
    index_csv: str | Path,
    index_audit_path: str | Path,
    label_manifest_path: str | Path,
    waveform_audit_path: str | Path,
    normalization_stats_path: str | Path,
    protocol_path: str | Path,
    output_root: str | Path,
    device_name: str | None = None,
    num_workers: int = 0,
) -> EcgPhase0Report:
    """Train, calibrate and externally certify the prospectively fixed ResNet baseline."""

    protocol_file = Path(protocol_path).expanduser().resolve()
    protocol_summary = validate_open_ecg_protocol(protocol_file)
    protocol = load_open_ecg_protocol(protocol_file)
    manifest = load_and_verify_label_manifest(label_manifest_path)
    rows, index_audit = load_and_verify_model_index(
        index_csv=index_csv,
        index_audit_path=index_audit_path,
    )
    waveform_audit = load_and_verify_waveform_audit(waveform_audit_path)
    normalization_stats = load_and_verify_normalization_stats(normalization_stats_path)

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
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    set_torch_determinism(seed)
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

    root = Path(challenge_training_root).expanduser().resolve()
    fit_loader = _loader(
        root,
        fit_rows,
        normalization_stats,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
    )
    optimization_loader = _loader(
        root,
        optimization_rows,
        normalization_stats,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
    )
    calibration_loader = _loader(
        root,
        calibration_rows,
        normalization_stats,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
    )
    internal_loader = _loader(
        root,
        internal_rows,
        normalization_stats,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
    )
    external_loader = _loader(
        root,
        external_rows,
        normalization_stats,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
    )

    y_fit = np.asarray([row.labels for row in fit_rows], dtype=np.int64)
    positive_weights = compute_positive_class_weights(y_fit).to(device)
    model = FixedResNet1D(n_labels=len(label_codes), contract=ResNet1DContract(seed=seed)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_metric = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epoch_history: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for inputs, labels in fit_loader:
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered during fixed ResNet training.")
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
            batch_n = int(inputs.shape[0])
            loss_sum += float(loss.detach().cpu()) * batch_n
            sample_count += batch_n

        optimization_logits, optimization_targets = _collect_logits(
            model,
            optimization_loader,
            device,
        )
        metric = macro_pr_auc_from_logits(optimization_targets, optimization_logits)
        epoch_history.append(
            {
                "epoch": epoch,
                "training_loss": loss_sum / max(sample_count, 1),
                "fold8_macro_pr_auc": metric,
            }
        )
        if metric > best_metric + minimum_delta:
            best_metric = metric
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is None or best_epoch <= 0:
        raise RuntimeError("Fixed ResNet training produced no valid early-stopping checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)

    calibration_logits, calibration_targets = _collect_logits(model, calibration_loader, device)
    calibrator = fit_platt_calibrators(
        calibration_logits,
        calibration_targets,
        label_codes=label_codes,
    )
    internal_logits, internal_targets = _collect_logits(model, internal_loader, device)
    external_logits, external_targets = _collect_logits(model, external_loader, device)
    expected_external_targets = np.asarray([row.labels for row in external_rows], dtype=np.int64)
    if not np.array_equal(external_targets, expected_external_targets):
        raise RuntimeError("External DataLoader order changed relative to the locked model index.")

    internal_probabilities = apply_platt_calibrators(calibrator, internal_logits)
    external_probabilities = apply_platt_calibrators(calibrator, external_logits)
    internal_metrics = _metrics_for_multilabel(internal_targets, internal_probabilities, label_codes)
    external = _external_certification(
        rows=external_rows,
        probabilities=external_probabilities,
        label_codes=label_codes,
    )

    state_hash = _state_dict_sha256(best_state)
    calibration_payload = _calibrator_payload(calibrator, label_codes)
    contract_hash = ResNet1DContract(seed=seed).sha256()
    model_hash = _combined_model_hash(state_hash, str(calibration_payload["sha256"]), contract_hash)

    output_dir = Path(output_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "state_dict": best_state,
        "label_codes": label_codes,
        "protocol_sha256": _sha256_file(protocol_file),
        "model_index_sha256": str(index_audit["index_sha256"]),
        "label_manifest_sha256": str(manifest["manifest_sha256"]),
        "normalization_stats_sha256": normalization_stats.stats_sha256,
        "resnet_contract_sha256": contract_hash,
        "best_epoch": best_epoch,
        "best_fold8_macro_pr_auc": float(best_metric),
        "random_seed": seed,
        "state_dict_sha256": state_hash,
        "combined_model_sha256": model_hash,
    }
    torch.save(checkpoint_payload, output_dir / "open_ecg_resnet_checkpoint.pt")
    calibration_path = output_dir / "open_ecg_resnet_calibration.json"
    calibration_path.write_text(
        json.dumps(calibration_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    history_path = output_dir / "open_ecg_resnet_training_history.json"
    history_payload = {
        "best_epoch": best_epoch,
        "best_fold8_macro_pr_auc": float(best_metric),
        "epochs_completed": len(epoch_history),
        "history": epoch_history,
    }
    history_path.write_text(json.dumps(history_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
    return _finalize_report(payload)
