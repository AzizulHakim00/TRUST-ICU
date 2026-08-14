"""Model inference and paired baseline comparisons for TRUST-ECG."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from trust_icu.ecg_baseline import (
    apply_platt_calibrators,
    evaluate_binary_probabilities,
    extract_handcrafted_features,
    fit_logistic_reference,
    fit_platt_calibrators,
    logistic_decision_scores,
)
from trust_icu.ecg_deep_phase0 import _combined_model_hash, _state_dict_sha256
from trust_icu.ecg_phase0_v04 import load_standardized_record
from trust_icu.ecg_phase1 import (
    _global_calibrated_probabilities,
    _verify_calibration_payload,
)
from trust_icu.ecg_resnet import FixedResNet1D, ResNet1DContract
from trust_icu.ecg_signal import normalize_signal
from trust_icu.ecg_statistical_core import (
    METRICS,
    benjamini_hochberg,
    paired_binary_bootstrap,
    paired_macro_bootstrap,
)

LABEL_NAMES = {
    "59118001": "RBBB",
    "164889003": "AF",
    "164909002": "LBBB",
    "270492004": "IAVB",
    "284470004": "PAC",
    "426783006": "NSR",
    "427084000": "STach",
}
EXTERNAL_SOURCES = ("georgia", "cpsc_2018", "cpsc_2018_extra")
PRIMARY_SEED = 20260808


class WaveformDataset(Dataset):
    def __init__(self, data_root: Path, rows: list[Any], normalization_stats: Any) -> None:
        self.data_root = data_root
        self.rows = rows
        self.normalization_stats = normalization_stats

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        standardized = load_standardized_record(self.data_root, row)
        normalized = normalize_signal(
            standardized.waveform_mv,
            standardized.valid_mask,
            self.normalization_stats,
        )
        return torch.from_numpy(normalized), torch.tensor(row.labels, dtype=torch.int64)


def load_verified_resnet(
    *,
    checkpoint_path: Path,
    calibration_path: Path,
    report: dict[str, Any],
    normalization_stats: Any,
    label_codes: tuple[str, ...],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_hash = _verify_calibration_payload(calibration_payload, label_codes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Fixed-ResNet checkpoint has an unexpected structure.")
    if tuple(checkpoint.get("label_codes", ())) != label_codes:
        raise ValueError("Checkpoint label order differs from the primary report.")
    expected_fields = {
        "protocol_sha256": report["protocol_sha256"],
        "model_index_sha256": report["model_index_sha256"],
        "label_manifest_sha256": report["label_manifest_sha256"],
        "normalization_stats_sha256": normalization_stats.stats_sha256,
    }
    for field, expected in expected_fields.items():
        if str(checkpoint.get(field)) != str(expected):
            raise ValueError(f"Checkpoint {field} differs from the frozen study state.")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing the fixed ResNet state_dict.")
    state_hash = _state_dict_sha256(state_dict)
    contract = ResNet1DContract(seed=int(checkpoint["random_seed"]))
    contract_hash = contract.sha256()
    if str(checkpoint.get("state_dict_sha256")) != state_hash:
        raise ValueError("Checkpoint state_dict SHA-256 verification failed.")
    if str(checkpoint.get("resnet_contract_sha256")) != contract_hash:
        raise ValueError("Checkpoint contract SHA-256 verification failed.")
    combined = _combined_model_hash(state_hash, calibration_hash, contract_hash)
    if str(checkpoint.get("combined_model_sha256")) != combined:
        raise ValueError("Checkpoint combined-model SHA-256 verification failed.")
    if combined != str(report["model_sha256"]):
        raise ValueError("Checkpoint and calibration do not reproduce the primary model hash.")
    model = FixedResNet1D(n_labels=len(label_codes), contract=contract)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, calibration_payload


def collect_resnet_probabilities(
    *,
    model: torch.nn.Module,
    rows: list[Any],
    data_root: Path,
    normalization_stats: Any,
    calibration_payload: dict[str, Any],
    label_codes: tuple[str, ...],
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        WaveformDataset(data_root, rows, normalization_stats),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    logits_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs.to(device=device, dtype=torch.float32))
            logits_parts.append(outputs.detach().cpu().numpy())
            target_parts.append(targets.numpy().astype(np.int64, copy=False))
    if not logits_parts:
        raise RuntimeError("No ECG batches were produced for addendum inference.")
    logits = np.vstack(logits_parts).astype(np.float64, copy=False)
    targets = np.vstack(target_parts).astype(np.int64, copy=False)
    expected = np.asarray([row.labels for row in rows], dtype=np.int64)
    if not np.array_equal(targets, expected):
        raise RuntimeError("Addendum DataLoader order changed relative to the frozen index.")
    probabilities = _global_calibrated_probabilities(logits, calibration_payload, label_codes)
    return probabilities, targets


def extract_feature_matrix(data_root: Path, rows: list[Any]) -> np.ndarray:
    features: list[np.ndarray] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        standardized = load_standardized_record(data_root, row)
        features.append(
            extract_handcrafted_features(
                standardized.waveform_mv,
                standardized.valid_mask,
            )
        )
        if index % 2000 == 0 or index == total:
            print(f"handcrafted_features {index}/{total}", flush=True)
    return np.vstack(features).astype(np.float64, copy=False)


def fit_logistic_probabilities(
    *,
    data_root: Path,
    fit_rows: list[Any],
    calibration_rows: list[Any],
    internal_rows: list[Any],
    external_rows: list[Any],
    label_codes: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    selected = [*fit_rows, *calibration_rows, *internal_rows, *external_rows]
    matrix = extract_feature_matrix(data_root, selected)
    first_calibration = len(fit_rows)
    first_internal = first_calibration + len(calibration_rows)
    first_external = first_internal + len(internal_rows)
    X_fit = matrix[:first_calibration]
    X_calibration = matrix[first_calibration:first_internal]
    X_internal = matrix[first_internal:first_external]
    X_external = matrix[first_external:]
    y_fit = np.asarray([row.labels for row in fit_rows], dtype=np.int64)
    y_calibration = np.asarray([row.labels for row in calibration_rows], dtype=np.int64)
    model = fit_logistic_reference(X_fit, y_fit, label_codes=label_codes)
    calibrator = fit_platt_calibrators(
        logistic_decision_scores(model, X_calibration),
        y_calibration,
        label_codes=label_codes,
    )
    internal = apply_platt_calibrators(calibrator, logistic_decision_scores(model, X_internal))
    external = apply_platt_calibrators(calibrator, logistic_decision_scores(model, X_external))
    return internal, external


def validate_resnet_report_metrics(
    *,
    report: dict[str, Any],
    internal_targets: np.ndarray,
    internal_probabilities: np.ndarray,
    external_rows: list[Any],
    external_targets: np.ndarray,
    external_probabilities: np.ndarray,
    label_codes: tuple[str, ...],
) -> None:
    for index, code in enumerate(label_codes):
        observed = asdict(
            evaluate_binary_probabilities(
                internal_targets[:, index],
                internal_probabilities[:, index],
            )
        )
        expected = report["internal_test"]["per_label"][code]
        for metric, value in observed.items():
            if not np.isclose(float(value), float(expected[metric]), rtol=1e-10, atol=1e-12):
                raise RuntimeError(f"Internal ResNet metric mismatch for {code}/{metric}.")
    for source in EXTERNAL_SOURCES:
        indices = [index for index, row in enumerate(external_rows) if row.source == source]
        for label_index, code in enumerate(label_codes):
            expected_pair = report["external_certification"][source][code]
            if expected_pair["metrics"] is None:
                continue
            observed = asdict(
                evaluate_binary_probabilities(
                    external_targets[indices, label_index],
                    external_probabilities[indices, label_index],
                )
            )
            for metric, value in observed.items():
                if not np.isclose(
                    float(value),
                    float(expected_pair["metrics"][metric]),
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        f"External ResNet metric mismatch for {source}/{code}/{metric}."
                    )


def _comparison_row(
    *,
    scope: str,
    source: str,
    code: str,
    metric: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    improvement = result["paired_improvement"][metric]
    return {
        "scope": scope,
        "source": source,
        "label_code": code,
        "label_name": "Macro average" if code == "MACRO" else LABEL_NAMES[code],
        "metric": metric,
        "resnet_point": result["point_candidate"][metric],
        "resnet_q025": result["candidate_intervals"][metric]["q025"],
        "resnet_q975": result["candidate_intervals"][metric]["q975"],
        "logistic_point": result["point_reference"][metric],
        "logistic_q025": result["reference_intervals"][metric]["q025"],
        "logistic_q975": result["reference_intervals"][metric]["q975"],
        "paired_improvement_point": (
            result["point_reference"][metric] - result["point_candidate"][metric]
            if metric == "brier"
            else result["point_candidate"][metric] - result["point_reference"][metric]
        ),
        "paired_improvement_median": improvement["median"],
        "paired_improvement_q025": improvement["q025"],
        "paired_improvement_q975": improvement["q975"],
        "bootstrap_p": improvement["two_sided_bootstrap_p"],
    }


def paired_model_results(
    *,
    internal_targets: np.ndarray,
    resnet_internal: np.ndarray,
    logistic_internal: np.ndarray,
    external_rows: list[Any],
    external_targets: np.ndarray,
    resnet_external: np.ndarray,
    logistic_external: np.ndarray,
    label_codes: tuple[str, ...],
    repeats: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    internal: dict[str, Any] = {
        "macro": paired_macro_bootstrap(
            internal_targets,
            resnet_internal,
            logistic_internal,
            repeats=repeats,
            seed=PRIMARY_SEED,
        ),
        "per_label": {},
    }
    internal_rows_out = [
        _comparison_row(
            scope="internal_fold10",
            source="ptb-xl",
            code="MACRO",
            metric=metric,
            result=internal["macro"],
        )
        for metric in METRICS
    ]
    for label_index, code in enumerate(label_codes):
        result = paired_binary_bootstrap(
            internal_targets[:, label_index],
            resnet_internal[:, label_index],
            logistic_internal[:, label_index],
            repeats=repeats,
            seed=PRIMARY_SEED + label_index + 1,
        )
        internal["per_label"][code] = result
        internal_rows_out.extend(
            _comparison_row(
                scope="internal_fold10",
                source="ptb-xl",
                code=code,
                metric=metric,
                result=result,
            )
            for metric in METRICS
        )

    external: dict[str, Any] = {}
    external_rows_out: list[dict[str, Any]] = []
    for source_index, source in enumerate(EXTERNAL_SOURCES):
        source_indices = np.asarray(
            [index for index, row in enumerate(external_rows) if row.source == source],
            dtype=np.int64,
        )
        external[source] = {}
        for label_index, code in enumerate(label_codes):
            y = external_targets[source_indices, label_index]
            if np.unique(y).size != 2:
                external[source][code] = {"estimable": False, "reason": "single_class"}
                continue
            result = paired_binary_bootstrap(
                y,
                resnet_external[source_indices, label_index],
                logistic_external[source_indices, label_index],
                repeats=repeats,
                seed=PRIMARY_SEED + 100 + source_index * 20 + label_index,
            )
            external[source][code] = {"estimable": True, **result}
            external_rows_out.extend(
                _comparison_row(
                    scope="external_certification",
                    source=source,
                    code=code,
                    metric=metric,
                    result=result,
                )
                for metric in METRICS
            )

    for output_rows in (internal_rows_out, external_rows_out):
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(output_rows):
            groups[str(row["metric"])].append(index)
        for indices in groups.values():
            q_values = benjamini_hochberg(
                [output_rows[index]["bootstrap_p"] for index in indices]
            )
            for index, q_value in zip(indices, q_values, strict=True):
                output_rows[index]["bootstrap_q_bh"] = q_value
    return {"internal": internal, "external": external}, internal_rows_out, external_rows_out
