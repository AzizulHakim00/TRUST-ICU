"""Conditional label-efficient probability recovery for TRUST-ECG protocol v0.4.

Phase 1 is fail-closed. It is activated only by a verified primary fixed-ResNet
Phase-0 report and only for label-domain pairs prospectively classified as
``calibration_recovery_candidate``. The frozen model is never retrained.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

from trust_icu.ecg_baseline import BinaryProbabilityMetrics, evaluate_binary_probabilities
from trust_icu.ecg_manifest import load_and_verify_label_manifest
from trust_icu.ecg_phase0_v04 import (
    load_and_verify_model_index,
    load_and_verify_normalization_stats,
    load_standardized_record,
)
from trust_icu.ecg_protocol import load_open_ecg_protocol, validate_open_ecg_protocol
from trust_icu.ecg_signal import normalize_signal

_EXTERNAL_SOURCES = ("georgia", "cpsc_2018", "cpsc_2018_extra")
_ALLOWED_PAIR_STATUSES = {
    "certified",
    "calibration_recovery_candidate",
    "discrimination_failure",
    "insufficient_support",
}
_ALLOWED_METHODS = (
    "frozen_no_update",
    "intercept_only_recalibration",
    "platt_recalibration",
)


@dataclass(frozen=True)
class EcgPhase1Candidate:
    source: str
    label_code: str


@dataclass(frozen=True)
class EcgPhase1Plan:
    study: str
    protocol_version: str
    status: str
    activated: bool
    minimum_candidate_domains: int
    candidate_domains: tuple[str, ...]
    candidate_pairs: tuple[EcgPhase1Candidate, ...]
    target_label_budgets: tuple[int, ...]
    repeats: int
    sampling_seed: int
    sampling_stratified: bool
    adaptation_records_excluded_from_evaluation: bool
    methods: tuple[str, ...]
    phase0_report_sha256: str
    phase0_model_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: dict[str, Any], key: str) -> str:
    material = dict(payload)
    material[key] = ""
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest.")
    return text


def load_and_verify_phase0_report(
    phase0_report_path: str | Path,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Load the fixed-ResNet Phase-0 report and fail closed on any drift."""

    report_path = Path(phase0_report_path).expanduser().resolve()
    protocol_file = Path(protocol_path).expanduser().resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"TRUST-ECG Phase-0 report not found: {report_path}")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("TRUST-ECG Phase-0 report must contain one JSON object.")
    if payload.get("report_version") != "0.1.0":
        raise ValueError("Unexpected TRUST-ECG Phase-0 report version.")
    if payload.get("study") != "TRUST-ECG":
        raise ValueError("Phase-1 activation requires a TRUST-ECG report.")
    if payload.get("model_name") != "resnet1d_fixed":
        raise ValueError("Phase-1 activation requires the fixed ResNet primary model.")
    if payload.get("primary_gate_eligible") is not True:
        raise ValueError("Phase-1 activation requires a primary-gate-eligible model report.")
    if payload.get("external_recovery_pool_used") is not False:
        raise ValueError("Phase-0 report indicates premature recovery-pool use.")
    if payload.get("optimization_role_used") is not True:
        raise ValueError("Fixed ResNet report must use the locked optimization role.")
    if payload.get("calibration_fit_role") != "ptb_xl_fold_9_only":
        raise ValueError("Fixed ResNet report does not use the locked fold-9 calibration role.")

    protocol_summary = validate_open_ecg_protocol(protocol_file)
    if str(payload.get("protocol_version")) != str(protocol_summary["version"]):
        raise ValueError("Phase-0 report and current protocol versions differ.")
    if str(payload.get("protocol_sha256")) != _sha256_file(protocol_file):
        raise ValueError("Phase-0 report and current protocol SHA-256 differ.")

    expected_report_hash = _canonical_hash(payload, "report_sha256")
    if str(payload.get("report_sha256")) != expected_report_hash:
        raise ValueError("Phase-0 report SHA-256 verification failed.")

    for field in (
        "report_sha256",
        "protocol_sha256",
        "model_index_audit_sha256",
        "model_index_sha256",
        "label_manifest_sha256",
        "model_sha256",
    ):
        _require_sha256(payload.get(field), field)

    label_codes = payload.get("label_codes")
    if (
        not isinstance(label_codes, list)
        or not label_codes
        or any(not isinstance(code, str) or not code for code in label_codes)
        or len(set(label_codes)) != len(label_codes)
    ):
        raise ValueError("Phase-0 report contains an invalid locked label-code list.")

    external = payload.get("external_certification")
    if not isinstance(external, dict) or set(external) != set(_EXTERNAL_SOURCES):
        raise ValueError("Phase-0 report must contain all three locked external sources.")
    for source in _EXTERNAL_SOURCES:
        source_payload = external[source]
        if not isinstance(source_payload, dict) or set(source_payload) != set(label_codes):
            raise ValueError(f"External certification matrix is incomplete for {source}.")
        for code in label_codes:
            pair = source_payload[code]
            if not isinstance(pair, dict) or pair.get("status") not in _ALLOWED_PAIR_STATUSES:
                raise ValueError(f"Invalid Phase-0 pair status for {source}/{code}.")

    return payload


def build_phase1_plan(
    phase0_report_path: str | Path,
    protocol_path: str | Path,
) -> EcgPhase1Plan:
    """Build the conditional Phase-1 plan without touching recovery waveforms."""

    report = load_and_verify_phase0_report(phase0_report_path, protocol_path)
    protocol = load_open_ecg_protocol(protocol_path)
    phase1 = protocol["phase1_if_phase0_passes"]
    activation = protocol["phase0_go_no_go"]["phase1_activation"]

    candidates: list[EcgPhase1Candidate] = []
    candidate_domains: set[str] = set()
    for source in _EXTERNAL_SOURCES:
        for code in report["label_codes"]:
            status = report["external_certification"][source][code]["status"]
            if status == "calibration_recovery_candidate":
                candidates.append(EcgPhase1Candidate(source=source, label_code=code))
                candidate_domains.add(source)

    minimum_domains = int(
        activation["minimum_external_domains_with_at_least_one_recovery_candidate"]
    )
    activated = len(candidate_domains) >= minimum_domains
    status = (
        "activated_for_label_efficient_probability_recovery"
        if activated
        else "blocked_insufficient_recovery_candidate_domains"
    )

    budgets = tuple(int(value) for value in phase1["target_label_budgets"])
    if budgets != (0, 50, 100, 250, 500, 1000):
        raise RuntimeError("Protocol drift: TRUST-ECG Phase-1 label budgets changed.")
    repeats = int(phase1["sampling"]["repeats"])
    seed = int(phase1["sampling"]["seed"])
    if repeats != 100:
        raise RuntimeError("Protocol drift: TRUST-ECG Phase-1 repeat count changed.")
    if phase1["sampling"]["label_stratification"] is not False:
        raise RuntimeError("Protocol drift: target-domain sampling must remain unstratified.")
    if phase1["sampling"]["draw_records_uniformly_without_replacement"] is not True:
        raise RuntimeError("Protocol drift: target-domain sampling rule changed.")
    if phase1["evaluation"]["adaptation_records_excluded_from_evaluation"] is not True:
        raise RuntimeError("Protocol drift: adaptation records must be excluded from evaluation.")
    methods = ("frozen_no_update", *tuple(str(x) for x in phase1["allowed_local_updates"]))
    if methods != _ALLOWED_METHODS:
        raise RuntimeError("Protocol drift: TRUST-ECG Phase-1 local update methods changed.")

    return EcgPhase1Plan(
        study="TRUST-ECG",
        protocol_version=str(report["protocol_version"]),
        status=status,
        activated=activated,
        minimum_candidate_domains=minimum_domains,
        candidate_domains=tuple(source for source in _EXTERNAL_SOURCES if source in candidate_domains),
        candidate_pairs=tuple(candidates),
        target_label_budgets=budgets,
        repeats=repeats,
        sampling_seed=seed,
        sampling_stratified=False,
        adaptation_records_excluded_from_evaluation=True,
        methods=methods,
        phase0_report_sha256=str(report["report_sha256"]),
        phase0_model_sha256=str(report["model_sha256"]),
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _probability_logit(probabilities: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(probs / (1.0 - probs))


def _fit_intercept_only(base_logits: np.ndarray, y: np.ndarray) -> float:
    targets = np.asarray(y, dtype=np.int64)
    logits = np.asarray(base_logits, dtype=np.float64)
    if targets.ndim != 1 or logits.shape != targets.shape:
        raise ValueError("Intercept-only recalibration inputs must align.")
    if np.unique(targets).size != 2:
        raise ValueError("Intercept-only recalibration requires both target classes.")

    prevalence = float(np.mean(targets))
    objective = lambda shift: float(np.mean(_sigmoid(logits + shift)) - prevalence)
    return float(brentq(objective, -60.0, 60.0))


def _fit_local_platt(base_logits: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    targets = np.asarray(y, dtype=np.int64)
    logits = np.asarray(base_logits, dtype=np.float64)
    if targets.ndim != 1 or logits.shape != targets.shape:
        raise ValueError("Platt recalibration inputs must align.")
    if np.unique(targets).size != 2:
        raise ValueError("Platt recalibration requires both target classes.")
    estimator = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    )
    estimator.fit(logits[:, None], targets)
    return float(estimator.coef_[0, 0]), float(estimator.intercept_[0])


def _stable_draw_indices(
    n: int,
    budget: int,
    *,
    seed: int,
    source: str,
    label_code: str,
    repeat: int,
) -> np.ndarray:
    if n < 0 or budget < 0 or repeat < 0:
        raise ValueError("Sampling dimensions must be non-negative.")
    used = min(n, budget)
    if used == 0:
        return np.empty(0, dtype=np.int64)
    material = f"{seed}|{source}|{label_code}|{budget}|{repeat}".encode()
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    rng = np.random.default_rng(derived_seed)
    return np.sort(rng.choice(n, size=used, replace=False).astype(np.int64, copy=False))


def _recovery_envelope_met(
    metrics: BinaryProbabilityMetrics,
    protocol: dict[str, Any],
) -> bool:
    gate = protocol["phase0_go_no_go"]
    discrimination = gate["discrimination_viability"]
    calibration = gate["calibration_envelope"]
    return bool(
        metrics.pr_auc_to_prevalence_ratio
        >= float(discrimination["minimum_pr_auc_to_prevalence_ratio"])
        and abs(metrics.calibration_slope - 1.0)
        <= float(calibration["maximum_absolute_calibration_slope_deviation"])
        and abs(metrics.calibration_intercept)
        <= float(calibration["maximum_absolute_calibration_intercept"])
        and metrics.brier_skill_vs_prevalence > 0.0
    )


def _metric_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "q025": float(np.quantile(array, 0.025)),
        "q975": float(np.quantile(array, 0.975)),
    }


def _summarize_repeats(records: list[dict[str, Any]]) -> dict[str, Any]:
    estimable = [record for record in records if record["estimable"]]
    summary: dict[str, Any] = {
        "repeats_requested": len(records),
        "estimable_repeats": len(estimable),
        "nonestimable_repeats": len(records) - len(estimable),
        "nonestimable_reasons": {},
        "recovery_envelope_met_count": 0,
        "recovery_envelope_met_rate_among_estimable": None,
        "metrics": {},
    }
    reasons: dict[str, int] = {}
    for record in records:
        if not record["estimable"]:
            reason = str(record["nonestimable_reason"])
            reasons[reason] = reasons.get(reason, 0) + 1
    summary["nonestimable_reasons"] = dict(sorted(reasons.items()))
    if not estimable:
        return summary

    recovered = sum(bool(record["recovery_envelope_met"]) for record in estimable)
    summary["recovery_envelope_met_count"] = recovered
    summary["recovery_envelope_met_rate_among_estimable"] = recovered / len(estimable)
    metric_names = (
        "pr_auc",
        "pr_auc_to_prevalence_ratio",
        "roc_auc",
        "brier",
        "brier_skill_vs_prevalence",
        "calibration_slope",
        "calibration_intercept",
    )
    for name in metric_names:
        summary["metrics"][name] = _metric_summary(
            [float(record["metrics"][name]) for record in estimable]
        )
    return summary


def _verify_calibration_payload(payload: dict[str, Any], label_codes: tuple[str, ...]) -> str:
    if payload.get("method") != "independent_platt_scaling_per_label":
        raise ValueError("Unexpected global calibration method.")
    if payload.get("fit_role") != "ptb_xl_fold_9_only":
        raise ValueError("Global calibration did not use PTB-XL fold 9.")
    if tuple(payload.get("label_codes", [])) != label_codes:
        raise ValueError("Global calibration label order differs from the locked task.")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != set(label_codes):
        raise ValueError("Global calibration parameters are incomplete.")
    expected = _canonical_hash(payload, "sha256")
    if str(payload.get("sha256")) != expected:
        raise ValueError("Global calibration SHA-256 verification failed.")
    return expected


def _global_calibrated_probabilities(
    raw_logits: np.ndarray,
    calibration_payload: dict[str, Any],
    label_codes: tuple[str, ...],
) -> np.ndarray:
    logits = np.asarray(raw_logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != len(label_codes):
        raise ValueError("Raw recovery logits do not align with the locked label set.")
    columns: list[np.ndarray] = []
    for index, code in enumerate(label_codes):
        params = calibration_payload["parameters"][code]
        coefficient = float(params["coefficient"])
        intercept = float(params["intercept"])
        columns.append(_sigmoid(coefficient * logits[:, index] + intercept))
    return np.column_stack(columns)


def execute_phase1_probability_recovery(
    *,
    phase0_report_path: str | Path,
    primary_data_root: str | Path,
    index_csv: str | Path,
    index_audit_path: str | Path,
    label_manifest_path: str | Path,
    normalization_stats_path: str | Path,
    checkpoint_path: str | Path,
    global_calibration_path: str | Path,
    protocol_path: str | Path,
    output_path: str | Path,
    device_name: str | None = None,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Execute the locked conditional recovery experiment on the untouched 40% pool."""

    plan = build_phase1_plan(phase0_report_path, protocol_path)
    if not plan.activated:
        raise RuntimeError("TRUST-ECG Phase 1 is not activated by the primary Phase-0 report.")

    report = load_and_verify_phase0_report(phase0_report_path, protocol_path)
    protocol = load_open_ecg_protocol(protocol_path)
    manifest = load_and_verify_label_manifest(label_manifest_path)
    rows, index_audit = load_and_verify_model_index(
        index_csv=index_csv,
        index_audit_path=index_audit_path,
    )
    normalization_stats = load_and_verify_normalization_stats(normalization_stats_path)

    label_codes = tuple(str(code) for code in report["label_codes"])
    if tuple(str(code) for code in index_audit["label_codes"]) != label_codes:
        raise ValueError("Model index label order differs from the primary report.")
    if str(index_audit["index_sha256"]) != str(report["model_index_sha256"]):
        raise ValueError("Model index SHA-256 differs from the primary report.")
    if str(index_audit["audit_sha256"]) != str(report["model_index_audit_sha256"]):
        raise ValueError("Model-index audit SHA-256 differs from the primary report.")
    if str(manifest["manifest_sha256"]) != str(report["label_manifest_sha256"]):
        raise ValueError("Label manifest SHA-256 differs from the primary report.")
    if tuple(str(item["canonical_code"]) for item in manifest["labels"]) != label_codes:
        raise ValueError("Label-manifest order differs from the primary report.")

    calibration_payload = json.loads(
        Path(global_calibration_path).expanduser().resolve().read_text(encoding="utf-8")
    )
    if not isinstance(calibration_payload, dict):
        raise ValueError("Global calibration artifact must contain one JSON object.")
    calibration_sha256 = _verify_calibration_payload(calibration_payload, label_codes)

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:  # pragma: no cover - optional heavy runtime
        raise RuntimeError(
            'PyTorch is required for TRUST-ECG Phase 1. Install with `pip install -e ".[ecg-deep]"`.'
        ) from exc

    from trust_icu.ecg_deep_phase0 import _combined_model_hash, _state_dict_sha256
    from trust_icu.ecg_resnet import FixedResNet1D, ResNet1DContract

    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Fixed-ResNet checkpoint has an unexpected structure.")
    if tuple(checkpoint.get("label_codes", ())) != label_codes:
        raise ValueError("Checkpoint label order differs from the primary report.")
    if str(checkpoint.get("protocol_sha256")) != str(report["protocol_sha256"]):
        raise ValueError("Checkpoint protocol SHA-256 differs from the primary report.")
    if str(checkpoint.get("model_index_sha256")) != str(report["model_index_sha256"]):
        raise ValueError("Checkpoint model-index SHA-256 differs from the primary report.")
    if str(checkpoint.get("label_manifest_sha256")) != str(report["label_manifest_sha256"]):
        raise ValueError("Checkpoint label-manifest SHA-256 differs from the primary report.")
    if str(checkpoint.get("normalization_stats_sha256")) != normalization_stats.stats_sha256:
        raise ValueError("Checkpoint normalization SHA-256 differs from the supplied statistics.")

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing the fixed ResNet state_dict.")
    state_hash = _state_dict_sha256(state_dict)
    contract = ResNet1DContract(seed=int(checkpoint["random_seed"]))
    contract_hash = contract.sha256()
    if str(checkpoint.get("state_dict_sha256")) != state_hash:
        raise ValueError("Checkpoint state_dict SHA-256 verification failed.")
    if str(checkpoint.get("resnet_contract_sha256")) != contract_hash:
        raise ValueError("Checkpoint ResNet contract SHA-256 verification failed.")
    combined_hash = _combined_model_hash(state_hash, calibration_sha256, contract_hash)
    if str(checkpoint.get("combined_model_sha256")) != combined_hash:
        raise ValueError("Checkpoint combined-model SHA-256 verification failed.")
    if combined_hash != str(report["model_sha256"]):
        raise ValueError("Checkpoint/calibration state differs from the primary Phase-0 model.")

    candidate_pairs = {(item.source, item.label_code) for item in plan.candidate_pairs}
    candidate_sources = {source for source, _ in candidate_pairs}
    recovery_rows = [
        row
        for row in rows
        if row.role == "external_recovery_pool" and row.source in candidate_sources
    ]
    if any(row.role != "external_recovery_pool" for row in recovery_rows):
        raise RuntimeError("Phase-1 loader selected a non-recovery statistical role.")
    if any(
        not any((row.source, code) in candidate_pairs for code in label_codes)
        for row in recovery_rows
    ):
        raise RuntimeError("Phase-1 loader selected a source without an eligible recovery pair.")
    for source in candidate_sources:
        if not any(row.source == source for row in recovery_rows):
            raise RuntimeError(f"Activated recovery source has no frozen recovery records: {source}")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

    root = Path(primary_data_root).expanduser().resolve()

    class RecoveryDataset(Dataset):
        def __init__(self, selected_rows: list[Any]) -> None:
            self.rows = selected_rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            standardized = load_standardized_record(root, row)
            normalized = normalize_signal(
                standardized.waveform_mv,
                standardized.valid_mask,
                normalization_stats,
            )
            return (
                torch.from_numpy(normalized),
                torch.tensor(row.labels, dtype=torch.int64),
            )

    model = FixedResNet1D(n_labels=len(label_codes), contract=contract)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    loader = DataLoader(
        RecoveryDataset(recovery_rows),
        batch_size=int(protocol["phase0_models"]["resnet1d_fixed"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    raw_logits_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs.to(device=device, dtype=torch.float32, non_blocking=True))
            raw_logits_parts.append(logits.detach().cpu().numpy())
            target_parts.append(targets.numpy().astype(np.int64, copy=False))
    if not raw_logits_parts:
        raise RuntimeError("No recovery-pool ECG batches were produced.")

    raw_logits = np.vstack(raw_logits_parts).astype(np.float64, copy=False)
    targets = np.vstack(target_parts).astype(np.int64, copy=False)
    expected_targets = np.asarray([row.labels for row in recovery_rows], dtype=np.int64)
    if not np.array_equal(targets, expected_targets):
        raise RuntimeError("Recovery DataLoader order changed relative to the frozen model index.")
    base_probabilities = _global_calibrated_probabilities(
        raw_logits,
        calibration_payload,
        label_codes,
    )

    phase1_protocol = protocol["phase1_if_phase0_passes"]
    min_positive_eval = int(
        phase1_protocol["evaluation"]["minimum_positive_evaluation_records_per_label"]
    )
    min_negative_eval = int(
        phase1_protocol["evaluation"]["minimum_negative_evaluation_records_per_label"]
    )
    repeats = int(phase1_protocol["sampling"]["repeats"])
    seed = int(phase1_protocol["sampling"]["seed"])
    budgets = tuple(int(value) for value in phase1_protocol["target_label_budgets"])

    pair_results: dict[str, dict[str, Any]] = {}
    for candidate in plan.candidate_pairs:
        source = candidate.source
        code = candidate.label_code
        source_indices = np.asarray(
            [index for index, row in enumerate(recovery_rows) if row.source == source],
            dtype=np.int64,
        )
        label_index = label_codes.index(code)
        y_source = targets[source_indices, label_index]
        p_source = base_probabilities[source_indices, label_index]
        base_logits = _probability_logit(p_source)
        source_key = f"{source}/{code}"
        budget_results: dict[str, Any] = {}

        for budget in budgets:
            method_records: dict[str, list[dict[str, Any]]] = {
                method: [] for method in _ALLOWED_METHODS
            }
            used_budget = min(int(budget), int(y_source.size))
            exceeds_pool = int(budget) > int(y_source.size)

            for repeat in range(repeats):
                adaptation_indices = _stable_draw_indices(
                    int(y_source.size),
                    int(budget),
                    seed=seed,
                    source=source,
                    label_code=code,
                    repeat=repeat,
                )
                evaluation_mask = np.ones(y_source.size, dtype=bool)
                evaluation_mask[adaptation_indices] = False
                evaluation_indices = np.flatnonzero(evaluation_mask)
                y_eval = y_source[evaluation_indices]
                p_eval = p_source[evaluation_indices]
                positives = int(y_eval.sum())
                negatives = int(y_eval.size - positives)
                evaluation_supported = (
                    positives >= min_positive_eval and negatives >= min_negative_eval
                )

                if not evaluation_supported:
                    reason = "phase1_evaluation_support_below_threshold"
                    for method in _ALLOWED_METHODS:
                        method_records[method].append(
                            {
                                "estimable": False,
                                "nonestimable_reason": reason,
                            }
                        )
                    continue

                frozen_metrics = evaluate_binary_probabilities(y_eval, p_eval)
                method_records["frozen_no_update"].append(
                    {
                        "estimable": True,
                        "nonestimable_reason": None,
                        "metrics": asdict(frozen_metrics),
                        "recovery_envelope_met": _recovery_envelope_met(
                            frozen_metrics,
                            protocol,
                        ),
                    }
                )

                if budget == 0:
                    for method in (
                        "intercept_only_recalibration",
                        "platt_recalibration",
                    ):
                        method_records[method].append(
                            {
                                "estimable": False,
                                "nonestimable_reason": "zero_label_budget_no_local_update",
                            }
                        )
                    continue

                y_adapt = y_source[adaptation_indices]
                if np.unique(y_adapt).size != 2:
                    for method in (
                        "intercept_only_recalibration",
                        "platt_recalibration",
                    ):
                        method_records[method].append(
                            {
                                "estimable": False,
                                "nonestimable_reason": "adaptation_sample_single_class",
                            }
                        )
                    continue

                eval_logits = base_logits[evaluation_indices]
                adapt_logits = base_logits[adaptation_indices]
                shift = _fit_intercept_only(adapt_logits, y_adapt)
                intercept_probs = _sigmoid(eval_logits + shift)
                intercept_metrics = evaluate_binary_probabilities(y_eval, intercept_probs)
                method_records["intercept_only_recalibration"].append(
                    {
                        "estimable": True,
                        "nonestimable_reason": None,
                        "metrics": asdict(intercept_metrics),
                        "recovery_envelope_met": _recovery_envelope_met(
                            intercept_metrics,
                            protocol,
                        ),
                    }
                )

                slope, intercept = _fit_local_platt(
                    adapt_logits,
                    y_adapt,
                    seed=seed,
                )
                platt_probs = _sigmoid(slope * eval_logits + intercept)
                platt_metrics = evaluate_binary_probabilities(y_eval, platt_probs)
                method_records["platt_recalibration"].append(
                    {
                        "estimable": True,
                        "nonestimable_reason": None,
                        "metrics": asdict(platt_metrics),
                        "recovery_envelope_met": _recovery_envelope_met(
                            platt_metrics,
                            protocol,
                        ),
                    }
                )

            budget_results[str(budget)] = {
                "requested_label_budget": int(budget),
                "used_label_budget": used_budget,
                "budget_exceeds_available_pool": exceeds_pool,
                "recovery_pool_records": int(y_source.size),
                "methods": {
                    method: _summarize_repeats(records)
                    for method, records in method_records.items()
                },
            }

        pair_results[source_key] = {
            "source": source,
            "label_code": code,
            "phase0_status": "calibration_recovery_candidate",
            "recovery_pool_records": int(y_source.size),
            "recovery_pool_positives": int(y_source.sum()),
            "recovery_pool_negatives": int(y_source.size - y_source.sum()),
            "budgets": budget_results,
        }

    payload: dict[str, Any] = {
        "report_version": "0.1.0",
        "study": "TRUST-ECG",
        "stage": "conditional_phase1_label_efficient_probability_recovery",
        "protocol_version": str(report["protocol_version"]),
        "protocol_sha256": str(report["protocol_sha256"]),
        "phase0_report_sha256": str(report["report_sha256"]),
        "phase0_model_sha256": str(report["model_sha256"]),
        "model_index_sha256": str(report["model_index_sha256"]),
        "label_manifest_sha256": str(report["label_manifest_sha256"]),
        "normalization_stats_sha256": normalization_stats.stats_sha256,
        "global_calibration_sha256": calibration_sha256,
        "phase1_plan": plan.to_dict(),
        "candidate_pair_count": len(plan.candidate_pairs),
        "candidate_domain_count": len(plan.candidate_domains),
        "recovery_pool_only": True,
        "target_domain_model_retraining": False,
        "target_domain_feature_selection": False,
        "target_domain_normalization_refit": False,
        "posthoc_threshold_search": False,
        "pair_results": pair_results,
        "report_sha256": "",
    }
    payload["report_sha256"] = _canonical_hash(payload, "report_sha256")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
