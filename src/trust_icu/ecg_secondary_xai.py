"""Direct frozen-ResNet explainability utilities for TRUST-ECG.

The methods here are descriptive secondary analyses. They do not train a
surrogate model and do not alter the primary ResNet weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as functional
except ImportError as exc:  # pragma: no cover - optional heavy runtime
    raise RuntimeError(
        'PyTorch is required for TRUST-ECG XAI. Install with `pip install -e ".[ecg-deep]"`.'
    ) from exc

MAX_XAI_BATCH = 8
MAX_IG_STEPS = 24
MAX_OCCLUSION_SAMPLE = 64


def deterministic_coverage_indices(
    labels: np.ndarray,
    *,
    max_samples: int = MAX_OCCLUSION_SAMPLE,
    seed: int = 20260808,
) -> np.ndarray:
    """Select deterministic samples while attempting positive/negative label coverage."""

    targets = np.asarray(labels, dtype=np.int64)
    if targets.ndim != 2 or targets.shape[0] == 0:
        raise ValueError("labels must have shape (records, labels) with at least one record.")
    if max_samples <= 0 or max_samples > MAX_OCCLUSION_SAMPLE:
        raise ValueError(f"max_samples must lie in [1, {MAX_OCCLUSION_SAMPLE}].")
    n = targets.shape[0]
    if n <= max_samples:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(n)
    selected: list[int] = []
    used: set[int] = set()

    def add_first(candidates: np.ndarray) -> None:
        candidate_set = set(int(item) for item in candidates.tolist())
        for index in permutation:
            item = int(index)
            if item in candidate_set and item not in used and len(selected) < max_samples:
                selected.append(item)
                used.add(item)
                return

    for label_index in range(targets.shape[1]):
        add_first(np.flatnonzero(targets[:, label_index] == 1))
        add_first(np.flatnonzero(targets[:, label_index] == 0))

    for index in permutation:
        item = int(index)
        if len(selected) >= max_samples:
            break
        if item not in used:
            selected.append(item)
            used.add(item)
    return np.asarray(selected, dtype=np.int64)


def _validate_tensor(inputs: Any) -> None:
    if not torch.is_tensor(inputs) or inputs.ndim != 3:
        raise ValueError("inputs must be a torch tensor with shape (batch, leads, time).")
    if inputs.shape[0] < 1:
        raise ValueError("inputs must contain at least one ECG.")
    if not torch.is_floating_point(inputs):
        raise ValueError("inputs must use a floating-point dtype.")


def occlude_lead(inputs: Any, lead_index: int) -> Any:
    """Return a clone with exactly one lead replaced by zeros."""

    _validate_tensor(inputs)
    index = int(lead_index)
    if index < 0 or index >= inputs.shape[1]:
        raise ValueError("lead_index is outside the input tensor.")
    output = inputs.clone()
    output[:, index, :] = 0.0
    return output


def occlude_time_window(inputs: Any, start: int, stop: int) -> Any:
    """Return a clone with one half-open temporal interval replaced by zeros."""

    _validate_tensor(inputs)
    begin = int(start)
    end = int(stop)
    if begin < 0 or end <= begin or end > inputs.shape[-1]:
        raise ValueError("Temporal occlusion window is outside the signal.")
    output = inputs.clone()
    output[:, :, begin:end] = 0.0
    return output


def integrated_gradients(
    model: Any,
    inputs: Any,
    *,
    target_index: int,
    steps: int = MAX_IG_STEPS,
    baseline: Any | None = None,
) -> Any:
    """Compute batch Integrated Gradients with the fixed publication compute cap."""

    _validate_tensor(inputs)
    if inputs.shape[0] > MAX_XAI_BATCH:
        raise ValueError(f"Integrated Gradients is capped at {MAX_XAI_BATCH} ECGs per call.")
    if steps < 1 or steps > MAX_IG_STEPS:
        raise ValueError(f"Integrated Gradients steps must lie in [1, {MAX_IG_STEPS}].")
    if baseline is None:
        reference = torch.zeros_like(inputs)
    else:
        _validate_tensor(baseline)
        if baseline.shape != inputs.shape:
            raise ValueError("Integrated Gradients baseline shape must match inputs.")
        reference = baseline.to(device=inputs.device, dtype=inputs.dtype)

    was_training = bool(model.training)
    model.eval()
    delta = inputs - reference
    gradient_sum = torch.zeros_like(inputs)
    alphas = torch.linspace(
        1.0 / float(steps),
        1.0,
        int(steps),
        device=inputs.device,
        dtype=inputs.dtype,
    )
    try:
        for alpha in alphas:
            scaled = (reference + alpha * delta).detach().requires_grad_(True)
            outputs = model(scaled)
            if outputs.ndim != 2 or target_index < 0 or target_index >= outputs.shape[1]:
                raise ValueError("target_index is outside the model output.")
            score = outputs[:, int(target_index)].sum()
            gradient = torch.autograd.grad(score, scaled, retain_graph=False)[0]
            gradient_sum += gradient.detach()
    finally:
        model.train(was_training)
    return delta * (gradient_sum / float(steps))


def _default_grad_cam_layer(model: Any) -> Any:
    stages = getattr(model, "stages", None)
    if stages is None or len(stages) == 0:
        raise ValueError("target_layer is required for a model without ResNet stages.")
    return stages[-1]


def grad_cam_1d(
    model: Any,
    inputs: Any,
    *,
    target_index: int,
    target_layer: Any | None = None,
) -> Any:
    """Compute normalized temporal Grad-CAM from a 1D convolutional layer."""

    _validate_tensor(inputs)
    if inputs.shape[0] > MAX_XAI_BATCH:
        raise ValueError(f"Grad-CAM is capped at {MAX_XAI_BATCH} ECGs per call.")
    layer = target_layer if target_layer is not None else _default_grad_cam_layer(model)
    activation_holder: list[Any] = []

    def capture_activation(_module: Any, _arguments: Any, output: Any) -> None:
        if not torch.is_tensor(output) or output.ndim != 3:
            raise ValueError("Grad-CAM target layer must produce (batch, channels, time).")
        output.retain_grad()
        activation_holder.append(output)

    handle = layer.register_forward_hook(capture_activation)
    was_training = bool(model.training)
    model.eval()
    try:
        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        if outputs.ndim != 2 or target_index < 0 or target_index >= outputs.shape[1]:
            raise ValueError("target_index is outside the model output.")
        outputs[:, int(target_index)].sum().backward()
        if len(activation_holder) != 1:
            raise RuntimeError("Grad-CAM target layer was not captured exactly once.")
        activation = activation_holder[0]
        gradient = activation.grad
        if gradient is None:
            raise RuntimeError("Grad-CAM gradients were not retained.")
        weights = gradient.mean(dim=-1, keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = functional.interpolate(
            cam,
            size=inputs.shape[-1],
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        minimum = cam.amin(dim=1, keepdim=True)
        shifted = cam - minimum
        maximum = shifted.amax(dim=1, keepdim=True)
        normalized = shifted / torch.clamp(maximum, min=1e-12)
        return normalized.detach()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
