"""Fixed, non-novel 1D ResNet execution contract for the TRUST-ECG primary baseline.

This module is intentionally optional because PyTorch is a heavy dependency. The architecture is
frozen by the prospective ECG protocol; this file implements that contract but does not perform
architecture search or target-domain adaptation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
    raise RuntimeError(
        'PyTorch is required for the optional TRUST-ECG deep baseline. Install with `pip install -e ".[ecg-deep]"`.'
    ) from exc


@dataclass(frozen=True)
class ResNet1DContract:
    in_channels: int = 12
    input_samples: int = 5000
    stem_channels: int = 64
    stem_kernel: int = 15
    stem_stride: int = 2
    stem_padding: int = 7
    max_pool_kernel: int = 3
    max_pool_stride: int = 2
    max_pool_padding: int = 1
    stage_channels: tuple[int, ...] = (64, 128, 256, 512)
    blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2)
    residual_kernel: int = 7
    downsample_first_block_of_stages: tuple[bool, ...] = (False, True, True, True)
    seed: int = 20260808

    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


class BasicBlock1D(nn.Module):
    """Two-convolution residual block used by the frozen ResNet18-style 1D baseline."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class FixedResNet1D(nn.Module):
    """Prospectively specified multi-label ECG backbone with no tunable architecture choices."""

    def __init__(self, n_labels: int, contract: ResNet1DContract | None = None) -> None:
        super().__init__()
        if n_labels <= 0:
            raise ValueError("n_labels must be positive.")
        self.contract = contract or ResNet1DContract()
        c = self.contract
        if len(c.stage_channels) != 4 or c.blocks_per_stage != (2, 2, 2, 2):
            raise ValueError("TRUST-ECG requires the locked four-stage 2-2-2-2 ResNet contract.")
        if c.downsample_first_block_of_stages != (False, True, True, True):
            raise ValueError("TRUST-ECG downsampling schedule cannot drift.")

        self.stem = nn.Sequential(
            nn.Conv1d(
                c.in_channels,
                c.stem_channels,
                c.stem_kernel,
                stride=c.stem_stride,
                padding=c.stem_padding,
                bias=False,
            ),
            nn.BatchNorm1d(c.stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=c.max_pool_kernel,
                stride=c.max_pool_stride,
                padding=c.max_pool_padding,
            ),
        )

        stages: list[nn.Module] = []
        in_channels = c.stem_channels
        for stage_index, (out_channels, blocks) in enumerate(
            zip(c.stage_channels, c.blocks_per_stage, strict=True)
        ):
            stride = 2 if c.downsample_first_block_of_stages[stage_index] else 1
            stage_blocks: list[nn.Module] = [
                BasicBlock1D(
                    in_channels,
                    out_channels,
                    stride=stride,
                    kernel_size=c.residual_kernel,
                )
            ]
            stage_blocks.extend(
                BasicBlock1D(
                    out_channels,
                    out_channels,
                    stride=1,
                    kernel_size=c.residual_kernel,
                )
                for _ in range(1, blocks)
            )
            stages.append(nn.Sequential(*stage_blocks))
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(c.stage_channels[-1], n_labels)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.contract.in_channels:
            raise ValueError("Input must have shape (batch, 12, time).")
        if x.shape[2] != self.contract.input_samples:
            raise ValueError(
                f"Input time dimension must be {self.contract.input_samples}, found {x.shape[2]}."
            )
        out = self.stem(x)
        out = self.stages(out)
        out = self.pool(out).squeeze(-1)
        return self.head(out)


def set_torch_determinism(seed: int = 20260808) -> None:
    """Apply the locked research seed and deterministic-algorithm request."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def compute_positive_class_weights(y: np.ndarray) -> torch.Tensor:
    """Compute BCE positive weights from development folds only.

    Callers are responsible for passing PTB-XL folds 1-7 only. The function fails closed when a
    locked diagnosis lacks either class in the supplied model-fitting data.
    """

    targets = np.asarray(y, dtype=np.int64)
    if targets.ndim != 2 or not np.isin(targets, [0, 1]).all():
        raise ValueError("Positive class weights require a two-dimensional binary target matrix.")
    positives = targets.sum(axis=0)
    negatives = targets.shape[0] - positives
    if np.any(positives == 0) or np.any(negatives == 0):
        raise ValueError("Every locked diagnosis must contain positives and negatives in model-fitting data.")
    return torch.tensor(negatives / positives, dtype=torch.float32)


def macro_pr_auc_from_logits(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Compute the locked fold-8 early-stopping metric from raw multi-label logits."""

    targets = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(logits, dtype=np.float64)
    if targets.ndim != 2 or scores.shape != targets.shape:
        raise ValueError("Fold-8 targets and logits must have identical two-dimensional shape.")
    if not np.isin(targets, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Fold-8 metric requires binary targets and finite logits.")
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -60.0, 60.0)))
    values: list[float] = []
    for index in range(targets.shape[1]):
        if np.unique(targets[:, index]).size != 2:
            raise ValueError(
                "Every locked diagnosis must contain both classes in fold 8 for the primary stopping metric."
            )
        values.append(float(average_precision_score(targets[:, index], probabilities[:, index])))
    return float(np.mean(values))


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
