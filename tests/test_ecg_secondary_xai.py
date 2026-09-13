from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from trust_icu.ecg_secondary_xai import (  # noqa: E402
    deterministic_coverage_indices,
    grad_cam_1d,
    integrated_gradients,
    occlude_lead,
    occlude_time_window,
)


class TinyWaveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, 4, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(4, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.relu(self.conv(x))
        return self.head(self.pool(features).squeeze(-1))


def test_deterministic_coverage_indices_obey_hard_caps() -> None:
    labels = np.zeros((100, 3), dtype=np.int64)
    labels[:20, 0] = 1
    labels[20:40, 1] = 1
    labels[40:60, 2] = 1
    first = deterministic_coverage_indices(labels, max_samples=64, seed=20260808)
    second = deterministic_coverage_indices(labels, max_samples=64, seed=20260808)
    assert np.array_equal(first, second)
    assert len(first) == 64
    assert len(np.unique(first)) == 64
    assert labels[first].sum(axis=0).min() > 0


def test_occlusion_targets_only_requested_lead_or_window() -> None:
    x = torch.arange(2 * 2 * 20, dtype=torch.float32).reshape(2, 2, 20)
    lead = occlude_lead(x, 1)
    assert torch.count_nonzero(lead[:, 1]) == 0
    assert torch.equal(lead[:, 0], x[:, 0])

    window = occlude_time_window(x, 5, 10)
    assert torch.count_nonzero(window[:, :, 5:10]) == 0
    assert torch.equal(window[:, :, :5], x[:, :, :5])
    assert torch.equal(window[:, :, 10:], x[:, :, 10:])


def test_integrated_gradients_is_finite_and_enforces_caps() -> None:
    torch.manual_seed(1)
    model = TinyWaveModel().eval()
    x = torch.randn(2, 2, 32)
    attribution = integrated_gradients(model, x, target_index=1, steps=8)
    assert attribution.shape == x.shape
    assert torch.isfinite(attribution).all()

    with pytest.raises(ValueError, match="24"):
        integrated_gradients(model, x, target_index=1, steps=25)
    with pytest.raises(ValueError, match="8"):
        integrated_gradients(model, torch.randn(9, 2, 32), target_index=1, steps=8)


def test_grad_cam_returns_normalized_temporal_map() -> None:
    torch.manual_seed(2)
    model = TinyWaveModel().eval()
    x = torch.randn(2, 2, 32)
    cam = grad_cam_1d(model, x, target_index=0, target_layer=model.conv)
    assert cam.shape == (2, 32)
    assert torch.isfinite(cam).all()
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0 + 1e-6

    with pytest.raises(ValueError, match="8"):
        grad_cam_1d(model, torch.randn(9, 2, 32), target_index=0, target_layer=model.conv)
