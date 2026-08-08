from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from trust_icu.ecg_resnet import (
    FixedResNet1D,
    ResNet1DContract,
    compute_positive_class_weights,
    macro_pr_auc_from_logits,
    set_torch_determinism,
    trainable_parameter_count,
)


def test_fixed_resnet_forward_shape_and_parameter_count() -> None:
    set_torch_determinism()
    model = FixedResNet1D(n_labels=7)
    model.eval()
    x = torch.zeros((2, 12, 5000), dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 7)
    assert trainable_parameter_count(model) == 8_740_999
    assert len(model.contract.sha256()) == 64


def test_resnet_rejects_wrong_time_dimension() -> None:
    model = FixedResNet1D(n_labels=3)
    with pytest.raises(ValueError, match="time dimension"):
        model(torch.zeros((1, 12, 4999)))


def test_resnet_contract_rejects_architecture_drift() -> None:
    drifted = ResNet1DContract(blocks_per_stage=(3, 4, 6, 3))
    with pytest.raises(ValueError, match="2-2-2-2"):
        FixedResNet1D(n_labels=7, contract=drifted)


def test_positive_class_weights_are_development_only_formula() -> None:
    y = np.array(
        [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
        ]
    )
    weights = compute_positive_class_weights(y)
    assert torch.allclose(weights, torch.tensor([1.0, 1.0]))


def test_macro_pr_auc_from_logits() -> None:
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    logits = np.array([[4.0, -4.0], [-4.0, 4.0], [3.0, -3.0], [-3.0, 3.0]])
    assert np.isclose(macro_pr_auc_from_logits(y, logits), 1.0)
