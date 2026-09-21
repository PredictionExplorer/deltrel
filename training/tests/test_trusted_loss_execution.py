"""Trusted replay retains exact supervision without dynamic validation gathers."""

from dataclasses import replace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from deltreltrain.losses import LossWeights, compute_losses
from deltreltrain.model import DeltrelModelOutput
from test_losses import outputs, targets


class RecordOperations(TorchDispatchMode):
    def __init__(self):
        self.operations = []
        super().__init__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations.append(func)
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("teacher", [False, True])
@pytest.mark.parametrize("supervision", ["full", "partial", "none"])
def test_trusted_losses_preserve_values_gradients_and_diagnostics(teacher, supervision):
    generator = torch.Generator().manual_seed(1729)
    reference = DeltrelModelOutput(
        *(
            torch.randn(t.shape, generator=generator).requires_grad_()
            for t in outputs(4, 5, 5)[:6]
        )
    )
    trusted = DeltrelModelOutput(
        *(t.detach().clone().requires_grad_() for t in reference[:6])
    )
    target = targets(4, 5, 5)
    target.policy[:, 0] = 1
    target.soft_policy[:, 1] = 1
    target.outcome[:] = torch.tensor([0, 1, 0, 1])
    target.score_margin[:] = torch.tensor([-100, 0, 3, 151])
    target.ownership[:] = torch.arange(5) % 3
    target.alive[:] = torch.arange(5) % 2
    available = {
        "full": [True, True, True, True],
        "partial": [True, False, True, False],
        "none": [False, False, False, False],
    }[supervision]
    for name in (
        "policy",
        "soft_policy",
        "outcome",
        "score_margin",
        "ownership",
        "alive",
    ):
        getattr(target, f"{name}_mask")[:] = torch.tensor(available)
    target = replace(
        target,
        sample_weight=torch.tensor([0.25, 0.0, 2.0, 0.5]),
        policy_weight=torch.tensor([0.5, 1.0, 0.0, 2.0]),
        clinch_mask=torch.tensor([True, False, True, False]),
        teacher_mask=torch.tensor(available) if teacher else None,
        teacher_policy=torch.full((4, 5), 0.2) if teacher else None,
        teacher_outcome=torch.full((4, 2), 0.5) if teacher else None,
        teacher_score_margin=torch.full((4, 303), 1 / 303) if teacher else None,
    )
    node_mask = torch.ones(4, 5, dtype=torch.bool)
    node_mask[2:, -1] = False
    options = dict(
        legal_action_mask=node_mask,
        node_mask=node_mask,
        weights=LossWeights(
            teacher_policy=0.2, teacher_outcome=0.3, teacher_score_margin=0.1
        ),
        include_diagnostics=True,
    )
    expected = compute_losses(reference, target, **options)
    with RecordOperations() as trace:
        actual = compute_losses(trusted, target, validate_targets=False, **options)
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    expected["total"].backward()
    actual["total"].backward()
    for left, right in zip(reference[:6], trusted[:6], strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    # Boolean indexing makes output sizes data-dependent and can synchronize a
    # CUDA stream. Trusted replay has already validated these labels on the CPU.
    assert torch.ops.aten.index.Tensor not in trace.operations
    assert torch.ops.aten.nonzero.default not in trace.operations
