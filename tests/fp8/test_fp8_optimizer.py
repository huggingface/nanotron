import pytest
import torch
from nanotron.optim.gradient_accumulator import FP32GradientAccumulator
from nanotron.optim.named_optimizer import NamedOptimizer
from nanotron.optim.optimizer_from_gradient_accumulator import (
    OptimizerFromGradientAccumulator,
)
from nanotron.parallel.parameters import NanotronParameter, sanity_check
from torch import nn


@pytest.mark.parametrize("half_precision", [torch.float16, torch.bfloat16])
def test_optimizer_can_step_gradient_in_fp32(half_precision: torch.dtype):
    model = nn.Linear(3, 2, bias=False, dtype=half_precision, device="cuda")
    original_weight = model.weight.detach().clone()

    # Create Nanotron Parameter
    model.weight = NanotronParameter(model.weight)

    # Add optimizer
    optimizer = OptimizerFromGradientAccumulator(
        gradient_accumulator_builder=lambda named_params: FP32GradientAccumulator(named_parameters=named_params),
        named_params_or_groups=model.named_parameters(),
        optimizer_builder=lambda named_param_groups: NamedOptimizer(
            named_params_or_groups=named_param_groups,
            optimizer_builder=lambda param_groups: torch.optim.AdamW(param_groups),
        ),
    )
    accumulator = optimizer.gradient_accumulator

    # Check that our model is a valid model
    sanity_check(model)

    # Compute backward
    input = torch.randn(5, 3, dtype=half_precision, device="cuda")
    accumulator.backward(model(input).sum())

    # Check that we have an high precision gradient and that the low precision one is cleared
    assert accumulator.parameters["weight"]["fp32"].grad.dtype == torch.float
    if model.weight.grad is not None:
        # We check that it's zero
        torch.testing.assert_close(model.weight.grad, torch.zeros_like(model.weight.grad), atol=1e-6, rtol=1e-7)

    optimizer.step()
    optimizer.zero_grad()

    # Check that we don't have gradients anymore and that it's set to `None`
    assert accumulator.parameters["weight"]["fp32"].grad is None
    assert model.weight.grad is None

    # Check that gradients have been set to zero
    fp32_grad = accumulator.get_grad_buffer(name="weight")
    torch.testing.assert_close(fp32_grad, torch.zeros_like(fp32_grad), atol=1e-6, rtol=1e-7)

    # weights has been updates
    assert not torch.allclose(original_weight, model.weight)
