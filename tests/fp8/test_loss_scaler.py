from copy import deepcopy

import pytest
import torch
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.loss_scaler import LossScaler, is_overflow
from nanotron.fp8.tensor import convert_tensor_from_fp8
from torch import nn
from torch.optim import Adam
from utils import convert_linear_to_fp8


def test_loss_scaler_attributes():
    scaling_value = torch.tensor(1.0)
    scaling_factor = torch.tensor(2.0)
    interval = 10

    loss_scaler = LossScaler(scaling_value, scaling_factor, interval)

    assert loss_scaler.scaling_value == scaling_value
    assert loss_scaler.scaling_factor == scaling_factor
    assert loss_scaler.interval == interval


# def test_loss_scaler_auto_cast_loss_to_fp32():
#     input = torch.randn(4, 4)
#     linear = nn.Linear(4, 4)
#     loss_scaler = LossScaler()

#     scaled_loss = loss_scaler.scale(linear(input).sum())

#     assert scaled_loss.dtype == torch.float32


def test_gradients_correctness():
    input = torch.randn(4, 4)
    linear = nn.Linear(4, 4)
    ref_linear = deepcopy(linear)
    loss_scaler = LossScaler()

    scaled_loss = loss_scaler.scale(linear(input).sum())
    loss = ref_linear(input).sum()

    assert not torch.allclose(scaled_loss, loss)

    loss.backward()
    scaled_loss.backward()

    assert not torch.allclose(linear.weight.grad, ref_linear.weight.grad)
    assert not torch.allclose(linear.bias.grad, ref_linear.bias.grad)

    assert torch.equal(linear.weight.grad, ref_linear.weight.grad * loss_scaler.scaling_value)
    assert torch.equal(linear.bias.grad, ref_linear.bias.grad * loss_scaler.scaling_value)


def test_unscale_scaled_gradients():
    input = torch.randn(4, 4)
    linear = nn.Linear(4, 4)
    ref_linear = deepcopy(linear)
    loss_scaler = LossScaler()

    scaled_loss = loss_scaler.scale(linear(input).sum())
    loss = ref_linear(input).sum()

    loss.backward()
    scaled_loss.backward()
    loss_scaler.unscale_(linear.parameters())

    assert torch.equal(linear.weight.grad, ref_linear.weight.grad)
    assert torch.equal(linear.bias.grad, ref_linear.bias.grad)


def test_fp8_gradients_correctness():
    SCALING_VALUE = torch.tensor(10.0, dtype=torch.float32)
    input = torch.randn(16, 16, device="cuda")
    fp32_linear = nn.Linear(16, 16, device="cuda")

    loss_scaler = LossScaler(scaling_value=SCALING_VALUE)
    linear = convert_linear_to_fp8(deepcopy(fp32_linear), DTypes.KFLOAT16)
    ref_linear = convert_linear_to_fp8(deepcopy(fp32_linear), DTypes.KFLOAT16)

    ref_loss = ref_linear(input).sum()
    scaled_loss = loss_scaler.scale(linear(input).sum())

    assert torch.equal(scaled_loss, ref_loss.to(torch.float32) * SCALING_VALUE)

    ref_loss.backward()
    scaled_loss.backward()

    assert not torch.equal(linear.weight.grad, ref_linear.weight.grad)
    assert not torch.equal(linear.bias.grad, ref_linear.bias.grad)

    loss_scaler.unscale_(linear.parameters())
    weight_grad = convert_tensor_from_fp8(linear.weight.grad, linear.weight.grad.fp8_meta, torch.float32)
    ref_weight_grad = convert_tensor_from_fp8(ref_linear.weight.grad, ref_linear.weight.grad.fp8_meta, torch.float32)

    # NOTE: because the scaling value is 10, so if the scaled gradients aren't scaled back
    # properly, the absolute difference would be around 10, we use 1 for sanity check
    assert torch.allclose(weight_grad, ref_weight_grad, atol=1.0)
    assert torch.equal(linear.bias.grad, ref_linear.bias.grad)


def test_loss_scaler_step():
    input = torch.randn(4, 4)
    linear = nn.Linear(4, 4)
    ref_linear = deepcopy(linear)
    optim = Adam(linear.parameters())
    ref_optim = Adam(linear.parameters())

    loss_scaler = LossScaler()

    linear(input).sum().backward()
    loss_scaler.scale(linear(input).sum()).backward()

    assert torch.allclose(linear.weight, ref_linear.weight)
    assert torch.allclose(linear.bias, ref_linear.bias)

    optim.step()
    loss_scaler.step(ref_optim)

    assert torch.allclose(linear.weight, linear.weight)
    assert torch.allclose(linear.bias, linear.bias)


def test_not_update_scaling_factor_until_calling_it():
    INTERVAL = 1

    input = torch.randn(4, 4)
    linear = nn.Linear(4, 4)
    optim = Adam(linear.parameters())
    loss_scaler = LossScaler(interval=INTERVAL)
    initial_scaling_value = deepcopy(loss_scaler.scaling_value)

    loss_scaler.scale(linear(input).sum()).backward()
    linear.weight.grad[0] = torch.tensor(float("inf"))

    assert torch.allclose(loss_scaler.scaling_value, initial_scaling_value)

    loss_scaler.step(optim)

    assert not torch.allclose(loss_scaler.scaling_value, initial_scaling_value)


def test_not_update_parameters_when_overflow_is_detected():
    input = torch.randn(4, 4)
    linear = nn.Linear(4, 4)
    ref_linear = deepcopy(linear)
    optim = Adam(linear.parameters())
    loss_scaler = LossScaler()

    linear(input).sum().backward()
    loss_scaler.scale(linear(input).sum()).backward()

    assert torch.allclose(linear.weight, ref_linear.weight)
    assert torch.allclose(linear.bias, ref_linear.bias)

    # NOTE: we intentionally set only 1 gradient to be inf
    linear.weight.grad[0] = torch.tensor(float("inf"))
    loss_scaler.step(optim)

    assert torch.allclose(linear.weight, ref_linear.weight)
    assert torch.allclose(linear.bias, ref_linear.bias)


@pytest.mark.parametrize("interval", [1, 5, 10])
def test_deplay_update_scaling_factor(interval):
    input = torch.randn(4, 4)
    linear = nn.Linear(4, 4)
    optim = Adam(linear.parameters())
    loss_scaler = LossScaler(interval=interval)
    current_scaling_value = deepcopy(loss_scaler.scaling_value)

    for i in range(1, 20):
        loss_scaler.scale(linear(input).sum()).backward()
        # NOTE: we set the gradients to be inf
        # so that we can test the overflow detection
        linear.weight.grad[0] = torch.tensor(float("inf"))

        loss_scaler.step(optim)
        loss_scaler.update()

        if i % interval == 0:
            assert not torch.allclose(loss_scaler.scaling_value, current_scaling_value)
            current_scaling_value = deepcopy(loss_scaler.scaling_value)
        else:
            assert torch.allclose(loss_scaler.scaling_value, current_scaling_value)


@pytest.mark.parametrize(
    "tensor, expected_output",
    [
        [torch.tensor(1.0), False],
        [torch.tensor(1e308).pow(2), True],
        [torch.tensor(float("inf")), True],
        [torch.randn(2, 3), True],
    ],
)
def test_overflow(tensor, expected_output):
    if tensor.ndim > 1:
        tensor[0, 0] = torch.tensor(float("inf"))

    output = is_overflow(tensor)

    assert isinstance(output, bool)
    assert output is expected_output


# TODO(xrsrke): test decrease the scaling factor when overflow is detected

# TODO(xrsrke): test increase the scaling factor when no overflow is detected for n intervals

# TODO(xrsrke): test that we don't scale the gradients of parameters that are not updated
# by the optimizer

# TODO(xrsrke): support detect overflow in TP
