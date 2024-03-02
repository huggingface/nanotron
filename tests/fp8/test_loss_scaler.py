from copy import deepcopy

import pytest
import torch
from nanotron.fp8.loss_scaler import LossScaler, is_overflow
from torch import nn
from torch.optim import Adam


def test_loss_scaler_attributes():
    loss_scaler = LossScaler()

    assert isinstance(loss_scaler.scaling_value, torch.Tensor)
    assert isinstance(loss_scaler.scaling_factor, torch.Tensor)
    assert isinstance(loss_scaler.interval, int)
    assert isinstance(loss_scaler.overflow_counter, int)


def test_gradients_correctness():
    linear = nn.Linear(4, 4)
    ref_linear = deepcopy(linear)
    input = torch.randn(4, 4)
    loss = linear(input).sum()

    loss_scaler = LossScaler()
    scaled_loss = loss_scaler.scale(ref_linear(input).sum())

    assert not torch.allclose(scaled_loss, loss)

    loss.backward()
    scaled_loss.backward()

    assert not torch.allclose(linear.weight.grad, ref_linear.weight.grad)
    assert not torch.allclose(linear.bias.grad, ref_linear.bias.grad)


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
    interval = 1

    input = torch.randn(4, 4)
    linear = nn.Linear(4, 4)
    optim = Adam(linear.parameters())
    loss_scaler = LossScaler(interval=interval)
    initial_scaling_value = deepcopy(loss_scaler.scaling_value)

    loss_scaler.scale(linear(input).sum()).backward()
    loss_scaler.step(optim)

    assert torch.allclose(loss_scaler.scaling_value, initial_scaling_value)

    loss_scaler.step(optim)

    assert not torch.allclose(loss_scaler.scaling_value, initial_scaling_value)


def test_delay_scaling():
    pass


def test_not_update_parameters_when_overflow_is_detected():
    input = torch.randn(4, 4)
    linear = nn.Linear(4, 4)
    optim = Adam(linear.parameters())
    loss_scaler = LossScaler()
    deepcopy(loss_scaler.scaling_value)

    linear(input).sum().backward()
    loss_scaler.scale(linear(input).sum()).backward()

    assert torch.allclose(linear.weight, linear.weight)
    assert torch.allclose(linear.bias, linear.bias)

    loss_scaler.step(optim)

    assert torch.allclose(linear.weight, linear.weight)
    assert torch.allclose(linear.bias, linear.bias)


@pytest.mark.parametrize("tensor, output", [[torch.tensor(1.0), False], [torch.tensor(1e308).pow(2), True]])
def test_overflow(tensor, output):
    assert is_overflow(tensor) == output


# TODO(xrsrke): test that we don't scale the gradients of parameters that are not updated
# by the optimizer

# TODO(xrsrke): support detect overflow in TP
