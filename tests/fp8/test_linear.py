from copy import deepcopy

import pytest
import torch
from nanotron.fp8 import FP8Linear, FP8Parameter
from torch import nn
from torch.optim import Adam
from utils import convert_linear_to_fp8


@pytest.mark.parametrize("is_bias", [True, False])
def test_fp8_linear_forward_pass(is_bias):
    input = torch.randn(16, 16, device="cuda", dtype=torch.float32)
    ref_input = input.detach().clone()
    ref_linear = nn.Linear(16, 16, bias=is_bias, device="cuda", dtype=torch.float32)

    fp8_linear = convert_linear_to_fp8(deepcopy(ref_linear))

    ref_output = ref_linear(ref_input)
    output = fp8_linear(input)

    assert isinstance(output, torch.Tensor)
    assert output.dtype == torch.float32

    # NOTE: this threshold is from fp8-lm, the paper shows that this is fine
    torch.testing.assert_allclose(output, ref_output, rtol=0, atol=0.1)


# TODO(xrsrke): add cases where the input requires and don't require grad
@pytest.mark.parametrize("input_requires_grad", [True, False])
def test_fp8_linear_backward_pass(input_requires_grad):
    input = torch.randn(16, 16, device="cuda", dtype=torch.float32, requires_grad=input_requires_grad)
    ref_input = input.detach().clone().requires_grad_(True)
    ref_linear = nn.Linear(16, 16, device="cuda", dtype=torch.float32)

    fp8_linear = convert_linear_to_fp8(deepcopy(ref_linear))

    ref_linear(ref_input).sum().backward()
    fp8_linear(input).sum().backward()

    # NOTE: this threshold is from fp8-lm, the paper shows that this is fine
    # TODO(xrsrke): investigate why input.grad is so high tolerance
    # assert torch.allclose(input.grad, ref_input.grad, 0.2, 0.2) if input_requires_grad else True
    # TODO(xrsrke): tune what is the minimum threshold for this to correctly converge
    torch.testing.assert_close(fp8_linear.weight.grad, ref_linear.weight.grad, rtol=0.1, atol=0.1)
    torch.testing.assert_close(fp8_linear.bias.grad, ref_linear.bias.grad, rtol=0, atol=0.1)


def test_fp8_linear_attrs():
    fp8_linear = FP8Linear(16, 16, device="cuda")

    assert next(fp8_linear.parameters()) is not None
    assert all(p.requires_grad for p in fp8_linear.parameters()) is True


def test_fp8_model_bwd():
    HIDEEN_SIZE = 128
    N_LAYERS = 5
    N_EPOCHS = 3

    input = torch.randn(HIDEEN_SIZE, HIDEEN_SIZE, device="cuda", requires_grad=True)

    model = nn.Sequential(
        *[nn.Sequential(FP8Linear(HIDEEN_SIZE, HIDEEN_SIZE, device="cuda"), nn.ReLU()) for _ in range(N_LAYERS)]
    )
    optim = Adam(model.parameters(), lr=1e-3)

    for _ in range(N_EPOCHS):
        optim.zero_grad()
        model(input).sum().backward()
        optim.step()

    assert all(p.grad is not None for p in model.parameters())


# NOTE: it seems that dynamic quantization should be in test_tensor
# but we only do this if we are in training => test it in a linear
@pytest.mark.parametrize("interval", [1, 5, 10])
def test_deplay_quantization(interval):
    # NOTE: test delay quantization (window size)
    # NOTE: test overflow, underflow, zeros
    # NOTE: test reduce/increase exponent bits

    HIDDEN_SIZE = 16
    N_STEPS = 4

    input = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    fp8_linear = FP8Linear(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda")

    for _ in range(N_STEPS):
        output = fp8_linear(input)
        output.sum().backward()


def test_linear_parameters():
    ref_linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(deepcopy(ref_linear))

    assert len(list(ref_linear.parameters())) == len(list(fp8_linear.parameters()))
    assert [isinstance(p, FP8Parameter) for p in fp8_linear.parameters()]


# TODO(xrsrke): test if FP8Linear has all the methods of a torch.nn.Linear


# TODO(xrsrke): test only calculating the gradients of the weight, bias, or input based
# on the requires_grad of the input, weight, or bias
