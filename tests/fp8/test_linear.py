import pytest
import torch
from nanotron.fp8 import DTypes, FP8Linear, FP8Parameter, FP8Tensor
from torch import nn
from torch.optim import Adam


@pytest.mark.parametrize("is_bias", [True, False])
def test_fp8_linear_forward_pass(is_bias):
    input = torch.randn(16, 16, device="cuda", dtype=torch.float32)
    ref_input = input.detach().clone()
    ref_linear = nn.Linear(16, 16, bias=is_bias, device="cuda", dtype=torch.float32)

    fp8_linear = FP8Linear(16, 16, bias=is_bias, device="cuda:0")
    fp8_linear.weight = FP8Parameter(ref_linear.weight.detach().clone(), DTypes.FP8E4M3)

    if is_bias:
        fp8_linear.bias.data = ref_linear.bias.detach().clone()

    ref_output = ref_linear(ref_input)
    output = fp8_linear(input)

    assert isinstance(output, torch.Tensor)
    assert output.dtype == torch.float32
    assert torch.allclose(output, ref_output, rtol=0, atol=0.1)


# TODO(xrsrke): add cases where the input requires and don't require grad
@pytest.mark.parametrize("input_requires_grad", [True, False])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_fp8_linear_backward_pass(input_requires_grad, device):
    input = torch.randn(16, 16, device=device, dtype=torch.float32, requires_grad=input_requires_grad)
    ref_input = input.detach().clone().requires_grad_(True)
    ref_linear = nn.Linear(16, 16, device=device, dtype=torch.float32)
    fp8_linear = FP8Linear(16, 16, device=device)

    if device == "cpu":
        fp8_linear.weight.data = ref_linear.weight.detach().clone()
    else:
        fp8_linear.weight.data = FP8Tensor(ref_linear.weight.detach().clone(), dtype=DTypes.FP8E4M3)
    fp8_linear.bias.data = ref_linear.bias.detach().clone()

    ref_linear(ref_input).sum().backward()
    fp8_linear(input).sum().backward()

    # TODO(xrsrke): investigate why input.grad is so high tolerance
    # assert torch.allclose(input.grad, ref_input.grad, 0.2, 0.2) if input_requires_grad else True
    assert torch.allclose(fp8_linear.weight.grad, ref_linear.weight.grad, 0.1, 0.1)
    assert torch.allclose(fp8_linear.bias.grad, ref_linear.bias.grad, 0, 0.1)


# TODO(xrsrke): test if FP8Linear has all the methods of a torch.nn.Linear


def test_fp8_linear_attrs():
    fp8_linear = FP8Linear(16, 16, device="cuda:0")

    assert next(fp8_linear.parameters()) is not None
    assert all(p.requires_grad for p in fp8_linear.parameters()) is True


# TODO(xrsrke): test only calculating the gradients of the weight, bias, or input based
# on the requires_grad of the input, weight, or bias


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
