import math
from copy import deepcopy
from functools import partial, reduce

import pytest
import torch
from nanotron.fp8.constants import FP8_DTYPES, QTYPE_TO_DTYPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.linear import FP8Linear
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.tensor import FP8Tensor, convert_tensor_from_fp8
from nanotron.fp8.utils import convert_linear_to_fp8, convert_to_fp8_module, is_overflow_underflow_nan
from nanotron.fp8.loss_scaler import LossScaler
# from timm.models.layers import trunc_normal_
from torch import nn


@pytest.mark.parametrize("accum_qtype", [DTypes.KFLOAT32, DTypes.KFLOAT16])
@pytest.mark.parametrize("bias", [True, False])
def test_create_an_fp8_linear_parameters(bias, accum_qtype):
    fp8_linear = FP8Linear(64, 64, bias=bias, device="cuda", accum_qtype=accum_qtype)

    assert isinstance(fp8_linear.weight, FP8Parameter)
    assert isinstance(fp8_linear.bias, torch.Tensor) if bias else True
    assert isinstance(fp8_linear.accum_qtype, DTypes)


def test_fp8_linear_parameters():
    ref_linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(deepcopy(ref_linear), accum_qtype=DTypes.KFLOAT32)

    assert len(list(ref_linear.parameters())) == len(list(fp8_linear.parameters()))
    assert all(p is not None for p in fp8_linear.parameters())
    assert isinstance(fp8_linear.weight, FP8Parameter)
    assert isinstance(fp8_linear.bias, torch.Tensor)
    assert all(p.requires_grad for p in fp8_linear.parameters()) is True


@pytest.mark.parametrize("n_layers", [1, 2])
@pytest.mark.parametrize(
    "input",
    [
        torch.randn(64, 64, device="cuda", dtype=torch.float32),  # [B, H]
        torch.randn(16, 64, device="cuda", dtype=torch.float32),  # [B, H]
        torch.randn(16, 32, 64, device="cuda", dtype=torch.float32),  # [B, N, H]
        torch.randn(64, 64, 64, device="cuda", dtype=torch.float32),  # [B, N, H]
    ],
)
@pytest.mark.parametrize("is_bias", [True, False])
@pytest.mark.parametrize("accum_qtype", [DTypes.KFLOAT32, DTypes.KFLOAT16])
def test_fp8_linear_forward_pass(n_layers, input, is_bias, accum_qtype):
    HIDDEN_SIZE = 64
    INTERDIM_SIZE = 64 * 4

    ref_input = input.detach().clone()
    ref_linear = nn.Sequential(
        *[
            layer
            for _ in range(n_layers)
            for layer in (nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=is_bias, device="cuda"), nn.ReLU())
        ]
    )

    fp8_linear = convert_to_fp8_module(deepcopy(ref_linear), accum_qtype)

    ref_output = ref_linear(ref_input)
    output = fp8_linear(input)

    assert isinstance(output, torch.Tensor)
    assert output.dtype == QTYPE_TO_DTYPE[accum_qtype]

    # NOTE: this threshold is from fp8-lm, the paper shows that this is fine
    torch.testing.assert_allclose(output, ref_output, rtol=0, atol=0.1)


# TODO(xrsrke): add cases where the input requires and don't require grad
@pytest.mark.parametrize("n_layers", [1, 2])
@pytest.mark.parametrize(
    "input",
    [
        torch.randn(64, 64, device="cuda", dtype=torch.float32),  # [B, H]
        # torch.randn(16, 64, device="cuda", dtype=torch.float32),  # [B, H]
        # torch.randn(16, 32, 64, device="cuda", dtype=torch.float32),  # [B, N, H]
        # torch.randn(64, 64, 64, device="cuda", dtype=torch.float32),  # [B, N, H]
    ],
)
# @pytest.mark.parametrize(
#     "init_method",
#     [
#         lambda weight: trunc_normal_(weight, std=0.02),
#         lambda weight: trunc_normal_(weight, std=math.sqrt(1 / 64)),
#         lambda weight: trunc_normal_(weight, std=math.sqrt(1 / 64 * 4)),
#         lambda weight: trunc_normal_(weight, std=1),
#     ],
# )
# @pytest.mark.parametrize("is_bias", [True, False])
@pytest.mark.parametrize("with_scaler", [True, False])
@pytest.mark.parametrize("accum_qtype", [DTypes.KFLOAT32, DTypes.KFLOAT16])
def test_fp8_linear_backward_pass(n_layers, input, with_scaler, accum_qtype):
    is_bias = False
    
    HIDDEN_SIZE = 64
    INTERDIM_SIZE = 64 * 4

    ref_input = input.detach().clone().requires_grad_(True)
    # ref_linear = nn.Linear(HIDDEN_SIZE, INTERDIM_SIZE, device="cuda", dtype=torch.float32)
    ref_linear = nn.Sequential(
        *[
            layer
            for _ in range(n_layers)
            for layer in (nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=is_bias, device="cuda"), nn.ReLU())
        ]
    )
    
    loss_scaler = LossScaler()
        
    # trunc_normal_(ref_linear.weight, std=0.02)
    # trunc_normal_(ref_linear.weight, std=math.sqrt(1 / (HIDDEN_SIZE)))

    # fp8_linear = convert_linear_to_fp8(deepcopy(ref_linear), accum_qtype)
    fp8_linear = convert_to_fp8_module(deepcopy(ref_linear), accum_qtype)

    ref_linear(ref_input).sum().backward()
    
    if with_scaler is False:
        fp8_linear(input).sum().backward()
    else:
        loss_scaler.scale(fp8_linear(input).sum()).backward()
        loss_scaler.unscale_(fp8_linear.parameters())
        
    for ref_p, p in zip(ref_linear.parameters(), fp8_linear.parameters()):
        if p.requires_grad is False:
            assert p.grad is None
            continue
        
        if isinstance(p, FP8Parameter):
            assert isinstance(p.grad, FP8Tensor)
            assert p.grad.dtype in FP8_DTYPES
            grad = convert_tensor_from_fp8(p.grad, p.grad.fp8_meta, torch.float32)
        else:
            assert isinstance(p.grad, torch.Tensor)
            assert p.grad.dtype == QTYPE_TO_DTYPE[accum_qtype]
        
        assert is_overflow_underflow_nan(grad) is False
        if p.ndim > 1:
            # NOTE: these weight threshold is tuned from the FP8-LM implementation
            # TODO(xrsrke): tune what is the minimum threshold for this to correctly converge
            torch.testing.assert_allclose(grad, ref_p.grad, rtol=0.06, atol=0.1)
        else:
            torch.testing.assert_allclose(grad, ref_p.grad)

    # assert isinstance(fp8_linear.weight.grad, FP8Tensor)
    # assert fp8_linear.weight.grad.dtype in FP8_DTYPES

    # assert isinstance(fp8_linear.bias.grad, torch.Tensor)
    # assert fp8_linear.bias.grad.dtype == QTYPE_TO_DTYPE[accum_qtype]

    # # TODO(xrsrke): investigate why input.grad is so high tolerance
    # # assert torch.allclose(input.grad, ref_input.grad, 0.2, 0.2) if input_requires_grad else True

    # # NOTE: these weight threshold is tuned from the FP8-LM implementation
    # # TODO(xrsrke): tune what is the minimum threshold for this to correctly converge
    # weight_grad = convert_tensor_from_fp8(fp8_linear.weight.grad, fp8_linear.weight.grad.fp8_meta, torch.float32)
    # torch.testing.assert_allclose(weight_grad, ref_linear.weight.grad, rtol=0.06, atol=0.1)
    # torch.testing.assert_allclose(fp8_linear.bias.grad, ref_linear.bias.grad)


@pytest.mark.parametrize("accum_qtype", [DTypes.KFLOAT32, DTypes.KFLOAT16])
def test_fp8_modules_trigger_the_entire_computational_graph(accum_qtype):
    HIDDEN_SIZE = 16
    TIMELINE = []

    def backward_hook(module, grad_input, grad_output, idx):
        TIMELINE.append(f"{module.__class__.__name__}.{idx}.backward")

    class Logger(nn.Module):
        def __init__(self, idx: int, module: nn.Linear):
            super().__init__()
            module.register_backward_hook(partial(backward_hook, idx=idx))
            self.module = module
            self.idx = idx

        def forward(self, input):
            TIMELINE.append(f"{self.module.__class__.__name__}.{self.idx}.forward")
            return self.module(input)

    input = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    fp8_linear = nn.Sequential(
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda", dtype=torch.float32),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda", dtype=torch.float32),
        nn.ReLU(),
    )
    fp8_linear = convert_to_fp8_module(fp8_linear, accum_qtype)
    fp8_linear = nn.ModuleList([Logger(idx, module) for idx, module in enumerate(fp8_linear)])

    output = reduce(lambda x, module: module(x), fp8_linear, input)
    scalar = torch.randn(1, device="cuda", dtype=output.dtype)
    (output.sum() * scalar).backward()

    assert TIMELINE == [
        "FP8Linear.0.forward",
        "ReLU.1.forward",
        "FP8Linear.2.forward",
        "ReLU.3.forward",
        "ReLU.3.backward",
        "FP8Linear.2.backward",
        "ReLU.1.backward",
        "FP8Linear.0.backward",
    ]
    
    for p in fp8_linear.parameters():
        if p.requires_grad is True:
            assert is_overflow_underflow_nan(p.grad) is False


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


# TODO(xrsrke): test if FP8Linear has all the methods of a torch.nn.Linear


# TODO(xrsrke): test only calculating the gradients of the weight, bias, or input based
# on the requires_grad of the input, weight, or bias

# TODO(xrsrke): test automatic padding if a input/weight shape isn't divisible by 16

