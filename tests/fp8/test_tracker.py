from typing import List

import pytest
import torch
from torch import nn

from nanotron.fp8 import tracker
from nanotron.fp8 import FP8Tensor, DTypes
from nanotron.fp8 import FP8Linear
from utils import convert_to_fp8_module


class MetaRecorder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: FP8Tensor, scaling_factors: List, amaxs: List) -> torch.Tensor:
        ctx.fp8_meta = tensor.fp8_meta
        ctx.scaling_factors = scaling_factors
        ctx.amaxs = amaxs
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        ctx.scaling_factors.append(ctx.fp8_meta.scale)
        ctx.amaxs.append(ctx.fp8_meta.amax)
        return grad_output
    
def record_meta(module: FP8Linear, scaling_factors: List, amaxs: List) -> FP8Linear:
    module.register_forward_hook(MetaRecorder.apply)
    return module
    

@pytest.mark.parametrize("interval", [1, 5, 10])
@pytest.mark.parametrize("n_expected_updates", [1, 5])
def test_scaling_tracker(interval, n_expected_updates):
    output = torch.randn((64, 64), dtype=torch.float32, device="cuda:0")

    linear = nn.Linear(64, 64, device="cuda:0")
    fp8_linear = convert_to_fp8_module(linear)
    fp8_linear = tracker.track(fp8_linear, interval=interval)
    
    scaling_factors = {"input_grad": [], "weight_grad": [], "output_grad": []}
    amaxs = {"input_grad": [], "weight_grad": [], "output_grad": []}
    
    for _ in range(interval*n_expected_updates):
        # TODO(xrsrke): how to setup this up without the edge case of overflow/underflow
        output = fp8_linear(output)
        output.sum().backward()

    assert len(scaling_factors) == n_expected_updates
    assert len(amaxs) == interval*n_expected_updates

    # TODO: assert skip the current delay scaling if we encounter overflow/underflow, or a zero only tensor


# TODO(xrsrke): decoupling the dynamic quantization from optimizer => fp8linear should work with torch optimizer
