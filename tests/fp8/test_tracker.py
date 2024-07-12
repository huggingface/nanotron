from copy import deepcopy
from typing import List

import pytest
import torch
from nanotron.fp8 import FP8Linear, FP8Tensor, tracker
from nanotron.fp8.utils import convert_linear_to_fp8
from torch import nn


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


@pytest.mark.skip
@pytest.mark.parametrize("interval", [1, 5, 10])
@pytest.mark.parametrize("n_expected_updates", [1, 5])
def test_amax_tracker(interval, n_expected_updates):
    input = torch.randn((64, 64), dtype=torch.float32, device="cuda:0")

    linear = nn.Linear(64, 64, device="cuda:0")
    fp8_linear = convert_linear_to_fp8(deepcopy(linear))
    fp8_linear = tracker.track(fp8_linear, interval=interval)

    amaxs = {"input_grad": [], "weight_grad": [], "output_grad": []}
    for _ in range(interval * n_expected_updates):
        fp8_linear(input).sum().backward()

    # TODO(xrsrke): check that it always keep the interval recent amax
    assert len(amaxs) == interval * n_expected_updates
