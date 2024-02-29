from copy import deepcopy

import torch
from nanotron.fp8.constants import FP8LM_RECIPE
from nanotron.fp8.dtypes import DTypes

# from utils import convert_linear_to_fp8, convert_to_fp8_module
from nanotron.fp8.linear import FP8Linear
from nanotron.fp8.optim import Adam as REFAdam
from nanotron.fp8.optim import (
    FP8Adam,
)
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.tensor import FP8Tensor, FP16Tensor, convert_tensor_from_fp8, convert_tensor_from_fp16
from torch import nn


def convert_linear_to_fp8(linear: nn.Linear) -> FP8Linear:
    in_features = linear.in_features
    out_features = linear.out_features
    is_bias = linear.bias is not None

    fp8_linear = FP8Linear(in_features, out_features, bias=is_bias, device=linear.weight.device)
    fp8_linear.weight = FP8Parameter(linear.weight.detach().clone(), FP8LM_RECIPE.linear.weight.dtype)

    if is_bias:
        # fp8_linear.bias.data = FP16Tensor(linear.bias.detach().clone(), FP8LM_RECIPE.linear.bias.dtype)
        fp8_linear.bias.orig_data = deepcopy(linear.bias.data)
        fp8_linear.bias.data = deepcopy(linear.bias.data).to(torch.float16)

    return fp8_linear


if __name__ == "__main__":
    LR = 1e-3
    BETAS = (0.9, 0.999)
    EPS = 1e-8
    WEIGHT_DECAY = 1e-3

    input = torch.randn(16, 16, device="cuda")
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(deepcopy(linear))

    optim = REFAdam(linear.parameters(), LR, BETAS, EPS, WEIGHT_DECAY)
    fp8_optim = FP8Adam(fp8_linear.parameters(), LR, BETAS, EPS, WEIGHT_DECAY)

    # for _ in range(1):
    #     linear(input).sum().backward()
    #     optim.step()
    #     # optim.zero_grad()

    #     fp8_linear(input).sum().backward()
    #     fp8_optim.step()
    #     # fp8_optim.zero_grad()

    linear(input).sum().backward()

    fp8_linear.weight.grad = FP8Tensor(deepcopy(linear.weight.grad), dtype=FP8LM_RECIPE.linear.weight_grad.dtype)
    # # fp8_linear.bias.grad = deepcopy(linear.bias.grad).to(torch.float16)
    fp8_linear.bias.grad = FP16Tensor(deepcopy(linear.bias.grad), dtype=DTypes.KFLOAT16)

    print("------------------[REF optim]---------------------- \n \n")
    optim.step()

    print("------------------[FP8 optim]---------------------- \n \n")
    fp8_optim.step()

    # NOTE: since optimizer update depends on the gradients
    # and in this test we only want to check whether fp8 optim step is correct
    # so we will set the gradients to the target one, and only check the optim step

    print("------------------[Evaluation]---------------------- \n \n")

    print(f"[FP8] fp8_linear.weight.data: {fp8_linear.weight.data[:2, :2]} \n")
    print(f"[FP8] fp8_linear.weight.data.fp8_meta: {fp8_linear.weight.data.fp8_meta} \n")

    weight_fp32 = convert_tensor_from_fp8(fp8_linear.weight.data, fp8_linear.weight.data.fp8_meta, torch.float32)
    bias_fp32 = convert_tensor_from_fp16(fp8_linear.bias, torch.float32)
    # NOTE: this specific threshold is based on the FP8-LM implementation
    # the paper shows that it don't hurt convergence
    # reference: https://github.com/Azure/MS-AMP/blob/51f34acdb4a8cf06e0c58185de05198a955ba3db/tests/optim/test_adamw.py#L85

    print(f"[FP8] fp8_linear.bias: {fp8_linear.bias} \n")
    print(f"[FP8] bias_fp32: {bias_fp32} \n")
    print(f"[FP8] weight_fp32.weight: {weight_fp32[:2, :2]} \n")
    print(f"[FP8] fp8_optim.fp32_p.data[:]: {fp8_optim.fp32_p.data[:2, :2]} \n")

    print(f"[FP32] linear.weight: {linear.weight[:2, :2]} \n")

    torch.testing.assert_close(bias_fp32, linear.bias, rtol=0, atol=3e-4)

    torch.testing.assert_close(fp8_optim.fp32_p.data[:], linear.weight[:], rtol=0, atol=3e-4)
    torch.testing.assert_close(weight_fp32, linear.weight, rtol=0.1, atol=0.1)
    torch.testing.assert_close(weight_fp32, linear.weight, rtol=0, atol=3e-4)
