from copy import deepcopy

import torch
from msamp.common.dtype import Dtypes as MS_Dtypes
from msamp.nn import LinearReplacer
from msamp.optim import LBAdam
from torch import nn
# from torch.optim import Adam

from nanotron.fp8.optim import FP8Adam, Adam
# from utils import convert_linear_to_fp8

import torch.nn as nn
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.linear import FP8Linear
from nanotron.fp8.parameter import FP8Parameter


def convert_linear_to_fp8(linear: nn.Linear) -> FP8Linear:
    in_features = linear.in_features
    out_features = linear.out_features
    is_bias = linear.bias is not None

    fp8_linear = FP8Linear(in_features, out_features, bias=is_bias, device=linear.weight.device)
    fp8_linear.weight = FP8Parameter(linear.weight.detach().clone(), DTypes.FP8E4M3)

    if is_bias:
        fp8_linear.bias = FP8Parameter(linear.bias.detach().clone(), DTypes.FP8E4M3)

    return fp8_linear

if __name__ == "__main__":
    HIDDEN_SIZE = 16
    N_STEPS = 1
    LR = 1e-3

    ref_linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda")
    linear = deepcopy(ref_linear)
    linear = convert_linear_to_fp8(linear)
    msamp_linear = deepcopy(ref_linear)
    msamp_linear = LinearReplacer.replace(msamp_linear, MS_Dtypes.kfloat16)

    ref_optim = Adam(ref_linear.parameters(), lr=LR)
    msamp_optim = LBAdam(msamp_linear.parameters(), lr=LR)
    optim = FP8Adam(linear.parameters(), lr=LR)

    input = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda", requires_grad=False)

    for _ in range(N_STEPS):
        print(f"##### Running reference ##### \n \n")
        ref_output = ref_linear(input)
        ref_output.sum().backward()
        ref_optim.step()
        ref_optim.zero_grad()
        
        print(f"##### Running MSAMP ##### \n \n")
        msamp_output = msamp_linear(input)
        msamp_output.sum().backward()
        # msamp_optim.all_reduce_grads(msamp_linear)
        msamp_optim.step()
        msamp_optim.zero_grad()
        
        print(f"##### Running FP8 ##### \n \n")

        output = linear(input)
        output.sum().backward()
        optim.step()
        optim.zero_grad()


    # NOTE: 3e-4 is from msamp
    torch.testing.assert_close(msamp_linear.weight.float(), ref_linear.weight, rtol=0, atol=3e-4)
    torch.testing.assert_close(msamp_linear.bias.float(), ref_linear.bias, rtol=0, atol=3e-4)
