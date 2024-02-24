from copy import deepcopy

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
        # fp8_linear.bias = FP8Parameter(linear.bias.detach().clone(), DTypes.FP8E4M3)
        fp8_linear.bias = deepcopy(linear.bias)

    return fp8_linear


def convert_to_fp8_module(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            fp8_linear = convert_linear_to_fp8(child)
            setattr(module, name, fp8_linear)

    return module
