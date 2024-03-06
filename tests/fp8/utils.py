from copy import deepcopy

import torch
import torch.nn as nn
from nanotron.fp8.constants import FP8LM_RECIPE, QTYPE_TO_DTYPE
from nanotron.fp8.linear import FP8Linear
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.dtypes import DTypes


def convert_linear_to_fp8(linear: nn.Linear, out_dtype: DTypes) -> FP8Linear:
    in_features = linear.in_features
    out_features = linear.out_features
    is_bias = linear.bias is not None

    fp8_linear = FP8Linear(in_features, out_features, bias=is_bias, device=linear.weight.device, out_dtype=out_dtype)
    fp8_linear.weight = FP8Parameter(linear.weight.detach().clone(), FP8LM_RECIPE.linear.weight.dtype)

    if is_bias:        
        # fp8_linear.bias.data = FP16Tensor(linear.bias.detach().clone(), FP8LM_RECIPE.linear.bias.dtype)
        fp8_linear.bias.orig_data = deepcopy(linear.bias.data)
        fp8_linear.bias.data = deepcopy(linear.bias.data).to(QTYPE_TO_DTYPE[out_dtype])

    return fp8_linear


def convert_to_fp8_module(module: nn.Module, out_dtype: DTypes) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            fp8_linear = convert_linear_to_fp8(child, out_dtype)
            setattr(module, name, fp8_linear)

    return module
