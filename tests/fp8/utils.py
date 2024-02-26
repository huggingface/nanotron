
import torch.nn as nn
from nanotron.fp8.constants import FP8LM_RECIPE
from nanotron.fp8.linear import FP8Linear
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.tensor import FP16Tensor


def convert_linear_to_fp8(linear: nn.Linear) -> FP8Linear:
    in_features = linear.in_features
    out_features = linear.out_features
    is_bias = linear.bias is not None

    fp8_linear = FP8Linear(in_features, out_features, bias=is_bias, device=linear.weight.device)
    fp8_linear.weight = FP8Parameter(linear.weight.detach().clone(), FP8LM_RECIPE.linear.weight.dtype)

    if is_bias:
        fp8_linear.bias.data = FP16Tensor(linear.bias.detach().clone(), FP8LM_RECIPE.linear.bias.dtype)

    return fp8_linear


def convert_to_fp8_module(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            fp8_linear = convert_linear_to_fp8(child)
            setattr(module, name, fp8_linear)

    return module
