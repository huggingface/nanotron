import torch
import torch.nn as nn

from nanotron.fp8.linear import FP8Linear
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.dtypes import DTypes

def convert_to_fp8_module(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            is_bias = child.bias is not None
            
            fp8_linear = FP8Linear(in_features, out_features, bias=is_bias, device=module.device)
            fp8_linear.weight = FP8Parameter(child.weight.detach().clone(), DTypes.FP8E4M3)
            
            if is_bias:
                fp8_linear.bias = FP8Parameter(child.bias.detach().clone(), DTypes.FP8E4M3)
            
            setattr(module, name, fp8_linear)

    return module
