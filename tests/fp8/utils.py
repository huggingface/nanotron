from copy import deepcopy
from contextlib import contextmanager

import torch
import torch.nn as nn
from nanotron.fp8.constants import FP8LM_RECIPE, QTYPE_TO_DTYPE
from nanotron.fp8.linear import FP8Linear
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.dtypes import DTypes
import pytest


def convert_linear_to_fp8(linear: nn.Linear, accum_qtype: DTypes) -> FP8Linear:
    in_features = linear.in_features
    out_features = linear.out_features
    is_bias = linear.bias is not None

    fp8_linear = FP8Linear(in_features, out_features, bias=is_bias, device=linear.weight.device, accum_qtype=accum_qtype)
    fp8_linear.weight = FP8Parameter(linear.weight.data.clone(), FP8LM_RECIPE.linear.weight.dtype)

    if is_bias:        
        fp8_linear.bias.orig_data = linear.bias.data.clone()
        fp8_linear.bias.data = linear.bias.data.to(QTYPE_TO_DTYPE[accum_qtype])

    return fp8_linear


def convert_to_fp8_module(module: nn.Module, accum_qtype: DTypes) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            fp8_linear = convert_linear_to_fp8(child, accum_qtype)
            setattr(module, name, fp8_linear)

    return module


@contextmanager
def fail_if_expect_to_fail(expect_to_fail: bool):
    try:
        yield
    except AssertionError as e:
        if expect_to_fail is True:
            pytest.xfail("Failed successfully")
        else:
            raise e
