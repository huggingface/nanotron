import pytest
import torch
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.linear import FP8Linear
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.utils import convert_linear_to_fp8, convert_to_fp8_module, is_overflow_underflow_nan
from torch import nn


@pytest.mark.parametrize("accum_dtype", [DTypes.KFLOAT32, DTypes.KFLOAT16])
def test_convert_linear_to_fp8(accum_dtype):
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(linear, accum_dtype)

    assert isinstance(fp8_linear, FP8Linear)
    assert isinstance(fp8_linear.weight, FP8Parameter)
    assert isinstance(fp8_linear.bias, nn.Parameter)


@pytest.mark.parametrize("accum_dtype", [DTypes.KFLOAT32, DTypes.KFLOAT16])
def test_convert_module_to_fp8(accum_dtype):
    linear = nn.Sequential(
        nn.Linear(16, 16, device="cuda"),
        nn.ReLU(),
        nn.Linear(16, 16, device="cuda"),
        nn.ReLU(),
    )
    fp8_linear = convert_to_fp8_module(linear, accum_dtype)

    for ref_module, fp8_module in zip(linear, fp8_linear):
        if not isinstance(ref_module, nn.Linear):
            assert isinstance(fp8_module, type(ref_module))
        else:
            assert isinstance(fp8_module, FP8Linear)

            assert ref_module.weight.shape == fp8_module.weight.shape
            assert ref_module.weight.numel() == fp8_module.weight.numel()
            assert ref_module.weight.requires_grad == fp8_module.weight.requires_grad
            assert ref_module.weight.device == fp8_module.weight.device

            assert ref_module.bias.shape == fp8_module.bias.shape
            assert ref_module.bias.numel() == fp8_module.bias.numel()
            assert ref_module.bias.requires_grad == fp8_module.bias.requires_grad
            assert ref_module.bias.device == fp8_module.bias.device


@pytest.mark.parametrize(
    "tensor, expected_output",
    [
        [torch.tensor(0), False],
        [torch.tensor(1.0), False],
        [torch.tensor(float("inf")), True],
        [torch.tensor(float("-inf")), True],
        [torch.tensor(float("nan")), True],
    ],
)
def test_detect_overflow_underflow_nan(tensor, expected_output):
    output = is_overflow_underflow_nan(tensor)
    assert output == expected_output
