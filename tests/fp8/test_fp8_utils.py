import pytest
import torch
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.linear import FP8Linear
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.utils import convert_linear_to_fp8, is_overflow_underflow_nan
from torch import nn


@pytest.mark.parametrize("out_dtype", [DTypes.KFLOAT32, DTypes.KFLOAT16])
def test_convert_linear_to_fp8(out_dtype):
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(linear, out_dtype)

    assert isinstance(fp8_linear, FP8Linear)
    assert isinstance(fp8_linear.weight, FP8Parameter)
    assert isinstance(fp8_linear.bias, nn.Parameter)


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
