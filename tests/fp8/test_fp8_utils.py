import pytest

from nanotron.fp8.linear import FP8Linear
from nanotron.fp8.parameter import FP8Parameter
from torch import nn
from utils import convert_linear_to_fp8
from nanotron.fp8.dtypes import DTypes


@pytest.mark.parametrize("out_dtype", [DTypes.KFLOAT32, DTypes.KFLOAT16])
def test_convert_linear_to_fp8(out_dtype):
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(linear, out_dtype)

    assert isinstance(fp8_linear, FP8Linear)
    assert isinstance(fp8_linear.weight, FP8Parameter)
    assert isinstance(fp8_linear.bias, nn.Parameter)
