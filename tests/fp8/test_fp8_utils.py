from nanotron.fp8.linear import FP8Linear
from nanotron.fp8.parameter import FP8Parameter, FP16Parameter
from torch import nn
from utils import convert_linear_to_fp8


def test_convert_linear_to_fp8():
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(linear)

    assert isinstance(fp8_linear, FP8Linear)
    assert isinstance(fp8_linear.weight, FP8Parameter)
    assert isinstance(fp8_linear.bias, FP16Parameter)
