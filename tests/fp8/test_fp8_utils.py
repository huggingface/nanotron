import pytest
import torch
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.linear import FP8Linear
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.utils import (
    convert_linear_to_fp8,
    convert_to_fp8_module,
    get_leaf_modules,
    is_overflow_underflow_nan,
    track_module_statistics,
    _log,
)
from torch import nn


@pytest.mark.parametrize("accum_dtype", [DTypes.KFLOAT32, DTypes.KFLOAT16])
def test_convert_linear_to_fp8(accum_dtype):
    linear = nn.Linear(16, 16, device="cuda")
    fp8_linear = convert_linear_to_fp8(linear, accum_dtype)

    assert isinstance(fp8_linear, FP8Linear)
    assert isinstance(fp8_linear.weight, FP8Parameter)
    assert isinstance(fp8_linear.bias, nn.Parameter)


@pytest.mark.parametrize("model_type", ["multilayers", "transformer"])
@pytest.mark.parametrize("accum_dtype", [DTypes.KFLOAT32, DTypes.KFLOAT16])
def test_convert_module_to_fp8(model_type, accum_dtype):
    HIDDEN_SIZE = 16
    input = torch.randn(1, HIDDEN_SIZE, device="cuda")

    if model_type == "multilayers":
        model = nn.Sequential(
            nn.Linear(16, 16, device="cuda"),
            nn.ReLU(),
            nn.Linear(16, 16, device="cuda"),
            nn.ReLU(),
        )
    else:
        model = nn.Transformer(nhead=2, num_encoder_layers=4, d_model=HIDDEN_SIZE).to("cuda")

    fp8_model = convert_to_fp8_module(model, accum_dtype)

    assert fp8_model(input).shape == model(input).shape

    ref_modules = get_leaf_modules(model)
    fp8_modules = get_leaf_modules(fp8_model)

    for (ref_name, ref_module), (fp8_name, fp8_module) in zip(ref_modules, fp8_modules):
        assert ref_name == fp8_name

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


def test_track_module_statistics():
    class FP8Model(nn.Module):
        def __init__(self):
            super(FP8Model, self).__init__()
            self.fin = FP8Linear(32, 32, device="cuda")
            self.relu = nn.ReLU()
            self.fout = FP8Linear(32, 32, device="cuda")
        
        def forward(self, x):
            return self.fout(self.relu(self.fin(x)))
        
    input = torch.randn(32, 32, device="cuda")
    model = FP8Model()
        
    logs = _log(model)
    
    for _ in range(5):
        model(input).sum().backward()
        
    # NOTE: now merge module_name:x:statistic into a flat dictionary
    assert logs.keys() == {"fin", "relu", "fout"}
    assert logs["fin"].keys() == {"weight", "bias", "input:0", "output:0", "grad_output:0"}
