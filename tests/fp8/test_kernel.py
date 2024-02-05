import pytest
import torch
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.kernel import fp8_matmul_kernel
from nanotron.fp8.tensor import FP8Tensor
from torch import nn


@pytest.mark.parametrize("hidden_size", [16, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("transpose_a", [True, False])
@pytest.mark.parametrize("transpose_b", [True, False])
@pytest.mark.parametrize("use_split_accumulator", [True, False])
def test_fp8_matmul_kernel(hidden_size, transpose_a, transpose_b, use_split_accumulator):
    input = torch.randn(hidden_size, hidden_size, device="cuda")
    linear = nn.Linear(hidden_size, hidden_size, bias=False, device="cuda")
    weight = linear.weight.detach().clone()

    input = input.T if transpose_a else input
    weight = weight.T if transpose_b else weight

    fp8_input = FP8Tensor(input.detach().clone(), DTypes.FP8E4M3)
    fp8_weight = FP8Tensor(weight.detach().clone(), DTypes.FP8E4M3)

    ref_output = torch.matmul(input, weight)

    output = fp8_matmul_kernel(
        mat_a=fp8_input,
        transpose_a=transpose_a,
        mat_b=fp8_weight,
        transpose_b=transpose_b,
        use_split_accumulator=use_split_accumulator,
    )

    assert torch.allclose(output, ref_output, 0.1, 0.1)
