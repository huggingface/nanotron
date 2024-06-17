from copy import deepcopy

import pytest
import torch
from nanotron.fp8.constants import QTYPE_TO_DTYPE
from nanotron.fp8.dtypes import DTypes

# from nanotron.fp8.kernel import correct_fp8_matmul_kernel
from nanotron.fp8.tensor import FP8Tensor


@pytest.mark.skip
@pytest.mark.parametrize(
    "input, weight, transpose_a, transpose_b",
    [
        (
            torch.randn(16, 32, device="cuda", dtype=torch.float32),
            torch.randn(16, 32, device="cuda", dtype=torch.float32),
            False,
            True,
        ),
        (
            torch.randn(16, 32, device="cuda", dtype=torch.float32),
            torch.randn(16, 32, device="cuda", dtype=torch.float32),
            True,
            False,
        ),
        (
            torch.randn(32, 16, device="cuda", dtype=torch.float32),
            torch.randn(64, 32, device="cuda", dtype=torch.float32),
            True,
            True,
        ),
        (
            torch.randn(32, 16, device="cuda", dtype=torch.float32),
            torch.randn(16, 64, device="cuda", dtype=torch.float32),
            False,
            False,
        ),
    ],
)
@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
@pytest.mark.parametrize("accum_dtype", [DTypes.KFLOAT32])
@pytest.mark.parametrize("use_split_accumulator", [True, False])
def test_fp8_matmul_kernel(input, weight, transpose_a, transpose_b, dtype, accum_dtype, use_split_accumulator):

    ref_weight = weight

    fp8_input = deepcopy(input)
    fp8_weight = deepcopy(ref_weight)
    fp8_input = FP8Tensor(fp8_input, dtype)
    fp8_weight = FP8Tensor(fp8_weight, dtype)

    try:
        ref_output = torch.matmul(input.t() if transpose_a else input, ref_weight.t() if transpose_b else ref_weight)
    except RuntimeError:
        # NOTE: tell pytest to skip this test if the input is not compatible with torch.matmul
        pytest.skip("This input is not compatible with torch.matmul")

    fp8_output = torch.zeros_like(ref_output)
    fp8_output = correct_fp8_matmul_kernel(
        mat_a=fp8_input,
        transpose_a=transpose_a,
        mat_b=fp8_weight,
        transpose_b=transpose_b,
        output=fp8_output,
        use_split_accumulator=use_split_accumulator,
        accum_qtype=accum_dtype,
    )

    assert isinstance(fp8_output, torch.Tensor)
    assert fp8_output.dtype == QTYPE_TO_DTYPE[accum_dtype]

    # # NOTE: this threshold is from fp8-lm, the paper shows that this is fine
    torch.testing.assert_allclose(fp8_output, ref_output, rtol=0, atol=0.1)
