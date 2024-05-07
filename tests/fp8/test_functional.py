import nanotron.fp8.functional as F
import pytest
import torch
from nanotron.fp8.constants import QTYPE_TO_DTYPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.linear import FP8Linear


@pytest.mark.parametrize("accum_qtype", [DTypes.KFLOAT32, DTypes.KFLOAT16])
def test_fp8_mm(accum_qtype):
    input = torch.randn(16, 16).to("cuda")
    linear = FP8Linear(16, 16, bias=False, device="cuda", accum_qtype=accum_qtype)
    ref_output = linear(input)

    output = torch.zeros_like(ref_output, device="cuda", dtype=QTYPE_TO_DTYPE[accum_qtype])
    output = F.mm(
        input=input,
        mat2=linear.weight.data.transpose_fp8(),
        out=output,
        accum_qtype=accum_qtype,
        metadatas=linear.metadatas,
    )

    assert ref_output.shape == output.shape
    assert torch.equal(ref_output, output)


# @pytest.mark.parametrize("beta, alpha", [(1., 1.), (1.5, 1.5), (0., 0.)])
@pytest.mark.parametrize("accum_qtype", [DTypes.KFLOAT32, DTypes.KFLOAT16])
def test_fp8_addmm(accum_qtype):
    input = torch.randn(16, 16).to("cuda")
    linear = FP8Linear(16, 16, bias=True, device="cuda", accum_qtype=accum_qtype)
    ref_output = linear(input)

    output = torch.zeros_like(ref_output, device="cuda", dtype=QTYPE_TO_DTYPE[accum_qtype])
    output = F.addmm(
        input=linear.bias,
        mat1=input,
        mat2=linear.weight.data.transpose_fp8(),
        output=output,
        beta=1.0,
        alpha=1.0,
        accum_qtype=accum_qtype,
        metadatas=linear.metadatas,
    )

    assert ref_output.shape == output.shape
    assert torch.equal(ref_output, output)


@pytest.mark.parametrize("accum_qtype", [DTypes.KFLOAT32, DTypes.KFLOAT16])
@pytest.mark.parametrize("bias", [True, False])
def test_fp8_linear(bias, accum_qtype):
    input = torch.randn(16, 16).to("cuda")
    linear = FP8Linear(16, 16, bias=bias, device="cuda", accum_qtype=accum_qtype)

    output = F.linear(
        input=input, weight=linear.weight.data, bias=linear.bias, accum_qtype=accum_qtype, metadatas=linear.metadatas
    )
    ref_output = linear(input)

    assert ref_output.shape == output.shape
    assert torch.equal(ref_output, output)
