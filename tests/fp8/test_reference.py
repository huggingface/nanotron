from copy import deepcopy

import numpy as np
import pytest
import torch
from msamp.common.dtype import Dtypes
from msamp.common.dtype import Dtypes as MS_Dtypes
from msamp.common.tensor import ScalingMeta, TypeCast
from msamp.nn import LinearReplacer
from msamp.optim import LBAdam
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.optim import FP8Adam
from nanotron.fp8.tensor import FP8Tensor, convert_tensor_from_fp8
from nanotron.fp8.utils import convert_linear_to_fp8
from torch import nn
from torch.optim import Adam


def test_optim():
    HIDDEN_SIZE = 16
    N_STEPS = 1
    LR = 1e-3

    ref_linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda")
    linear = deepcopy(ref_linear)
    linear = convert_linear_to_fp8(linear, accum_qtype=DTypes.KFLOAT16)
    msamp_linear = deepcopy(ref_linear)
    msamp_linear = LinearReplacer.replace(msamp_linear, MS_Dtypes.kfloat16)

    ref_optim = Adam(ref_linear.parameters(), lr=LR)
    msamp_optim = LBAdam(msamp_linear.parameters(), lr=LR)
    optim = FP8Adam(linear.parameters(), lr=LR)

    input = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda", requires_grad=False)

    for _ in range(N_STEPS):
        ref_output = ref_linear(input)
        ref_output.sum().backward()
        ref_optim.step()
        ref_optim.zero_grad()

        msamp_output = msamp_linear(input)
        msamp_output.sum().backward()
        # msamp_optim.all_reduce_grads(msamp_linear)
        msamp_optim.step()
        msamp_optim.zero_grad()

        output = linear(input)
        output.sum().backward()
        optim.step()
        optim.zero_grad()

    # NOTE: 3e-4 is from msamp
    torch.testing.assert_close(msamp_linear.weight.float(), ref_linear.weight, rtol=0, atol=3e-4)
    torch.testing.assert_close(msamp_linear.bias.float(), ref_linear.bias, rtol=0, atol=3e-4)

    # torch.testing.assert_close(linear.weight, ref_linear.weight, rtol=0.1, atol=3e-4)
    # torch.testing.assert_close(linear.bias, ref_linear.bias, rtol=0, atol=3e-4)


# NOTE: because we do O3 optimization, it only available for deepspeed
# so can't do reference test
# def test_fwd():
#     HIDDEN_SIZE = 16
#     ref_linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda")
#     msamp_linear = deepcopy(ref_linear)
#     msamp_linear = LinearReplacer.replace(msamp_linear, MS_Dtypes.kfloat16)

#     linear = convert_linear_to_fp8(deepcopy(ref_linear), accum_qtype=DTypes.KFLOAT16)

#     input = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda")

#     ref_output = ref_linear(input)
#     msamp_output = msamp_linear(input)
#     output = linear(input)

#     torch.testing.assert_close(msamp_output.float(), ref_output, rtol=0, atol=0.1)
#     # assert torch.equal(output, msamp_output)


def test_fwd_and_bwd():
    HIDDEN_SIZE = 16
    ref_linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda")
    msamp_linear = deepcopy(ref_linear)
    msamp_linear = LinearReplacer.replace(msamp_linear, MS_Dtypes.kfloat16)

    linear = convert_linear_to_fp8(deepcopy(ref_linear), accum_qtype=DTypes.KFLOAT16)

    input = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda")

    ref_output = ref_linear(input)
    msamp_output = msamp_linear(input)
    output = linear(input)

    torch.testing.assert_close(msamp_output.float(), ref_output, rtol=0, atol=0.1)

    msamp_output.sum().backward()
    ref_output.sum().backward()
    output.sum().backward()

    # weight_fp32 = convert_tensor_from_fp8(linear.weight, linear.weight.fp8_meta, torch.float32)
    # assert torch.equal(weight_fp32, msamp_linear.weight.grad.float())

    torch.testing.assert_close(msamp_linear.weight.grad.float(), ref_linear.weight.grad, rtol=0.1, atol=0.1)
    torch.testing.assert_close(msamp_linear.bias.grad, ref_linear.bias.grad, rtol=0, atol=0.1)


@pytest.mark.parametrize("size", [4, 8, 16, 64])
@pytest.mark.parametrize(
    "dtype, msamp_dtype, ", [(DTypes.FP8E4M3, Dtypes.kfloat8_e4m3), (DTypes.FP8E5M2, Dtypes.kfloat8_e5m2)]
)
def test_quantize_and_dequantize_tensor_in_fp8(size, dtype, msamp_dtype):
    tensor = torch.randn((size, size), dtype=torch.float32, device="cuda")
    ref_tensor = deepcopy(tensor)

    meta = ScalingMeta(msamp_dtype)
    msamp_fp8_tensor = TypeCast.cast_to_fp8(deepcopy(tensor), meta)
    msamp_tensor = TypeCast.cast_from_fp8(msamp_fp8_tensor, meta, Dtypes.kfloat32)

    fp8_tensor = FP8Tensor(deepcopy(tensor), dtype=dtype)

    assert torch.equal(fp8_tensor, msamp_fp8_tensor)
    assert not np.array_equal(fp8_tensor.cpu().numpy(), ref_tensor.cpu().numpy())

    tensor = convert_tensor_from_fp8(fp8_tensor, fp8_tensor.fp8_meta, torch.float32)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == ref_tensor.dtype

    # NOTE: this tolerance is from FP8-LM's implementation
    # reference: https://github.com/Azure/MS-AMP/blob/9ac98df5371f3d4174d8f103a1932b3a41a4b8a3/tests/common/tensor/test_cast.py#L23
    # NOTE: i tried to use rtol=0, atol=0.1
    # but even msamp fails to pass 6/8 tests
    # so now use 0.1, but better do a systematic tuning
    assert torch.equal(tensor, msamp_tensor)
    torch.testing.assert_close(msamp_tensor, ref_tensor, rtol=0.1, atol=0.1)
    torch.testing.assert_close(tensor, ref_tensor, rtol=0.1, atol=0.1)
