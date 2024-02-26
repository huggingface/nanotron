from copy import deepcopy

import numpy as np
import pytest
import torch
import transformer_engine as te  # noqa
import transformer_engine_extensions as tex
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.tensor import FP8Tensor, FP16Tensor, convert_tensor_from_fp8, convert_tensor_from_fp16


@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
def test_fp8_meta_of_a_fp8_tensor(dtype):
    tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")
    ref_tensor = deepcopy(tensor)

    fp8_tensor = FP8Tensor(tensor, dtype=dtype)

    # TODO(xrsrke): remove the fixed 1 factor
    # it couples with the current implementation of FP8Meta
    # because we initialize scale with 1
    assert fp8_tensor.fp8_meta.amax == ref_tensor.abs().max()
    assert isinstance(fp8_tensor.fp8_meta.inverse_scale, torch.Tensor)
    assert fp8_tensor.fp8_meta.scale != 0.1 and fp8_tensor.fp8_meta.scale != 1
    assert isinstance(fp8_tensor.fp8_meta.te_dtype, tex.DType)


@pytest.mark.parametrize("size", [4, 8, 16, 64])
@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
def test_quantize_and_dequantize_tensor_in_fp8(size, dtype):
    tensor = torch.randn((size, size), dtype=torch.float32, device="cuda")
    ref_tensor = deepcopy(tensor)

    fp8_tensor = FP8Tensor(tensor, dtype=dtype)

    assert not np.array_equal(fp8_tensor.cpu().numpy(), ref_tensor.cpu().numpy())

    tensor = convert_tensor_from_fp8(fp8_tensor, fp8_tensor.fp8_meta, torch.float32)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == ref_tensor.dtype

    # NOTE: this tolerance is from FP8-LM's implementation
    # reference: https://github.com/Azure/MS-AMP/blob/9ac98df5371f3d4174d8f103a1932b3a41a4b8a3/tests/common/tensor/test_cast.py#L23
    # NOTE: i tried to use rtol=0, atol=0.1
    # but even msamp fails to pass 6/8 tests
    # so now use 0.1, but better do a systematic tuning
    torch.testing.assert_close(tensor, ref_tensor, rtol=0.1, atol=0.1)


@pytest.mark.parametrize("size", [4, 8, 16, 64])
def test_quantize_and_dequantize_tensor_in_fp16(size):
    tensor = torch.randn((size, size), dtype=torch.float32, device="cuda")
    ref_tensor = deepcopy(tensor)

    fp16_tensor = FP16Tensor(tensor, dtype=DTypes.KFLOAT16)

    assert not np.array_equal(fp16_tensor.cpu().numpy(), ref_tensor.cpu().numpy())

    tensor = convert_tensor_from_fp16(fp16_tensor, torch.float32)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == ref_tensor.dtype

    # NOTE: this tolerance is from FP8-LM's implementation
    # reference: https://github.com/Azure/MS-AMP/blob/9ac98df5371f3d4174d8f103a1932b3a41a4b8a3/tests/common/tensor/test_cast.py#L35
    torch.testing.assert_close(tensor, ref_tensor, rtol=0, atol=1e-03)


@pytest.mark.parametrize(
    "tensor_cls, dtype", [(FP8Tensor, DTypes.FP8E4M3), (FP8Tensor, DTypes.FP8E5M2), (FP16Tensor, DTypes.KFLOAT16)]
)
def test_fp8_and_fp16_tensor_repr(tensor_cls, dtype):
    tensor = torch.randn((64, 64), dtype=torch.float32, device="cuda:0")
    fp8_tensor = tensor_cls(tensor, dtype)

    # NOTE: in some cases, it causes an infinite loop
    # in repr(tensor), so just check if it doesn't loop
    assert isinstance(repr(fp8_tensor), str)


@pytest.mark.parametrize(
    "tensor_cls, dtype, expected_dtype",
    [
        (FP8Tensor, DTypes.FP8E4M3, torch.uint8),
        (FP8Tensor, DTypes.FP8E5M2, torch.uint8),
        (FP16Tensor, DTypes.KFLOAT16, torch.float16),
    ],
)
def test_fp8_and_fp16_tensor_attrs(tensor_cls, dtype, expected_dtype):
    tensor = torch.randn((64, 64), dtype=torch.float32, device="cuda:0")
    ref_tensor = tensor.detach().clone()

    fp8_tensor = tensor_cls(tensor, dtype)

    assert isinstance(fp8_tensor, tensor_cls)
    assert isinstance(fp8_tensor.fp8_meta, FP8Meta)
    assert fp8_tensor.dtype == expected_dtype
    assert fp8_tensor.device == ref_tensor.device
    assert fp8_tensor.shape == ref_tensor.shape
    assert fp8_tensor.numel() == ref_tensor.numel()
    assert fp8_tensor.device == ref_tensor.device


# TODO(xrsrke): test it has all the methods of torch.Tensor

# TODO(xrsrke): test it has all the attributes of its input tensor

# TODO(xrsrke): test automatic padding if a tensor shape isn't divisible by 16
