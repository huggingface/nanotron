from copy import deepcopy

import numpy as np
import pytest
import torch
import transformer_engine as te  # noqa
import transformer_engine_extensions as tex
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.tensor import FP8Tensor, convert_tensor_from_fp8


@pytest.mark.parametrize("size", [4, 8, 16, 64])
@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
def test_quantize_and_dequantize_tensor_in_fp8(size, dtype):
    tensor = torch.randn((size, size), dtype=torch.float32, device="cuda")
    ref_tensor = deepcopy(tensor)

    fp8_tensor = FP8Tensor(tensor, dtype=dtype)

    assert isinstance(fp8_tensor, FP8Tensor)
    assert isinstance(fp8_tensor.fp8_meta, FP8Meta)
    assert fp8_tensor.device == ref_tensor.device
    assert fp8_tensor.dtype == torch.uint8
    assert fp8_tensor.shape == ref_tensor.shape
    assert fp8_tensor.numel() == ref_tensor.numel()
    assert not np.array_equal(fp8_tensor.cpu().numpy(), ref_tensor.cpu().numpy())

    # TODO(xrsrke): remove the fixed 1 factor
    # it couples with the current implementation of FP8Meta
    # because we initialize scale with 1
    assert fp8_tensor.fp8_meta.amax == ref_tensor.abs().max()
    assert isinstance(fp8_tensor.fp8_meta.inverse_scale, torch.Tensor)
    assert fp8_tensor.fp8_meta.scale != 0.1 and fp8_tensor.fp8_meta.scale != 1
    assert isinstance(fp8_tensor.fp8_meta.te_dtype, tex.DType)

    tensor = convert_tensor_from_fp8(fp8_tensor, fp8_tensor.fp8_meta, torch.float32)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == ref_tensor.dtype
    assert torch.allclose(tensor, ref_tensor, rtol=1e-1, atol=1e-1)


@pytest.marak.parametrize("tensor_cls, dtype", [(FP8Tensor, DTypes.FP8E4M3), (FP16, DTypes.FP8E5M2)])
def test_fp8_tensor_repr():
    tensor = torch.randn((64, 64), dtype=torch.float32, device="cuda:0")
    fp8_tensor = FP8Tensor(tensor, DTypes.FP8E4M3)

    # NOTE: in some cases, it causes an infinite loop
    # in repr(tensor), so just check if it doesn't loop
    assert isinstance(repr(fp8_tensor), str)


def test_fp8_tensor_attrs():
    tensor = torch.randn((64, 64), dtype=torch.float32, device="cuda:0")
    ref_tensor = tensor.detach().clone()

    fp8_tensor = FP8Tensor(tensor, DTypes.FP8E4M3)

    assert isinstance(fp8_tensor, FP8Tensor)
    assert isinstance(fp8_tensor.fp8_meta, FP8Meta)
    assert fp8_tensor.device == ref_tensor.device
    assert fp8_tensor.dtype == torch.uint8
    assert fp8_tensor.shape == ref_tensor.shape
    assert fp8_tensor.numel() == ref_tensor.numel()
    assert fp8_tensor.device == ref_tensor.device


# TODO(xrsrke): test it has all the methods of torch.Tensor

# TODO(xrsrke): test it has all the attributes of its input tensor

# TODO(xrsrke): test automatic padding if a tensor shape isn't divisible by 16
