from copy import deepcopy

import numpy as np
import pytest
import torch
import transformer_engine as te  # noqa
import transformer_engine_extensions as tex
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.tensor import FP8Tensor, FP16Tensor, convert_tensor_from_fp8, convert_tensor_from_fp16
from utils import fail_if_expect_to_fail


@pytest.mark.parametrize(
    "tensor_cls, dtype", [(FP8Tensor, DTypes.FP8E4M3), (FP8Tensor, DTypes.FP8E5M2), (FP16Tensor, DTypes.KFLOAT16)]
)
def test_fp8_and_fp16_metadata(tensor_cls, dtype):
    tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")
    ref_tensor = deepcopy(tensor)

    fp8_tensor = tensor_cls(tensor, dtype=dtype)

    # TODO(xrsrke): remove the fixed 1 factor
    # it couples with the current implementation of FP8Meta
    # because we initialize scale with 1
    assert fp8_tensor.fp8_meta.amax == ref_tensor.abs().max()
    assert isinstance(fp8_tensor.fp8_meta.inverse_scale, torch.Tensor)
    assert fp8_tensor.fp8_meta.scale != 0.1 and fp8_tensor.fp8_meta.scale != 1.0
    assert isinstance(fp8_tensor.fp8_meta.te_dtype, tex.DType)


@pytest.mark.parametrize("size", [4, 8, 16, 64])
@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
def test_quantize_and_dequantize_tensor_in_fp8(size, dtype):
    tensor = torch.randn((size, size), dtype=torch.float32, device="cuda")
    ref_tensor = deepcopy(tensor)

    fp8_tensor = FP8Tensor(tensor, dtype=dtype)

    assert not np.array_equal(fp8_tensor.cpu().numpy(), ref_tensor.cpu().numpy())

    tensor = convert_tensor_from_fp8(fp8_tensor, fp8_tensor.fp8_meta, torch.float32)
    # NOTE: sometimes type(tensor) is FP8Tensor, but it still passes, so we directly check the class name
    # to make sure it's a torch.Tensor
    assert tensor.__class__.__name__ == torch.Tensor.__name__
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
    # NOTE: sometimes type(tensor) is FP16Tensor, but it still passes
    assert tensor.__class__.__name__ == torch.Tensor.__name__
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


@pytest.mark.parametrize(
    "tensor_cls, dtype",
    [
        (FP8Tensor, DTypes.FP8E4M3),
        (FP8Tensor, DTypes.FP8E5M2),
        (FP16Tensor, DTypes.KFLOAT16),
    ],
)
@pytest.mark.parametrize(
    "scale",
    [
        torch.ones(1, device="cuda:0").squeeze() * 2,  # an random scalar
        torch.ones(1, device="cuda:0") * 2,
        torch.ones(4, 4, device="cuda:0") * 2,
    ],
)
def test_multiple_fp8_tensor(tensor_cls, dtype, scale):
    tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda:0")
    ref_tensor = tensor.detach().clone()

    fp8_tensor = tensor_cls(deepcopy(tensor), dtype)
    ref_fp8_tensor = fp8_tensor.clone()

    with fail_if_expect_to_fail(expect_to_fail=scale.ndim > 1):
        fp8_tensor.mul_(scale)

    assert torch.equal(fp8_tensor, ref_fp8_tensor)
    assert fp8_tensor.fp8_meta.scale != ref_fp8_tensor.fp8_meta.scale

    if isinstance(fp8_tensor, FP8Tensor):
        # NOTE: with the current implementation, we only scale the metadata
        # not the tensor itself, so we expect the tensor to be the same
        tensor = convert_tensor_from_fp8(fp8_tensor, fp8_tensor.fp8_meta, torch.float32)
        # NOTE: use the same tolerance as test_quantize_and_dequantize_tensor_in_fp8
        torch.testing.assert_allclose(tensor, ref_tensor * scale, rtol=0.1, atol=0.1)
    else:
        tensor = convert_tensor_from_fp16(fp8_tensor, torch.float32)
        torch.testing.assert_close(tensor, ref_tensor * scale, rtol=0, atol=1e-03)


@pytest.mark.parametrize(
    "tensor_cls, dtype",
    [
        (FP8Tensor, DTypes.FP8E4M3),
        (FP8Tensor, DTypes.FP8E5M2),
        (FP16Tensor, DTypes.KFLOAT16),
    ],
)
@pytest.mark.parametrize(
    "scale",
    [
        torch.ones(1, device="cuda:0").squeeze() * 2,  # an random scalar
        torch.ones(1, device="cuda:0") * 2,
        torch.ones(4, 4, device="cuda:0") * 2,
    ],
)
def test_divide_fp8_tensor(tensor_cls, dtype, scale):
    # NOTE: the reason that we use 2 as the scale is because
    # the purpose of this test is to test whether we scale the magnitude
    # of the tensor, so if use other values from normal distribution,
    # some values could lead to quantization error, and for this test we don't
    # test the quantization error
    tensor = torch.randn((64, 64), dtype=torch.float32, device="cuda:0")
    ref_tensor = tensor.detach().clone()

    fp8_tensor = tensor_cls(deepcopy(tensor), dtype)
    ref_fp8_tensor = fp8_tensor.clone()

    with fail_if_expect_to_fail(expect_to_fail=scale.ndim > 1):
        fp8_tensor.div_(scale)

    assert torch.equal(fp8_tensor, ref_fp8_tensor)
    assert fp8_tensor.fp8_meta.scale != ref_fp8_tensor.fp8_meta.scale

    if isinstance(fp8_tensor, FP8Tensor):
        tensor = convert_tensor_from_fp8(fp8_tensor, fp8_tensor.fp8_meta, torch.float32)
        # NOTE: use the same tolerance as test_quantize_and_dequantize_tensor_in_fp8
        torch.testing.assert_allclose(tensor, ref_tensor / scale, rtol=0.1, atol=0.1)
    else:
        tensor = convert_tensor_from_fp16(fp8_tensor, torch.float32)
        torch.testing.assert_close(tensor, ref_tensor / scale, rtol=0, atol=1e-03)


@pytest.mark.parametrize(
    "tensor_cls, dtype",
    [
        (FP8Tensor, DTypes.FP8E4M3),
        (FP8Tensor, DTypes.FP8E5M2),
        (FP16Tensor, DTypes.KFLOAT16),
    ],
)
def test_add_fp8_tensor(tensor_cls, dtype):
    tensor = torch.randn((64, 64), dtype=torch.float32, device="cuda:0")
    fp8_tensor = tensor_cls(deepcopy(tensor), dtype)

    with pytest.raises(ValueError):
        fp8_tensor + 1


@pytest.mark.parametrize(
    "tensor_cls, dtype",
    [
        (FP8Tensor, DTypes.FP8E4M3),
        (FP8Tensor, DTypes.FP8E5M2),
        (FP16Tensor, DTypes.KFLOAT16),
    ],
)
def test_subtract_fp8_tensor(tensor_cls, dtype):
    tensor = torch.randn((64, 64), dtype=torch.float32, device="cuda:0")
    fp8_tensor = tensor_cls(deepcopy(tensor), dtype)

    with pytest.raises(ValueError):
        fp8_tensor - 1


@pytest.mark.parametrize(
    "tensor_cls, dtype",
    [
        (FP8Tensor, DTypes.FP8E4M3),
        (FP8Tensor, DTypes.FP8E5M2),
        (FP16Tensor, DTypes.KFLOAT16),
    ],
)
def test_clone_fp8_tensor(tensor_cls, dtype):
    tensor = torch.randn((64, 64), dtype=torch.float32, device="cuda:0")
    fp8_tensor = tensor_cls(deepcopy(tensor), dtype)

    cloned_fp8_tensor = fp8_tensor.clone()

    assert isinstance(cloned_fp8_tensor, tensor_cls)
    assert id(cloned_fp8_tensor) != id(fp8_tensor)
    assert cloned_fp8_tensor.device == fp8_tensor.device

    assert torch.equal(cloned_fp8_tensor, fp8_tensor)
    assert cloned_fp8_tensor.data_ptr() != fp8_tensor.data_ptr()
    assert cloned_fp8_tensor.data.data_ptr() != fp8_tensor.data.data_ptr()

    assert cloned_fp8_tensor.fp8_meta == fp8_tensor.fp8_meta
    assert id(cloned_fp8_tensor.fp8_meta) != id(fp8_tensor.fp8_meta)


@pytest.mark.parametrize(
    "tensor_cls, dtype",
    [
        (FP8Tensor, DTypes.FP8E4M3),
        (FP8Tensor, DTypes.FP8E5M2),
        (FP16Tensor, DTypes.KFLOAT16),
    ],
)
def test_determistic_quantization(tensor_cls, dtype):
    tensor = torch.randn((64, 64), dtype=torch.float32, device="cuda:0")
    fp8_tensor = tensor_cls(deepcopy(tensor), dtype)
    ref_fp8_tensor = tensor_cls(deepcopy(tensor), dtype)

    assert torch.equal(fp8_tensor, ref_fp8_tensor)
    assert fp8_tensor.fp8_meta == ref_fp8_tensor.fp8_meta


@pytest.mark.parametrize(
    "tensor_cls, dtype", [(FP8Tensor, DTypes.FP8E4M3), (FP8Tensor, DTypes.FP8E5M2), (FP16Tensor, DTypes.KFLOAT16)]
)
def test_fp8_and_fp16_tensor_storage_memory(tensor_cls, dtype):
    tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")
    ref_tensor = deepcopy(tensor)

    fp8_tensor = tensor_cls(tensor, dtype=dtype)

    assert id(fp8_tensor) != id(ref_tensor)

    assert isinstance(fp8_tensor.data, torch.Tensor)
    assert id(fp8_tensor.data) != id(ref_tensor.data)
    assert fp8_tensor.data_ptr() == fp8_tensor.data.data_ptr()
    assert fp8_tensor.data.data_ptr() != ref_tensor.data_ptr()


# @pytest.mark.parametrize(
#     "tensor_cls, dtype", [(FP8Tensor, DTypes.FP8E4M3), (FP8Tensor, DTypes.FP8E5M2), (FP16Tensor, DTypes.KFLOAT16)]
# )
# def test_set_data_for_fp8_and_fp16_tensor(tensor_cls, dtype):
#     tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")
#     fp8_tensor = tensor_cls(tensor, dtype=dtype)

#     new_data = torch.randint(low=0, high=256, size=(4, 4), dtype=torch.uint8)
#     # new_data = tensor_cls(tensor, dtype=dtype)
#     fp8_tensor.data = new_data.data

#     # assert id(fp8_tensor.data) == id(new_data)
#     # assert torch.equal(fp8_tensor.data, new_data)
#     # assert fp8_tensor.data_ptr() == new_data.data.data_ptr()

#     # assert fp8_tensor.fp8_meta is new_data.fp8_meta

#     assert id(fp8_tensor.data) == id(new_data.data)
#     assert torch.equal(fp8_tensor.data, new_data)
#     assert fp8_tensor.data.data_ptr() == new_data.data_ptr()


@pytest.mark.parametrize(
    "tensor_cls, dtype", [(FP8Tensor, DTypes.FP8E4M3), (FP8Tensor, DTypes.FP8E5M2), (FP16Tensor, DTypes.KFLOAT16)]
)
def test_set_data_for_fp8_and_fp16_tensor(tensor_cls, dtype):
    tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")
    fp8_tensor = tensor_cls(tensor, dtype=dtype)

    new_data = torch.randint(low=0, high=256, size=(4, 4), dtype=torch.uint8)
    fp8_tensor.data = new_data.data

    assert id(fp8_tensor.data) == id(new_data)
    assert torch.equal(fp8_tensor.data, new_data)
    assert fp8_tensor.data.data_ptr() == new_data.data_ptr()


# TODO(xrsrke): test it has all the methods of torch.Tensor

# TODO(xrsrke): test it has all the attributes of its input tensor

# TODO(xrsrke): test automatic padding if a tensor shape isn't divisible by 16
