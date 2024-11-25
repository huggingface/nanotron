from copy import deepcopy
from typing import cast

import numpy as np
import pytest
import torch
from nanotron.fp8.constants import (
    FP8_ATOL_THRESHOLD,
    FP8_RTOL_THRESHOLD,
    FP16_ATOL_THRESHOLD,
    FP16_RTOL_THRESHOLD,
    QTYPE_TO_DTYPE,
)
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.tensor import FP8Tensor, FP16Tensor, convert_tensor_from_fp8, convert_tensor_from_fp16
from nanotron.testing.utils import TestContext


@pytest.mark.parametrize(
    "tensor_cls, dtype", [(FP8Tensor, DTypes.FP8E4M3), (FP8Tensor, DTypes.FP8E5M2), (FP16Tensor, DTypes.KFLOAT16)]
)
@pytest.mark.parametrize("interval", [1, 5])
def test_fp8_and_fp16_metadata(tensor_cls, dtype, interval):
    import transformer_engine as te  # noqa
    import transformer_engine_torch as tex

    tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")
    ref_tensor = deepcopy(tensor)

    fp8_tensor = tensor_cls(tensor, dtype=dtype, interval=interval)
    fp8_meta = cast(FP8Meta, fp8_tensor.fp8_meta)

    # TODO(xrsrke): remove the fixed 1 factor
    # it couples with the current implementation of FP8Meta
    # because we initialize scale with 1
    # assert fp8_meta.amax == ref_tensor.abs().max()
    assert torch.equal(fp8_meta.amax, ref_tensor.amax())
    assert isinstance(fp8_meta.inverse_scale, torch.Tensor)
    assert fp8_meta.scale != 0.1 and fp8_meta.scale != 1.0
    assert isinstance(fp8_meta.te_dtype, tex.DType)
    assert fp8_meta.interval == interval


@pytest.mark.parametrize("size", [4, 8, 16, 64])
@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
def test_quantize_and_dequantize_tensor_in_fp8(size, dtype):
    tensor = torch.randn((size, size), dtype=torch.float32, device="cuda")
    ref_tensor = deepcopy(tensor)
    fp8_tensor = FP8Tensor(tensor, dtype=dtype)

    assert not np.array_equal(fp8_tensor.cpu().numpy(), ref_tensor.cpu().numpy())

    tensor = convert_tensor_from_fp8(fp8_tensor, fp8_tensor.fp8_meta, torch.float32)
    # tensor = fp8_tensor.to(torch.float32)
    # NOTE: sometimes type(tensor) is FP8Tensor, but it still passes, so we directly check the class name
    # to make sure it's a torch.Tensor
    assert tensor.__class__ == torch.Tensor
    assert tensor.dtype == ref_tensor.dtype

    torch.testing.assert_close(tensor, ref_tensor, rtol=FP8_RTOL_THRESHOLD, atol=FP8_ATOL_THRESHOLD)


# @pytest.mark.parametrize("interval", [1, 5])
@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
def test_create_fp8_tensor_from_metadata(dtype):
    INTERVAL = 5
    TOTAL_STEPS, REMAINING_STEPS = 20, 16
    tensor = torch.randn((16, 16), dtype=torch.float32, device="cuda")
    fp8_tensor = FP8Tensor(tensor, dtype=dtype, interval=INTERVAL)

    new_values = [torch.randn((16, 16), dtype=torch.float32, device="cuda") for i in range(TOTAL_STEPS)]

    for i in range(TOTAL_STEPS):
        if TOTAL_STEPS - REMAINING_STEPS == i:
            current_tensor = fp8_tensor.orig_data
            fp8_meta = deepcopy(fp8_tensor.fp8_meta)

        fp8_tensor.data = new_values[i]

    resumed_fp8_tensor = FP8Tensor.from_metadata(current_tensor, fp8_meta)
    for i in range(TOTAL_STEPS - REMAINING_STEPS, TOTAL_STEPS):
        resumed_fp8_tensor.data = new_values[i]

    # NOTE: we expect a resume tensor to have the state trajectory of the original tensor
    assert resumed_fp8_tensor == fp8_tensor


@pytest.mark.parametrize("size", [4, 8, 16, 64])
def test_quantize_and_dequantize_tensor_in_fp16(size):
    tensor = torch.randn((size, size), dtype=torch.float32, device="cuda")
    ref_tensor = deepcopy(tensor)

    fp16_tensor = FP16Tensor(tensor, dtype=DTypes.KFLOAT16)

    assert not np.array_equal(fp16_tensor.cpu().numpy(), ref_tensor.cpu().numpy())

    tensor = convert_tensor_from_fp16(fp16_tensor, torch.float32)
    # tensor = fp16_tensor.to(torch.float32)
    # NOTE: sometimes type(tensor) is FP16Tensor, but it still passes
    assert tensor.__class__ == torch.Tensor
    assert tensor.dtype == ref_tensor.dtype

    # NOTE: this tolerance is from FP8-LM's implementation
    # reference: https://github.com/Azure/MS-AMP/blob/9ac98df5371f3d4174d8f103a1932b3a41a4b8a3/tests/common/tensor/test_cast.py#L35
    torch.testing.assert_close(tensor, ref_tensor, rtol=FP16_RTOL_THRESHOLD, atol=FP16_ATOL_THRESHOLD)


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


# @pytest.mark.parametrize(
#     "tensor_cls, dtype",
#     [
#         (FP8Tensor, DTypes.FP8E4M3),
#         (FP8Tensor, DTypes.FP8E5M2),
#         (FP16Tensor, DTypes.KFLOAT16),
#     ],
# )
# @pytest.mark.parametrize(
#     "scale",
#     [
#         torch.ones(1, device="cuda:0").squeeze() * 2,  # an random scalar
#         torch.ones(1, device="cuda:0") * 2,
#         torch.ones(4, 4, device="cuda:0") * 2,
#     ],
# )
# def test_multiple_fp8_tensor(tensor_cls, dtype, scale):
#     RTOL, ATOL = (
#         (FP8_RTOL_THRESHOLD, FP8_ATOL_THRESHOLD)
#         if tensor_cls == FP8Tensor
#         else (FP16_RTOL_THRESHOLD, FP16_ATOL_THRESHOLD)
#     )
#     tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda:0")
#     ref_tensor = tensor.detach().clone()

#     fp8_tensor = tensor_cls(deepcopy(tensor), dtype)
#     ref_fp8_tensor = fp8_tensor.clone()

#     with fail_if_expect_to_fail(expect_to_fail=scale.ndim > 1):
#         fp8_tensor.mul_(scale)

#     assert torch.equal(fp8_tensor, ref_fp8_tensor)
#     assert fp8_tensor.fp8_meta.scale != ref_fp8_tensor.fp8_meta.scale

#     if isinstance(fp8_tensor, FP8Tensor):
#         # NOTE: with the current implementation, we only scale the metadata
#         # not the tensor itself, so we expect the tensor to be the same
#         tensor = convert_tensor_from_fp8(fp8_tensor, fp8_tensor.fp8_meta, torch.float32)
#     else:
#         tensor = convert_tensor_from_fp16(fp8_tensor, torch.float32)

#     torch.testing.assert_allclose(tensor, ref_tensor * scale, rtol=RTOL, atol=ATOL)


# @pytest.mark.parametrize(
#     "tensor_cls, dtype",
#     [
#         (FP8Tensor, DTypes.FP8E4M3),
#         (FP8Tensor, DTypes.FP8E5M2),
#         (FP16Tensor, DTypes.KFLOAT16),
#     ],
# )
# @pytest.mark.parametrize(
#     "scale",
#     [
#         torch.ones(1, device="cuda:0").squeeze() * 2,  # an random scalar
#         torch.ones(1, device="cuda:0") * 2,
#         torch.ones(4, 4, device="cuda:0") * 2,
#     ],
# )
# def test_divide_fp8_tensor(tensor_cls, dtype, scale):
#     # NOTE: the reason that we use 2 as the scale is because
#     # the purpose of this test is to test whether we scale the magnitude
#     # of the tensor, so if use other values from normal distribution,
#     # some values could lead to quantization error, and for this test we don't
#     # test the quantization error
#     tensor = torch.randn((64, 64), dtype=torch.float32, device="cuda:0")
#     ref_tensor = deepcopy(tensor)

#     fp8_tensor = tensor_cls(deepcopy(tensor), dtype)
#     ref_fp8_tensor = fp8_tensor.clone()

#     with fail_if_expect_to_fail(expect_to_fail=scale.ndim > 1):
#         fp8_tensor.div_(scale)

#     assert torch.equal(fp8_tensor, ref_fp8_tensor)
#     assert fp8_tensor.fp8_meta.scale != ref_fp8_tensor.fp8_meta.scale

#     if isinstance(fp8_tensor, FP8Tensor):
#         tensor = convert_tensor_from_fp8(fp8_tensor, fp8_tensor.fp8_meta, torch.float32)
#         # NOTE: use the same tolerance as test_quantize_and_dequantize_tensor_in_fp8
#         torch.testing.assert_allclose(tensor, ref_tensor / scale, rtol=FP8_RTOL_THRESHOLD, atol=FP8_ATOL_THRESHOLD)
#     else:
#         tensor = convert_tensor_from_fp16(fp8_tensor, torch.float32)
#         torch.testing.assert_close(tensor, ref_tensor / scale, rtol=FP16_RTOL_THRESHOLD, atol=FP16_ATOL_THRESHOLD)


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
        # (FP16Tensor, DTypes.KFLOAT16),
    ],
)
def test_transpose_fp8_tensor(tensor_cls, dtype):
    tensor = torch.randn((16, 16), dtype=torch.float32, device="cuda:0")
    ref_transposed_tensor = deepcopy(tensor).transpose()
    fp8_tensor = tensor_cls(tensor, dtype)

    transposed_fp8_tensor = fp8_tensor.transpose()

    assert isinstance(transposed_fp8_tensor, FP8Tensor)

    dequant_transposed_fp8_tensor = transposed_fp8_tensor.to(torch.float32)
    torch.testing.assert_close(dequant_transposed_fp8_tensor, ref_transposed_tensor)


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


@pytest.mark.parametrize(
    "tensor_cls, dtype", [(FP8Tensor, DTypes.FP8E4M3), (FP8Tensor, DTypes.FP8E5M2), (FP16Tensor, DTypes.KFLOAT16)]
)
@pytest.mark.parametrize("is_quantized", [True, False])
def test_setting_new_data_for_fp8_and_fp16_tensor(tensor_cls, dtype, is_quantized):
    RTOL, ATOL = (
        (FP8_RTOL_THRESHOLD, FP8_ATOL_THRESHOLD)
        if tensor_cls == FP8Tensor
        else (FP16_RTOL_THRESHOLD, FP16_ATOL_THRESHOLD)
    )

    tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")
    quant_tensor = tensor_cls(tensor, dtype=dtype)

    new_data = torch.randn(quant_tensor.shape, dtype=torch.float32, device="cuda") * 2
    ref_new_data = deepcopy(new_data)
    expected_quantized_tensor = tensor_cls(ref_new_data, dtype=dtype)

    new_data = tensor_cls(new_data, dtype=dtype) if is_quantized else new_data
    quant_tensor.data = new_data
    assert dequant_tensor.data.data_ptr() == new_data.data.data_ptr()

    assert quant_tensor.data.dtype == QTYPE_TO_DTYPE[dtype]
    assert torch.equal(quant_tensor, expected_quantized_tensor)

    if is_quantized:
        # if tensor_cls == FP8Tensor:
        #     dequant_tensor = convert_tensor_from_fp8(quant_tensor, fp8_tensor.fp8_meta, torch.float32)
        # else:
        #     dequantized_tensor = convert_tensor_from_fp16(fp8_tensor, torch.float32)

        dequant_tensor = quant_tensor.to(torch.float32)
        assert torch.allclose(dequant_tensor, ref_new_data, rtol=RTOL, atol=ATOL)


# @pytest.mark.parametrize(
#     "tensor_cls, dtype", [(FP8Tensor, DTypes.FP8E4M3), (FP8Tensor, DTypes.FP8E5M2), (FP16Tensor, DTypes.KFLOAT16)]
# )
# @pytest.mark.parametrize("is_quantized", [True, False])
# def test_setting_None_data_for_fp8_and_fp16_tensor(tensor_cls, dtype, is_quantized):
#     tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")
#     fp8_tensor = tensor_cls(tensor, dtype=dtype)

#     fp8_tensor.set_data(None)

#     assert fp8_tensor is None
#     assert fp8_tensor.data is None

# @pytest.mark.parametrize(
#     "tensor_cls, dtype", [(FP8Tensor, DTypes.FP8E4M3), (FP8Tensor, DTypes.FP8E5M2), (FP16Tensor, DTypes.KFLOAT16)]
# )
# def test_quantize_overflow_fp8_and_fp16_tensor(tensor_cls, dtype):
#     tensor = torch.randn((64, 64), dtype=torch.float32, device="cuda:0")
#     tensor[0, 0] = torch.tensor(float("inf"))
#     fp8_tensor = tensor_cls(tensor, dtype)


@pytest.mark.parametrize(
    "tensor_cls, dtype", [(FP8Tensor, DTypes.FP8E4M3), (FP8Tensor, DTypes.FP8E5M2), (FP16Tensor, DTypes.KFLOAT16)]
)
def test_zero_out_data_of_fp8_and_fp16_tensor(tensor_cls, dtype):
    tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")
    quant_tensor = tensor_cls(tensor, dtype=dtype)

    quant_tensor.zero_()

    assert torch.equal(quant_tensor, torch.zeros_like(quant_tensor))

    dequant_tensor = quant_tensor.to(torch.float32)
    assert torch.equal(dequant_tensor, torch.zeros_like(tensor))


# NOTE: add testing based on tensor metadata
@pytest.mark.parametrize("is_meta_the_same", [True, False])
@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
def test_fp8_and_fp16_tensor_equality_based_on_tensor_value(is_meta_the_same, dtype):
    # TODO(xrsrke): support torch.equal for FP8Tensor
    tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")
    ref_tensor = deepcopy(tensor)

    fp8_tensor = FP8Tensor(tensor, dtype=dtype)
    ref_fp8_tensor = FP8Tensor(ref_tensor, dtype=dtype)

    if not is_meta_the_same:
        fp8_tensor.fp8_meta.scale = ref_fp8_tensor.fp8_meta.scale * 2

    assert (fp8_tensor == ref_fp8_tensor) is is_meta_the_same

    new_data = torch.randn(tensor.shape, dtype=torch.float32, device="cuda")
    ref_fp8_tensor.data = new_data

    assert not fp8_tensor == ref_fp8_tensor


# TODO(xrsrke): test it has all the methods of torch.Tensor

# TODO(xrsrke): test it has all the attributes of its input tensor


@pytest.mark.parametrize(
    "tensor_cls, dtype", [(FP8Tensor, DTypes.FP8E4M3), (FP8Tensor, DTypes.FP8E5M2), (FP16Tensor, DTypes.KFLOAT16)]
)
def test_serialize_fp8_tensor(tensor_cls, dtype):
    test_context = TestContext()
    store_folder = test_context.get_auto_remove_tmp_dir()
    tensor = torch.randn((4, 4), dtype=torch.float32, device="cuda")

    fp8_tensor = tensor_cls(tensor, dtype=dtype)

    torch.save(fp8_tensor, f"{store_folder}/fp8_tensor.pt")
    torch.load(f"{store_folder}/fp8_tensor.pt")
