from __future__ import annotations

from abc import abstractstaticmethod
from copy import deepcopy
from typing import Union, cast

import torch
import transformer_engine as te  # noqa
import transformer_engine_extensions as tex

from nanotron.fp8.constants import DTYPE_TO_FP8_MAX, FP8_DTYPES, INITIAL_SCALING_FACTOR
from nanotron.fp8.dtypes import DTypes


class LowPrecisionTensor(torch.Tensor):
    def __new__(cls, tensor: torch.Tensor, dtype: DTypes, interval: int = 1) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor), "tensor must be a tensor"
        assert tensor.dtype not in FP8_DTYPES, "The tensor already quantized to FP8"

        # TODO(xrsrke): if the tensor is on cpu, then bypass the quantization
        # because the current kernels only support gpu tensor
        assert tensor.device != torch.device("cpu"), "FP8Tensor only supports CUDA device"
        assert dtype in [DTypes.FP8E4M3, DTypes.FP8E5M2, DTypes.KFLOAT16]

        fp8_meta = cls._get_metadata(tensor, dtype, interval)
        fp8_tensor = cls._quantize(tensor, fp8_meta)

        # TODO(xrsrke): move update inverse scaling to FP8Meta's initialization
        obj = torch.Tensor._make_subclass(cls, fp8_tensor)
        # TODO(xrsrke): use a different name, because FP16Tensor also has fp8_meta
        obj.fp8_meta = fp8_meta
        return obj

    # @property
    # def data(self) -> torch.Tensor:
    #     # return self.__dict__['data']
    #     return super().data

    # @data.setter
    # def data(self, data: Union[torch.Tensor, FP8Tensor]):
    #     assert isinstance(data, FP8Tensor)
    #     assert data.dtype == self.dtype, "The data must have the same dtype as the tensor"
    #     self.__dict__['data'] = data.data
    #     self.fp8_meta = data.fp8_meta

    def __setattr__(self, __name: str, __value: torch.Any) -> None:
        if __name == "data":
            assert 1 == 1

        return super().__setattr__(__name, __value)

    @staticmethod
    def _get_metadata(tensor: torch.Tensor, dtype: DTypes, interval: int) -> "FP8Meta":
        # TODO(xrsrke): there is a circular import issue
        # between tensor.py and meta.py fix this
        from nanotron.fp8.meta import FP8Meta

        amax = tensor.abs().max().clone()
        scale = update_scaling_factor(amax, torch.tensor(INITIAL_SCALING_FACTOR, dtype=torch.float32), dtype)
        fp8_meta = FP8Meta(amax, scale, dtype, interval)
        return fp8_meta

    @abstractstaticmethod
    def _quantize(tensor: torch.Tensor, fp8_meta: "FP8Meta") -> torch.Tensor:
        raise NotImplementedError

    def mul_(self, other: torch.Tensor):
        from nanotron.fp8.meta import FP8Meta

        assert isinstance(other, torch.Tensor)
        assert (
            other.ndim == 0 or other.ndim == 1
        ), "FP8Tensor don't support directly do matrix multiplication in FP8. You should cast it to a higher precision format."

        other = other.squeeze() if other.ndim == 1 else other
        self.fp8_meta = cast(FP8Meta, self.fp8_meta)
        self.fp8_meta.scale = 1 / (self.fp8_meta.inverse_scale * other)

    def div_(self, other: torch.Tensor):
        assert isinstance(other, torch.Tensor)
        assert (
            other.ndim == 0 or other.ndim == 1
        ), "FP8Tensor don't support directly do matrix division in FP8. You should cast it to a higher precision format."
        self.mul_(1 / other)

    def __add__(self, other: torch.Tensor):
        raise ValueError(
            "You can't directly add a FP8Tensor with another tensor. You should cast it to a higher precision format"
        )

    def __sub__(self, other: torch.Tensor):
        raise ValueError(
            "You can't directly subtract a FP8Tensor with another tensor. You should cast it to a higher precision format"
        )

    def __eq__(self, other: LowPrecisionTensor) -> bool:
        assert isinstance(
            other, self.__class__
        ), "Expected other tensor to be an instance of {self.__class__}, got {other.__class__}"
        return True if self.fp8_meta == other.fp8_meta and torch.equal(self.data, other.data) else False

    # TODO(xrsrke): directly set a tensor data using tensor.data = new_data
    def set_data(self, data: Union[torch.Tensor, LowPrecisionTensor]):
        assert isinstance(data, (self.__class__, torch.Tensor)), f"data must be a torch.Tensor or a {self.__class__}"
        if data.__class__ in [FP8Tensor, FP16Tensor]:
            assert data.dtype == self.data.dtype, "The data must have the same dtype as the tensor, got {data.dtype}"
            quantized_data = data
        else:
            quantized_data = self.__class__(data, self.fp8_meta.dtype, self.fp8_meta.interval)

        self.data = quantized_data.data

        # NOTE: for delay scaling
        new_amax = quantized_data.fp8_meta.amax
        self.fp8_meta.add_amax(new_amax)

        max_amax = torch.max(torch.stack(self.fp8_meta.amaxs))
        self.fp8_meta.scale = update_scaling_factor(
            max_amax, torch.tensor(INITIAL_SCALING_FACTOR, dtype=torch.float32), self.fp8_meta.dtype
        )

    def __repr__(self) -> str:
        if hasattr(self, "fp8_meta"):
            return f"FP8Tensor({repr(self.data)}, fp8_meta={self.fp8_meta})"
        return super().__repr__()

    def clone(self) -> FP8Tensor:
        tensor = super().clone()
        tensor.fp8_meta = deepcopy(self.fp8_meta)
        return tensor


class FP8Tensor(LowPrecisionTensor):
    """FP8 Tensor."""

    @staticmethod
    def _quantize(tensor: torch.Tensor, fp8_meta: "FP8Meta") -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype not in FP8_DTYPES, "The tensor already quantized to FP8"

        return convert_tensor_to_fp8(tensor, fp8_meta)


class FP16Tensor(LowPrecisionTensor):

    # TODO(xrsrke): remove specifying the dtype KFLOAT16
    # in initialization
    # TODO(xrsrke): change the name to lp_meta = low_precision_meta
    @staticmethod
    def _quantize(tensor: torch.Tensor, fp8_meta: "FP8Meta") -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype != torch.float16, "You can't quantize a tensor to FP16 if it's already FP16"

        return (tensor * fp8_meta.scale).to(torch.float16)


def convert_torch_dtype_to_te_dtype(dtype: torch.dtype) -> tex.DType:
    # NOTE: transformer engine maintains it own dtype mapping
    # so we need to manually map torch dtypes to TE dtypes
    TORCH_DTYPE_TE_DTYPE_NAME_MAPPING = {
        torch.int32: "kInt32",
        torch.float32: "kFloat32",
        torch.float16: "kFloat16",
        torch.bfloat16: "kBFloat16",
        # torch.fp8e5m2: "kFloat8E5M2",
        # torch.fp8e4m3: "kFloat8E4M3",
        # torch.int8: "kFloat8E5M2",
        # torch.uint8: "kFloat8E4M3",
        DTypes.FP8E4M3: "kFloat8E4M3",
        DTypes.FP8E5M2: "kFloat8E5M2",
        DTypes.KFLOAT16: "kFloat16",
    }
    return getattr(tex.DType, TORCH_DTYPE_TE_DTYPE_NAME_MAPPING[dtype])


# TODO(xrsrke): add type hint for meta after fixing
# circular import between tensor.py and meta.py
def convert_tensor_to_fp8(tensor: torch.Tensor, meta) -> FP8Tensor:
    te_dtype = convert_torch_dtype_to_te_dtype(meta.dtype)
    # TODO(xrsrke): after casting to fp8, update the scaling factor
    # TODO(xrsrke): it's weird that TE only take inverse_scale equal to 1
    inverse_scale = torch.tensor(1.0, device=tensor.device, dtype=torch.float32)
    return tex.cast_to_fp8(tensor, meta.scale, meta.amax, inverse_scale, te_dtype)


def convert_tensor_from_fp8(tensor: torch.Tensor, meta, dtype: torch.dtype) -> torch.Tensor:
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(dtype, torch.dtype)
    tensor_dtype = convert_torch_dtype_to_te_dtype(meta.dtype)
    output_dtype = convert_torch_dtype_to_te_dtype(dtype)

    return tex.cast_from_fp8(tensor, meta.inverse_scale, tensor_dtype, output_dtype)


def convert_tensor_from_fp16(tensor: FP16Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert isinstance(dtype, torch.dtype)
    # TODO(xrsrke): this is a hacky way to turn a fp16 tensor to a non-quantize tensor
    inverse_scale = tensor.fp8_meta.inverse_scale
    tensor = tensor.clone()
    tensor = (tensor * inverse_scale).to(dtype)
    return torch.tensor(tensor, dtype=dtype).squeeze(dim=0)


def _convert_tensor_from_fp16(tensor: FP16Tensor, fp8_meta, dtype: torch.dtype) -> torch.Tensor:
    assert isinstance(dtype, torch.dtype)

    inverse_scale = fp8_meta.inverse_scale
    tensor = tensor.clone()
    tensor = (tensor * inverse_scale).to(dtype)
    return torch.tensor(tensor, dtype=dtype).squeeze(dim=0)


def update_scaling_factor(
    amax: torch.Tensor, scaling_factor: torch.Tensor, dtype: DTypes, margin: float = 0
) -> torch.Tensor:
    """
    Update the scaling factor to quantize a tensor to FP8.
    Credits: https://github.com/Azure/MS-AMP/blob/d562f0f0bcfc9b712fa0726b73428753ff1300ab/msamp/common/tensor/meta.py#L39
    """
    # TODO(xrsrke): sometimes we store some params in fp16
    # make this configurable
    assert amax.dtype in [torch.float32, torch.float16]
    # TODO(xrsrke): can we use lower precision for scaling_factor?
    assert scaling_factor.dtype == torch.float32

    # NOTE: Since fp8_max is a fixed number based on two FP8 data types,
    # we prefer not to take fp8_max in the input arguments.
    fp8_max = torch.tensor(DTYPE_TO_FP8_MAX[dtype], dtype=torch.float32)

    # NOTE: torch.jit only take a concrete value rather than a DTYPE_TO_FP8_MAX[dtype],
    # so we create an inner function to bypass that
    @torch.jit.script
    def _inner(amax: torch.Tensor, fp8_max: torch.Tensor, scaling_factor: torch.Tensor, margin: float):
        # NOTE: calculate the number of bits to shift the exponent
        ratio = fp8_max / amax
        exp = torch.floor(torch.log2(ratio)) - margin
        new_scaling_factor = torch.round(torch.pow(2, torch.abs(exp)))
        new_scaling_factor = torch.where(amax > 0.0, new_scaling_factor, scaling_factor)
        new_scaling_factor = torch.where(torch.isfinite(amax), new_scaling_factor, scaling_factor)
        new_scaling_factor = torch.where(exp < 0, 1 / new_scaling_factor, new_scaling_factor)
        return new_scaling_factor

    return _inner(amax, fp8_max, scaling_factor, margin)
