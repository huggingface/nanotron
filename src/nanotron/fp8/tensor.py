from __future__ import annotations

from abc import abstractstaticmethod
from copy import deepcopy
from typing import Optional, Union, cast

import torch
import transformer_engine as te  # noqa
import transformer_engine_torch as tex

from nanotron import constants, logging
from nanotron.fp8.constants import DTYPE_TO_FP8_MAX, FP8_DTYPES, INITIAL_SCALING_FACTOR
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.meta import FP8Meta

logger = logging.get_logger(__name__)


# @torch.no_grad()
@torch.jit.script
def get_amax(tensor: torch.Tensor, sync: bool) -> torch.Tensor:

    # NOTE: do .clone() somehow fixes nan grad,
    # check `exp801_fp8_nan_debug` for more details
    amax = tensor.abs().max().clone()
    # amax = tensor.amax().clone()

    # if sync is True:
    #     import torch.distributed as dist

    #     from nanotron import constants

    #     if constants.CONFIG.fp8.sync_amax_func == "default":
    #         world_size = dist.get_world_size(group=constants.PARALLEL_CONTEXT.tp_pg)
    #         if world_size > 1:
    #             # log_rank(f"Local amax is {amax}", logger=logger, level=logging.INFO)
    #             dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=constants.PARALLEL_CONTEXT.tp_pg)
    #             # log_rank(f"Global amax is {amax}", logger=logger, level=logging.INFO)
    #     else:
    #         raise ValueError(f"Unknown sync_amax_func: {constants.CONFIG.fp8.sync_amax_func}")

    return amax


class LowPrecisionTensor(torch.Tensor):
    def __new__(
        cls,
        tensor: torch.Tensor,
        dtype: Optional[DTypes] = None,
        interval: Optional[int] = 1,
        fp8_meta: Optional[FP8Meta] = None,
        sync: bool = False,
    ) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor), "tensor must be a tensor"

        # TODO(xrsrke): if the tensor is on cpu, then bypass the quantization
        # because the current kernels only support gpu tensor
        assert tensor.device != torch.device("cpu"), "FP8Tensor only supports CUDA device"

        if fp8_meta is None:
            assert dtype in [DTypes.FP8E4M3, DTypes.FP8E5M2, DTypes.KFLOAT16]

            with torch.no_grad():
                fp8_meta = cls._get_metadata(tensor, dtype, interval, sync=sync)

        backup_fp8_meta = deepcopy(fp8_meta)
        if tensor.dtype not in FP8_DTYPES:
            fp8_tensor = cls._quantize(tensor, fp8_meta)
        else:
            fp8_tensor = tensor

        # TODO(xrsrke): move update inverse scaling to FP8Meta's initialization
        obj = torch.Tensor._make_subclass(cls, fp8_tensor)
        # TODO(xrsrke): use a different name, because FP16Tensor also has fp8_meta
        obj.fp8_meta = backup_fp8_meta
        if constants.ITERATION_STEP == 1:
            obj.orig_data = tensor

        return obj

    def __init__(
        self,
        tensor: torch.Tensor,
        dtype: Optional[DTypes] = None,
        interval: Optional[int] = 1,
        fp8_meta: Optional[FP8Meta] = None,
        sync: bool = False,
    ) -> None:
        pass

    @staticmethod
    # @torch.no_grad()
    def _get_metadata(tensor: torch.Tensor, dtype: DTypes, interval: int, sync: bool) -> "FP8Meta":
        # TODO(xrsrke): there is a circular import issue
        # between tensor.py and meta.py fix this
        from nanotron.fp8.meta import FP8Meta

        # NOTE: detach from original computational graph
        # amax = tensor.abs().max().clone().detach()
        # amax = get_amax(tensor, sync)
        amax = tensor.amax().clone()

        scale = update_scaling_factor(amax, torch.tensor(INITIAL_SCALING_FACTOR, dtype=torch.float32), dtype)
        scale = scale.clone().detach()
        fp8_meta = FP8Meta(amax, scale, dtype, interval)
        return fp8_meta

    @abstractstaticmethod
    def _quantize(tensor: torch.Tensor, fp8_meta: "FP8Meta") -> torch.Tensor:
        ...

    def mul_(self, other: torch.Tensor):

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

    # TODO(xrsrke): need some more work to make it work with torch.equal
    def __eq__(self, other: LowPrecisionTensor) -> bool:
        assert isinstance(
            other, self.__class__
        ), "Expected other tensor to be an instance of {self.__class__}, got {other.__class__}"
        return True if self.fp8_meta == other.fp8_meta and torch.equal(self.data, other.data) else False

    # TODO(xrsrke): directly set a tensor data using tensor.data = new_data
    def set_data(self, data: Union[torch.Tensor, LowPrecisionTensor, None], sync: bool = False):
        assert isinstance(data, (self.__class__, torch.Tensor)), f"data must be a torch.Tensor or a {self.__class__}"
        if data.__class__ in [FP8Tensor, FP16Tensor]:
            assert data.dtype == self.data.dtype, "The data must have the same dtype as the tensor, got {data.dtype}"
            quantized_data = data
        else:
            quantized_data = self.__class__(
                data, dtype=self.fp8_meta.dtype, interval=self.fp8_meta.interval, sync=sync
            )

        self.data = quantized_data.data
        self._orig_data_after_set_data = data

        if constants.ITERATION_STEP == 1:
            self.orig_data = quantized_data.orig_data

        self.fp8_meta.add_amax(quantized_data.fp8_meta.amax)

    @staticmethod
    @torch.no_grad()
    def from_metadata(data: torch.Tensor, metadata: "FP8Meta", sync: bool = False) -> Union[FP8Tensor, FP16Tensor]:
        assert isinstance(data, (FP8Tensor, torch.Tensor)), "data must be a torch.Tensor or a FP8Tensor"
        # NOTE: don't do deepcopy, because we reuse the same metadata
        # for other iterations in fp8linear
        # metadata.add_amax(data.abs().max().clone())
        amax = get_amax(data, sync)
        metadata.add_amax(amax)

        quantized_data = FP8Tensor(data, metadata.dtype, metadata.interval, fp8_meta=metadata, sync=sync)
        return quantized_data

    def transpose_fp8(self) -> FP8Tensor:
        """Transpose the tensor."""
        transposed_t = tex.fp8_transpose(self, self.fp8_meta.te_dtype)
        transposed_t.fp8_meta = self.fp8_meta
        return self.__class__(transposed_t, fp8_meta=self.fp8_meta)

    def __repr__(self) -> str:
        if hasattr(self, "fp8_meta"):
            if self.__class__ == FP16Tensor:
                return f"FP16Tensor({repr(self.data)}, fp8_meta={self.fp8_meta})"
            elif self.__class__ == FP8Tensor:
                return f"FP8Tensor({repr(self.data)}, fp8_meta={self.fp8_meta})"
            else:
                raise ValueError(f"Unknown tensor class: {self.__class__}")

        return super().__repr__()

    def clone(self) -> FP8Tensor:
        tensor = super().clone()
        tensor.fp8_meta = deepcopy(self.fp8_meta)
        return tensor

    # def __torch_function__(self, func, types, args=(), kwargs=None):
    #     return super().__torch_function__(func, types, args, kwargs)

    # @classmethod
    # def __torch_function__(cls, func, types, args, kwargs=None):
    #     kwargs = kwargs or {}
    #     if func is torch.transpose:
    #         assert type(args[0]) == cls
    #         assert type(args[1]) == type(args[2]) == int
    #         # return CustomMaskedSum.apply(*args, **kwargs)
    #         assert 1 == 1
    #     else:
    #         super().__torch_function__(func, types, args, kwargs)


class FP8Tensor(LowPrecisionTensor):
    """FP8 Tensor."""

    @staticmethod
    def _quantize(tensor: torch.Tensor, fp8_meta: "FP8Meta") -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype not in FP8_DTYPES, "The tensor already quantized to FP8"

        tensor = tensor.contiguous()
        return convert_tensor_to_fp8(tensor, fp8_meta)


class FP16Tensor(LowPrecisionTensor):

    # TODO(xrsrke): remove specifying the dtype KFLOAT16
    # in initialization
    # TODO(xrsrke): change the name to lp_meta = low_precision_meta
    @staticmethod
    def _quantize(tensor: torch.Tensor, fp8_meta: "FP8Meta") -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        # assert tensor.dtype != torch.float16, "You can't quantize a tensor to FP16 if it's already FP16"

        tensor = tensor.contiguous()
        # TODO(xrsrke): convert it to int8 format
        return (tensor * fp8_meta.scale).to(torch.float16)


def convert_torch_dtype_to_te_dtype(dtype: torch.dtype) -> tex.DType:
    # NOTE: transformer engine maintains it own dtype mapping
    # so we need to manually map torch dtypes to TE dtypes
    TORCH_DTYPE_TE_DTYPE_NAME_MAPPING = {
        torch.int32: "kInt32",
        torch.float32: "kFloat32",
        torch.float16: "kFloat16",
        torch.bfloat16: "kBFloat16",
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


# @torch.jit.script
def update_scaling_factor(
    amax: torch.Tensor, scaling_factor: torch.Tensor, dtype: DTypes, margin: float = 0
) -> torch.Tensor:
    """
    Update the scaling factor to quantize a tensor to FP8.
    Credits: https://github.com/Azure/MS-AMP/blob/d562f0f0bcfc9b712fa0726b73428753ff1300ab/msamp/common/tensor/meta.py#L39
    """
    # TODO(xrsrke): sometimes we store some params in fp16
    # make this configurable
    assert amax.dtype in [torch.float32, torch.float16, torch.bfloat16], f"amax.dtype: {amax.dtype}"
    # TODO(xrsrke): can we use lower precision for scaling_factor?
    assert scaling_factor.dtype == torch.float32

    # NOTE: Since fp8_max is a fixed number based on two FP8 data types,
    # we prefer not to take fp8_max in the input arguments.
    # NOTE: create cuda tensor slows down by 7%
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
