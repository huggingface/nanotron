import torch
import transformer_engine as te  # noqa
import transformer_engine_extensions as tex

from nanotron.fp8.constants import DTYPE_TO_FP8_MAX, FP8_DTYPES, INITIAL_SCALING_FACTOR
from nanotron.fp8.dtypes import DTypes


class FP8Tensor(torch.Tensor):
    """FP8 Tensor."""

    def __new__(cls, tensor: torch.Tensor, dtype: DTypes) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor), "tensor must be a tensor"
        assert tensor.dtype not in FP8_DTYPES, "The tensor already quantized to FP8"
        
        # TODO(xrsrke): there is a circular import issue
        # between tensor.py and meta.py fix this
        from nanotron.fp8.meta import FP8Meta

        # TODO(xrsrke): if the tensor is on cpu, then bypass the quantization
        # because the current kernels only support gpu tensor
        assert tensor.device != torch.device("cpu"), "FP8Tensor only supports CUDA device"
        assert isinstance(dtype, DTypes)

        amax = tensor.abs().max().clone()
        scale = update_scaling_factor(amax, torch.tensor(INITIAL_SCALING_FACTOR, dtype=torch.float32), dtype)
        fp8_meta = FP8Meta(amax, scale, dtype)
        fp8_tensor = convert_tensor_to_fp8(tensor, fp8_meta)

        # TODO(xrsrke): move update inverse scaling to FP8Meta's initialization
        obj = torch.Tensor._make_subclass(cls, fp8_tensor)
        obj.fp8_meta = fp8_meta
        return obj

    def __repr__(self) -> str:
        return f"FP8Tensor({self}, fp8_meta={self.fp8_meta})"


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


def update_scaling_factor(
    amax: torch.Tensor, scaling_factor: torch.Tensor, dtype: DTypes, margin: float = 0
) -> torch.Tensor:
    """
    Update the scaling factor to quantize a tensor to FP8.
    Credits: https://github.com/Azure/MS-AMP/blob/d562f0f0bcfc9b712fa0726b73428753ff1300ab/msamp/common/tensor/meta.py#L39
    """
    assert amax.dtype == torch.float32
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
