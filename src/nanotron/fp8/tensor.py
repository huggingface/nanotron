import warnings

import torch

from nanotron.fp8.constants import FP8_DTYPES
from nanotron.fp8.dtypes import DTypes

try:
    import transformer_engine as te  # noqa
    import transformer_engine_extensions as tex
except ImportError:
    warnings.warn("Please install Transformer engine for FP8 training.")


class FP8Tensor(torch.Tensor):
    """FP8 Tensor."""

    def __new__(cls, tensor: torch.Tensor, dtype: DTypes) -> torch.Tensor:
        # TODO(xrsrke): there is a circular import issue
        # between tensor.py and meta.py fix this
        from nanotron.fp8.meta import FP8Meta

        # TODO(xrsrke): if the tensor is on cpu, then bypass the quantization
        # because the current kernels only support gpu tensor
        assert tensor.device != torch.device("cpu"), "FP8Tensor only supports CUDA device"
        assert isinstance(dtype, DTypes)

        # TODO(xrsrke): can we store inverse_scale in lower precision?
        inverse_scale = torch.tensor(1.0, device=tensor.device, dtype=torch.float32)
        fp8_meta = FP8Meta(tensor.abs().max().clone(), dtype, inverse_scale)

        if tensor.dtype not in FP8_DTYPES:
            fp8_tensor = convert_tensor_to_fp8(tensor, fp8_meta)
        else:
            fp8_tensor = tensor

        # TODO(xrsrke): move update inverse scaling to FP8Meta's initialization
        fp8_meta._update_inverse_scale()
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
    return tex.cast_to_fp8(tensor, meta.scale, meta.amax, meta.inverse_scale, te_dtype)


def convert_tensor_from_fp8(tensor: torch.Tensor, meta, dtype: torch.dtype) -> torch.Tensor:
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(dtype, torch.dtype)
    tensor_dtype = convert_torch_dtype_to_te_dtype(meta.dtype)
    output_dtype = convert_torch_dtype_to_te_dtype(dtype)

    return tex.cast_from_fp8(tensor, meta.inverse_scale, tensor_dtype, output_dtype)
