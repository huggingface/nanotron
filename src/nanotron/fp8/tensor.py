import torch

# from brrr.utils import convert_tensor_to_fp8
import transformer_engine as te  # noqa
import transformer_engine_extensions as tex

from nanotron.fp8.constants import DTypes
from nanotron.fp8.meta import FP8Meta


class FP8Tensor(torch.Tensor):
    """FP8 Tensor."""

    # TODO(xrsrke): add type hints for fp8_meta
    def __new__(cls, tensor: torch.Tensor, dtype: DTypes):
        # if isinstance(tensor, FP8Tensor):
        #     return tensor

        # TODO(xrsrke): if the tensor is on cpu, then bypass the quantization
        assert tensor.device != torch.device("cpu"), "FP8Tensor only supports CUDA device"
        assert isinstance(dtype, DTypes)

        inverse_scale = torch.tensor(1.0, device=tensor.device, dtype=torch.float32)
        fp8_meta = FP8Meta(tensor.abs().max().clone(), dtype, inverse_scale)

        if tensor.dtype not in [torch.int8, torch.uint8]:
            fp8_tensor = convert_tensor_to_fp8(tensor, fp8_meta)
        else:
            fp8_tensor = tensor
        fp8_meta._update_inverse_scale()
        obj = torch.Tensor._make_subclass(cls, fp8_tensor)
        obj._fp8_meta = fp8_meta
        obj.fp8_meta = fp8_meta
        obj.your_mom = True
        return obj

    # @property
    # def fp8_meta(self) -> FP8Meta:
    #     return self._fp8_meta

    # @fp8_meta.setter
    # def fp8_meta(self, value: FP8Meta):
    #     self._fp8_meta = value

    # def __repr__(self) -> str:
    #     return f"FP8Tensor({self}, fp8_meta={self.fp8_meta})"


def convert_torch_dtype_to_te_dtype(dtype: torch.dtype) -> tex.DType:
    # setattr(torch, "fp8e5m2", torch.int8)
    # setattr(torch, "fp8e4m3", torch.uint8)
    from nanotron.fp8.constants import DTypes

    # NOTE: transformer engine maintains it own dtype mapping
    # so we need to manually map torch dtypes to te dtypes
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
        DTypes.kfloat16: "kFloat16",
    }
    return getattr(tex.DType, TORCH_DTYPE_TE_DTYPE_NAME_MAPPING[dtype])


def convert_tensor_to_fp8(tensor: torch.Tensor, meta: "FP8Meta") -> FP8Tensor:
    te_dtype = convert_torch_dtype_to_te_dtype(meta.dtype)
    # TODO(xrsrke): after casting to fp8, update the scaling factor
    return tex.cast_to_fp8(tensor, meta.scale, meta.amax, meta.inverse_scale, te_dtype)


def convert_tensor_from_fp8(tensor: torch.Tensor, meta, dtype: torch.dtype) -> torch.Tensor:
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(dtype, torch.dtype)
    tensor_dtype = convert_torch_dtype_to_te_dtype(meta.dtype)
    output_dtype = convert_torch_dtype_to_te_dtype(dtype)
    print(
        f"convert_tensor_from_fp8: tensor: {tensor[0, :3]} inverse_scale: {meta.inverse_scale}, tensor_dtype: {tensor_dtype}, output_dtype: {output_dtype}"
    )
    return tex.cast_from_fp8(tensor, meta.inverse_scale, tensor_dtype, output_dtype)
