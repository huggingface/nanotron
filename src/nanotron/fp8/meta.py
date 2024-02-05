import warnings
from dataclasses import dataclass
from typing import Union

import torch

from nanotron.fp8.constants import DTYPE_TO_FP8_MAX
from nanotron.fp8.tensor import convert_torch_dtype_to_te_dtype

try:
    import transformer_engine as te  # noqa
    import transformer_engine_extensions as tex
except ImportError:
    warnings.warn("Please install Transformer engine for FP8 training!")


@dataclass
class FP8Meta:
    """Metadata for FP8Tensor."""

    amax: Union[int, float]

    # TODO(xrsrke): change to Literal[torch.int8, torch.uint8]
    dtype: torch.dtype

    @property
    def te_dtype(self) -> tex.DType:
        return convert_torch_dtype_to_te_dtype(self.dtype)

    def __post_init__(self):
        # NOTE: transformer engine only accepts torch tensors
        # NOTE: remove fill scale factor when initializing
        # TODO(xrsrke): check if we haven't computed the scaling factor, then do an initial scaling factor
        # don't set the initial to 1
        self.scale = 1  # initial scaling factor
        self.scale = self._compute_scaling_factor()
        self.amax = torch.tensor(self.amax, device="cuda") if not isinstance(self.amax, torch.Tensor) else self.amax

    @property
    def fp8_max(self) -> float:
        """Return the maximum normal value for the current dtype."""
        return DTYPE_TO_FP8_MAX[self.dtype]

    # @torch.jit.script
    def _compute_scaling_factor(self, margin: float = 0) -> torch.Tensor:
        # Credits: https://github.com/Azure/MS-AMP/blob/d562f0f0bcfc9b712fa0726b73428753ff1300ab/msamp/common/tensor/meta.py#L39
        # NOTE: seems like python 3.11.5 don't call __post_init__ when using dataclass
        # so it don't cast [int, float] to torch.Tensor
        # so we need to manually cast it
        amax = torch.tensor(self.amax) if not isinstance(self.amax, torch.Tensor) else self.amax
        scale = torch.tensor(self.scale) if not isinstance(self.scale, torch.Tensor) else self.scale

        # NOTE: calculate the number of bits to shift the exponent
        ratio = self.fp8_max / amax
        exp = torch.floor(torch.log2(ratio)) - margin
        sf = torch.round(torch.pow(2, torch.abs(exp)))
        sf = torch.where(amax > 0.0, sf, scale)
        sf = torch.where(torch.isfinite(amax), sf, scale)
        sf = torch.where(exp < 0, 1 / sf, sf)
        return sf

    @property
    def inverse_scale(self) -> torch.Tensor:
        return 1 / self.scale

    def __repr__(self) -> str:
        return f"FP8Meta(amax={self.amax}, scale={self.scale}, inverse_scale={self.inverse_scale}, dtype={self.dtype})"
