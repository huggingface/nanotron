from dataclasses import dataclass
from typing import Union

import torch
import transformer_engine as te  # noqa
import transformer_engine_extensions as tex

from nanotron.fp8.constants import DTYPE_TO_FP8_MAX
from nanotron.fp8.tensor import convert_torch_dtype_to_te_dtype


@dataclass
class FP8Meta:
    """Metadata for FP8Tensor."""

    amax: Union[int, float]
    scale: torch.Tensor

    # TODO(xrsrke): change to Literal[torch.int8, torch.uint8]
    dtype: torch.dtype

    @property
    def te_dtype(self) -> tex.DType:
        return convert_torch_dtype_to_te_dtype(self.dtype)

    def __post_init__(self):
        # NOTE: transformer engine only accepts torch tensors
        self.amax = torch.tensor(self.amax, device="cuda") if not isinstance(self.amax, torch.Tensor) else self.amax

    @property
    def fp8_max(self) -> float:
        """Return the maximum normal value for the current dtype."""
        return DTYPE_TO_FP8_MAX[self.dtype]

    @property
    def inverse_scale(self) -> torch.Tensor:
        return 1 / self.scale

    def __repr__(self) -> str:
        return f"FP8Meta(amax={self.amax}, scale={self.scale}, inverse_scale={self.inverse_scale}, dtype={self.dtype})"
