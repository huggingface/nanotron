from dataclasses import dataclass
from typing import List

import torch
import transformer_engine as te  # noqa
import transformer_engine_extensions as tex

from nanotron.fp8.constants import DTYPE_TO_FP8_MAX
from nanotron.fp8.dtypes import DTypes


@dataclass
class FP8Meta:
    """Metadata for FP8Tensor."""

    amax: torch.Tensor
    scale: torch.Tensor

    # TODO(xrsrke): change to Literal[torch.int8, torch.uint8]
    dtype: DTypes
    interval: int
    # TODO(xrsrke): change to is_delay_scaling
    is_dynamic_scaling: bool = False

    @property
    def te_dtype(self) -> tex.DType:
        from nanotron.fp8.tensor import convert_torch_dtype_to_te_dtype
        return convert_torch_dtype_to_te_dtype(self.dtype)

    def __post_init__(self):
        # assert isinstance(self.scale, torch.Tensor)
        assert isinstance(self.amax, torch.Tensor)
        assert isinstance(self.dtype, DTypes)
        assert isinstance(self.interval, int)
        assert self.interval > 0, f"Expected interval to be greater than 0, got {self.interval}"
        assert (
            self.scale.dtype == torch.float32
        ), f"Expected scale to be of dtype torch.float32, got {self.scale.dtype}"
        assert self.amax.dtype in [
            torch.float32,
            torch.float16,
        ], f"Expected amax to be of dtype torch.float32 or torch.float16, got {self.amax.dtype}"

        # NOTE: transformer engine only accepts torch tensors
        self.amax = torch.tensor(self.amax, device="cuda") if not isinstance(self.amax, torch.Tensor) else self.amax
        self._amaxs: List[torch.Tensor] = [self.amax]
        self._num_remaining_steps_until_rescale: int = self.interval - 1
        # self._scale: torch.Tensor

    @property
    def fp8_max(self) -> float:
        """Return the maximum normal value for the current dtype."""
        return DTYPE_TO_FP8_MAX[self.dtype]
    
    # @property
    # def scale(self) -> torch.Tensor:
    #     return self._scale
    
    # @scale.setter
    # def scale(self, value: torch.Tensor):
    #     # if len(self._amaxs) == 0:
    #     #     self._scale = value
    #     # elif self.is_ready_to_scale is True:
    #     #     # NOTE: now compute the scaling factor based on the intervals
    #     self._scale = value


    @property
    def inverse_scale(self) -> torch.Tensor:
        # TODO(xrsrke): this is a hacky way, remove the _inverse_scale
        return 1 / self.scale

    # TODO(xrsrke): move to strategy pattern
    def add_amax(self, amax: torch.Tensor):
        from nanotron.fp8.utils import is_overflow_underflow_nan

        if len(self._amaxs) == self.interval:
            # TODO(xrsrke): do we have to clear the old amax
            # from memory?
            self._amaxs.pop(0)

        is_overflowed = is_overflow_underflow_nan(amax)

        if is_overflowed:
            # NOTE: if amax is inf or nan, we use 0 as the new amax
            amax = torch.tensor(0.0, dtype=torch.float32, device="cuda")

        self._amaxs.append(amax)

        if is_overflowed:
            self._num_remaining_steps_until_rescale = 0
        elif self.interval != 1:
            self._num_remaining_steps_until_rescale -= 1

        if self.is_ready_to_scale:
            self.rescale()

    @property
    def amaxs(self) -> List[torch.Tensor]:
        return self._amaxs

    @property
    def is_ready_to_scale(self) -> bool:
        if self.is_dynamic_scaling is False:
            # NOTE: if this is not dynamic scaling, then we scale every interval
            return True
        
        if self.is_dynamic_scaling and self._num_remaining_steps_until_rescale == 0:
            # NOTE: if this is dynamic scaling, then we only scale once we reach the interval
            return True
        
        return False

    def rescale(self):
        assert self.is_ready_to_scale is True, "Cannot rescale if not ready to scale"
        from nanotron.fp8.tensor import update_scaling_factor

        max_amax = torch.max(torch.stack(self.amaxs))
        current_scale = self.scale
        self.scale = update_scaling_factor(max_amax, current_scale, self.dtype)
        self._num_remaining_steps_until_rescale = self.interval

    def __repr__(self) -> str:
        return f"FP8Meta(amax={self.amax}, scale={self.scale}, inverse_scale={self.inverse_scale}, dtype={self.dtype})"
