from abc import ABC, abstractmethod
from typing import List, cast

import torch

from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.tensor import FP8Tensor


class ScalingStrategy(ABC):
    """Scaling stragey for dynamic quantization."""

    def __init__(self, tensor: FP8Tensor):
        self.tensor = tensor

    @abstractmethod
    def __bool__(self):
        """If it return True, it means that we can calculate the scaling factor now."""
        raise NotImplementedError


class WarmupStrategy(ScalingStrategy):
    def __init__(self, tensor: FP8Tensor, interval: int):
        super().__init__(tensor)
        self.interval = interval
        self.amax_history: List[torch.Tensor] = []

    """A warmup phase is a phase where the rolling number of amaxs is less than the update interval."""

    def __bool__(self) -> bool:
        fp8_meta = cast(FP8Meta, self.tensor.fp8_meta)
        num_amaxs = torch.count_nonzero(fp8_meta.amax_history)
        # NOTE: if the number of amaxs is less than the interval, then it's a warmup phase
        # => not calculate the scaling factor
        return not num_amaxs < fp8_meta.interval


class IntimeStrategy(ScalingStrategy):
    def __bool__(self) -> bool:
        return True


class SkipOverflowStrategy(ScalingStrategy):
    def __bool__(self) -> bool:
        fp8_meta = cast(FP8Meta, self.tensor.fp8_meta)
        num_amaxs = torch.count_nonzero(fp8_meta.amax_history)
        return num_amaxs < fp8_meta.interval


class SkipZeroOnlyStrategy(ScalingStrategy):
    def __bool__(self) -> bool:
        pass
