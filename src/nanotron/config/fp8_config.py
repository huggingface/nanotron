from dataclasses import dataclass
from typing import List, Literal, Optional, Union

from nanotron.fp8.constants import DTypes
from nanotron.fp8.recipe import FP8Accumulate, FP8OptimRecipe, FP8SplitAccumulator, FP8TensorRecipe
from nanotron.logging import get_logger

logger = get_logger(__name__)

DEFAULT_GENERATION_SEED = 42


@dataclass
class TorchDtype:
    """Keep in torch tensor"""

    dtype: Literal["bfloat16", "float16", "float32"]


@dataclass
class FP8LayerArgs:
    module_name: str
    accum_dtype: DTypes
    input: Union[FP8TensorRecipe, TorchDtype]
    weight: Union[FP8TensorRecipe, TorchDtype]
    bias: Union[FP8TensorRecipe, TorchDtype]
    input_grad: Union[FP8TensorRecipe, TorchDtype]
    weight_grad: Union[FP8TensorRecipe, TorchDtype]
    output_grad: Union[FP8TensorRecipe, TorchDtype]
    split_accumulator: FP8SplitAccumulator
    accumulate: FP8Accumulate


@dataclass
class FP8Args:
    model: Optional[List[FP8LayerArgs]] = None
    optim: Optional[FP8OptimRecipe] = None
