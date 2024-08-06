from dataclasses import dataclass
from typing import List, Optional

import torch

from nanotron.fp8.recipe import FP8LinearRecipe, FP8OptimRecipe
from nanotron.logging import get_logger

logger = get_logger(__name__)

DEFAULT_GENERATION_SEED = 42


# class TorchDtype(Enum):
#     BFLOAT16 = "BFLOAT16"
#     FLOAT16 = "FLOAT16"
#     FLOAT32 = "FLOAT32"


# @dataclass
# class TorchDtypeArg:
#     """Keep in torch tensor"""

#     dtype: Literal["bfloat16", "float16", "float32"]


# @dataclass
# class FP8LayerArgs:
#     module_name: str
#     accum_dtype: torch.dtype
#     input: Union[FP8TensorRecipe, torch.dtype]
#     weight: Union[FP8TensorRecipe, torch.dtype]
#     bias: Union[FP8TensorRecipe, torch.dtype]
#     input_grad: Union[FP8TensorRecipe, torch.dtype]
#     weight_grad: Union[FP8TensorRecipe, torch.dtype]
#     output_grad: Union[FP8TensorRecipe, torch.dtype]
#     split_accumulator: FP8SplitAccumulator
#     accumulate: FP8Accumulate
@dataclass
class FP8LayerArgs(FP8LinearRecipe):
    module_name: str


@dataclass
class FP8Args:
    # NOTE: this is the datatype for residual stream (aka: non-fp8 operation)
    resid_dtype: torch.dtype
    # NOTE: the datatype for fp8 operation's accumulation
    accum_dtype: torch.dtype

    model: Optional[List[FP8LayerArgs]] = None
    optim: Optional[FP8OptimRecipe] = None

    clipped_softmax: bool = False
    clipped_softmax_zeta: Optional[float] = None
    clipped_softmax_gamma: Optional[float] = None

    gated_attention: bool = False
    layer_scale: bool = False
