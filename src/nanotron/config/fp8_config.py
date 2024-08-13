from dataclasses import dataclass
from typing import List, Optional

import torch

from nanotron.fp8.recipe import FP8LinearRecipe, FP8OptimRecipe


@dataclass
class FP8LayerArgs(FP8LinearRecipe):
    module_name: Optional[str] = None

    def __post_init__(self):
        assert self.module_name is not None, "module_name must be specified"


@dataclass
class FP8Args:
    # NOTE: this is the datatype for residual stream (aka: non-fp8 operation)
    resid_dtype: torch.dtype
    # NOTE: the datatype for fp8 operation's accumulation
    accum_dtype: torch.dtype

    model: Optional[List[FP8LayerArgs]] = None
    optim: Optional[FP8OptimRecipe] = None

    run_fp8_sanity_check: bool = False

    clipped_softmax: bool = False
    clipped_softmax_zeta: Optional[float] = None
    clipped_softmax_gamma: Optional[float] = None

    gated_attention: bool = False
    layer_scale: bool = False
    layer_scale_init: Optional[str] = None

    qk_norm: bool = False
    qk_norm_before_pos: bool = False

    smooth_quant: Optional[bool] = None
    stochastic_rounding: bool = False
    update_clipping: bool = False
    skip_param_update_if_nan: bool = False

    sync_amax_in_wgrad: bool = False
