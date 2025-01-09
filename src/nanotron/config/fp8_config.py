from dataclasses import dataclass
from typing import List, Optional

import torch

from nanotron.fp8.constants import FP8LM_OPTIM_RECIPE
from nanotron.fp8.recipe import FP8LinearRecipe, FP8OptimRecipe


@dataclass
class FP8LayerArgs(FP8LinearRecipe):
    module_name: Optional[str] = None

    def __post_init__(self):
        assert self.module_name is not None, "module_name must be specified"


@dataclass
class FP8Args:
    # NOTE: this is the datatype of model initialization, before casting to fp8
    init_dtype: torch.dtype = torch.float32
    # NOTE: this is the datatype for residual stream (aka: non-fp8 operation)
    resid_dtype: torch.dtype = torch.float32
    # NOTE: the datatype for fp8 operation's accumulation
    accum_dtype: torch.dtype = torch.bfloat16

    model: Optional[List[FP8LayerArgs]] = None
    optim: Optional[FP8OptimRecipe] = FP8LM_OPTIM_RECIPE

    run_fp8_sanity_check: bool = False

    update_clipping: bool = False
    skip_param_update_if_nan: bool = False

    sync_amax_in_input: bool = False
    sync_amax_in_weight: bool = False
    sync_amax_in_igrad: bool = False
    sync_amax_in_wgrad: bool = False
    sync_amax_func: str = "default"
    weight_decay_without_lr_decay: bool = False

    triton_rms_norm: bool = False

    is_sanity_logging: bool = False
    is_post_scaling_all_reduce: bool = True
    # NOTE: 1.0e-6 was the default
    gradient_clipping_eps: float = 1.0e-6

    is_quant_all_except_first_and_last: Optional[bool] = None
    fp8_linear_config_temp: Optional[FP8LayerArgs] = None
