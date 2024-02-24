import torch

from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.recipe import FP8LinearRecipe, FP8OptimRecipe, FP8TensorRecipe, FP8TrainingRecipe

FP8_GPU_NAMES = ["h100", "rtx 4090"]

INITIAL_AMAX = 1.0
INITIAL_SCALING_FACTOR = 1.0

FP8_DTYPES = [torch.int8, torch.uint8]
FP8E4M3_DTYPE = torch.int8
FP8E5M2_DTYPE = torch.uint8

DTYPE_TO_FP8_MAX = {DTypes.FP8E4M3: 448.0, DTypes.FP8E5M2: 57344.0, DTypes.KFLOAT16: 65504.0}


# NOTE: the training recipe of the FP8-LM paper
# FP8-LM: Training FP8 Large Language Models
# https://arxiv.org/abs/2310.18313
FP8LM_RECIPE = FP8TrainingRecipe(
    linear=FP8LinearRecipe(
        input=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16),
        weight=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16),
        # NOTE: these are the dtypes for the gradients
        input_grad=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16),
        weight_grad=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16),
        output_grad=FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=1),
    ),
    optim=FP8OptimRecipe(
        master_weight_dtype=DTypes.KFLOAT16,
        exp_avg_dtype=DTypes.FP8E4M3,
        exp_avg_sq_dtype=DTypes.KFLOAT16,
    ),
)
