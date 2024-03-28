import torch

from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.recipe import (
    FP8LinearRecipe,
    FP8OptimRecipe,
    FP8SplitAccumulator,
    FP8TensorRecipe,
    FP8TrainingRecipe,
)

FP8_GPU_NAMES = ["h100", "rtx 4090"]

INITIAL_AMAX = torch.tensor(1.0, dtype=torch.float32)
INITIAL_SCALING_FACTOR = torch.tensor(1.0, dtype=torch.float32)

FP8_DTYPES = [torch.int8, torch.uint8]
FP8E4M3_DTYPE = torch.int8
FP8E5M2_DTYPE = torch.uint8

# TODO(xrsrke): rename to DTYPE_TO_FP_MAX
# TODO(xrsrke): change to QDTYPE
DTYPE_TO_FP8_MAX = {DTypes.FP8E4M3: 448.0, DTypes.FP8E5M2: 57344.0, DTypes.KFLOAT16: 65504.0}

QTYPE_TO_DTYPE = {
    # DTypes.FP8E4M3: torch.int8,
    # TODO(xrsrke): FP8E4M3 stores as uint8?
    DTypes.FP8E4M3: torch.uint8,
    DTypes.FP8E5M2: torch.uint8,
    DTypes.KFLOAT16: torch.float16,
    DTypes.KFLOAT32: torch.float32,
}

# NOTE: the training recipe of the FP8-LM paper
# FP8-LM: Training FP8 Large Language Models
# https://arxiv.org/abs/2310.18313
FP8LM_RECIPE = FP8TrainingRecipe(
    linear=FP8LinearRecipe(
        input=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16, is_dynamic_scaling=True),
        weight=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1, is_dynamic_scaling=False),
        bias=FP8TensorRecipe(dtype=DTypes.KFLOAT16, margin=0, interval=16, is_dynamic_scaling=False),
        # NOTE: these are the dtypes for the gradients
        input_grad=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16, is_dynamic_scaling=False),
        weight_grad=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1, is_dynamic_scaling=True),
        output_grad=FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=16, is_dynamic_scaling=True),
        split_accumulator=FP8SplitAccumulator(output=False, input_grad=True, weight_grad=True),
    ),
    optim=FP8OptimRecipe(
        master_weight_dtype=DTypes.KFLOAT16,
        exp_avg_dtype=DTypes.FP8E4M3,
        exp_avg_sq_dtype=DTypes.KFLOAT16,
    ),
)

### FOR DYNAMIC LOSS SCALING ###

# TODO(xrsrke): Make it more deliberate, like if people import this constant,
# they should know that it is a constant for dynamic loss scaling
# NOTE: these initial scaling factors are from deepspeed, but we are technically free to choose our own
# LS_INITIAL_SCALING_VALUE = torch.tensor(2**32, dtype=torch.float32)
# 2^15 = 32768
LS_INITIAL_SCALING_VALUE = torch.tensor(2**32, dtype=torch.float32)
LS_INITIAL_SCALING_FACTOR = torch.tensor(2.0, dtype=torch.float32)
LS_INTERVAL = 1000
