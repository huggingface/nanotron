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
    DTypes.FP8E4M3: torch.int8,
    DTypes.FP8E5M2: torch.uint8,
    DTypes.KFLOAT16: torch.float16,
    DTypes.KFLOAT32: torch.float32,
}

# NOTE: the training recipe of the FP8-LM paper
# FP8-LM: Training FP8 Large Language Models
# https://arxiv.org/abs/2310.18313

# FP8-LM
# weight.window_size = 1, input.window_size = 16,
# wgrad.window_size = 1, ograd.window_size = 16 (this one is the input of the backward pass),
# input_grad.window_size = 1 (this one is the output of the backward pass)

# TODO(xrsrke): differentiate the precision that you initializes model weight
# and the accumulation precision in FP8 recipe

# FP8LM_LINEAR_RECIPE = FP8LinearRecipe(
#     accum_dtype=DTypes.KFLOAT16,
#     input=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16),
#     weight=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
#     bias=FP8TensorRecipe(dtype=DTypes.KFLOAT16, margin=0, interval=16),
#     # NOTE: these are the dtypes for the gradients
#     input_grad=FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=16),  # NOTE: this is output_grad
#     weight_grad=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
#     output_grad=FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=16),
#     split_accumulator=FP8SplitAccumulator(output=True, input_grad=True, weight_grad=True),
#     # NOTE: tested, and it works
#     # accumulate=FP8SplitAccumulator(output=False, input_grad=False, weight_grad=False),
#     accumulate=FP8SplitAccumulator(output=True, input_grad=True, weight_grad=True),
# )
FP8LM_LINEAR_RECIPE = FP8LinearRecipe(
    accum_dtype=torch.float16,
    input=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16),
    weight=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
    bias=torch.float16,
    # NOTE: these are the dtypes for the gradients
    input_grad=FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=16),  # NOTE: this is output_grad
    weight_grad=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
    output_grad=FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=16),
    split_accumulator=FP8SplitAccumulator(output=True, input_grad=True, weight_grad=True),
    # NOTE: tested, and it works
    # accumulate=FP8SplitAccumulator(output=False, input_grad=False, weight_grad=False),
    accumulate=FP8SplitAccumulator(output=True, input_grad=True, weight_grad=True),
)

# FP8LM_OPTIM_RECIPE = FP8OptimRecipe(
#     accum_dtype=DTypes.KFLOAT32,
#     master_weight_dtype=DTypes.KFLOAT16,
#     exp_avg_dtype=DTypes.FP8E4M3,
#     exp_avg_sq_dtype=DTypes.KFLOAT16,
# )
FP8LM_OPTIM_RECIPE = FP8OptimRecipe(
    accum_dtype=torch.float32,
    master_weight_dtype=torch.float32,
    exp_avg_dtype=torch.float32,
    exp_avg_sq_dtype=torch.float32,
)

FP8LM_RECIPE = FP8TrainingRecipe(
    # linear=FP8LinearRecipe(
    #     accum_dtype=DTypes.KFLOAT16,
    #     input=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16, is_delayed_scaling=True),
    #     weight=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1, is_delayed_scaling=False),
    #     bias=FP8TensorRecipe(dtype=DTypes.KFLOAT16, margin=0, interval=16, is_delayed_scaling=False),
    #     # NOTE: these are the dtypes for the gradients
    #     input_grad=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16, is_delayed_scaling=True),
    #     weight_grad=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1, is_delayed_scaling=True),
    #     output_grad=FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=1, is_delayed_scaling=False),
    #     split_accumulator=FP8SplitAccumulator(output=False, input_grad=True, weight_grad=True),
    # ),
    # # # NOTE: FP8-LM recipe
    # linear=FP8LinearRecipe(
    #     accum_dtype=DTypes.KFLOAT16,
    #     input=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16),
    #     weight=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
    #     bias=FP8TensorRecipe(dtype=DTypes.KFLOAT16, margin=0, interval=16),
    #     # NOTE: these are the dtypes for the gradients
    #     input_grad=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16),
    #     weight_grad=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
    #     output_grad=FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=1),
    #     split_accumulator=FP8SplitAccumulator(output=True, input_grad=True, weight_grad=True),
    # ),
    # # NOTE: FP8-LM recipe
    linear=FP8LM_LINEAR_RECIPE,
    # NOTE: works for 8B
    # linear=FP8LinearRecipe(
    #     accum_dtype=DTypes.KFLOAT16,
    #     input=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
    #     weight=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
    #     bias=FP8TensorRecipe(dtype=DTypes.KFLOAT16, margin=0, interval=1),
    #     # NOTE: these are the dtypes for the gradients
    #     input_grad=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
    #     weight_grad=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
    #     output_grad=FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=1),
    #     # split_accumulator=FP8SplitAccumulator(output=False, input_grad=True, weight_grad=True), # NOTE: msamp use this
    #     split_accumulator=FP8SplitAccumulator(output=True, input_grad=True, weight_grad=True),
    # ),
    optim=FP8LM_OPTIM_RECIPE,
)

### FOR DYNAMIC LOSS SCALING ###

# TODO(xrsrke): Make it more deliberate, like if people import this constant,
# they should know that it is a constant for dynamic loss scaling
# NOTE: these initial scaling factors are from deepspeed, but we are technically free to choose our own
# LS_INITIAL_SCALING_VALUE = torch.tensor(2**32, dtype=torch.float32)
# 2^15 = 32768
# LS_INITIAL_SCALING_VALUE = torch.tensor(2**32, dtype=torch.float32)
LS_INITIAL_SCALING_VALUE = torch.tensor(32768, dtype=torch.float32)
LS_INITIAL_SCALING_FACTOR = torch.tensor(2.0, dtype=torch.float32)
LS_INTERVAL = 1000


# FOR TESTING
# NOTE: this tolerance is from FP8-LM's implementation
# reference: https://github.com/Azure/MS-AMP/blob/9ac98df5371f3d4174d8f103a1932b3a41a4b8a3/tests/common/tensor/test_cast.py#L23
# NOTE: i tried to use rtol=0, atol=0.1
# but even msamp fails to pass 6/8 tests
# so now use 0.1, but better do a systematic tuning
FP8_RTOL_THRESHOLD = 0.1
FP8_ATOL_THRESHOLD = 0.1

FP16_RTOL_THRESHOLD = 0
FP16_ATOL_THRESHOLD = 1e-03

# NOTE: FP8-LM use RTOL is 0, and ATOL is 3e-4 for model weights
FP8_WEIGHT_RTOL_THRESHOLD = 0
FP8_WEIGHT_ATOL_THRESHOLD = 0.1

FP8_1ST_OPTIM_STATE_RTOL_THRESHOLD = 0
FP8_1ST_OPTIM_STATE_ATOL_THRESHOLD = 0.1
FP8_2ND_OPTIM_STATE_RTOL_THRESHOLD = 0
FP8_2ND_OPTIM_STATE_ATOL_THRESHOLD = 0.1
