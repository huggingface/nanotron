from dataclasses import dataclass
from typing import Union

import torch

from nanotron.fp8.dtypes import DTypes


# TODO(xrsrke): rename to LowPrecisionTensorRecipe or LPTensorRecipe
@dataclass
class FP8TensorRecipe:
    dtype: DTypes
    margin: int
    interval: int
    # is_delayed_scaling: bool


@dataclass
class FP8SplitAccumulator:
    output: bool
    input_grad: bool
    weight_grad: bool


@dataclass
class FP8LinearRecipe:
    accum_dtype: DTypes

    input: FP8TensorRecipe
    weight: FP8TensorRecipe
    bias: FP8TensorRecipe

    # NOTE: for the gradients
    input_grad: FP8TensorRecipe
    weight_grad: FP8TensorRecipe
    output_grad: FP8TensorRecipe

    # TODO(xrsrke): this is a low-level implementation details
    # we should hide this from high-level apis later on
    split_accumulator: FP8SplitAccumulator


@dataclass
class FP8OptimRecipe:
    # NOTE: these are just storage dtypes
    # not FP8Tensor that need to dynamically change
    # during training
    master_weight_dtype: DTypes
    accum_dtype: Union[torch.dtype, DTypes]

    exp_avg_dtype: DTypes
    exp_avg_sq_dtype: DTypes


@dataclass
class FP8TrainingRecipe:
    linear: FP8LinearRecipe
    optim: FP8OptimRecipe
