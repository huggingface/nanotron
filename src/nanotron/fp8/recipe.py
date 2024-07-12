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
class FP8Accumulate:
    output: bool
    input_grad: bool
    weight_grad: bool


@dataclass
class FP8LinearRecipe:
    accum_dtype: DTypes

    input: FP8TensorRecipe
    weight: FP8TensorRecipe
    # TODO(xrsrke): remove bias recipe, because we don't quantize bias
    bias: FP8TensorRecipe

    # NOTE: for the gradients
    input_grad: FP8TensorRecipe
    weight_grad: FP8TensorRecipe
    # TODO(xrsrke): we don't need this, because the output gradients of a layer
    # is the input gradients of the other layer
    output_grad: FP8TensorRecipe

    # TODO(xrsrke): this is a low-level implementation details
    # we should hide this from high-level apis later on
    split_accumulator: FP8SplitAccumulator
    accumulate: FP8Accumulate


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
    # TODO(xrsrke): add initialization dtype as a part of the recipe
    # currently we use float32 for initialization, then quantize it

    # TODO(xrsrke): allow disable FP8 for some specific layers like lm_head, mlp, etc.
    # maybe specify fp8 in the modeling code!

    linear: FP8LinearRecipe
    optim: FP8OptimRecipe
