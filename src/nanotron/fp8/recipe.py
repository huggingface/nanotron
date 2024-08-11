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
    accum_dtype: torch.dtype

    input: Union[FP8TensorRecipe, torch.dtype]
    weight: Union[FP8TensorRecipe, torch.dtype]
    # TODO(xrsrke): remove bias recipe, because we don't quantize bias
    bias: Union[FP8TensorRecipe, torch.dtype]
    # gemm_accum_dtype: DTypes

    # NOTE: for the gradients
    input_grad: Union[FP8TensorRecipe, torch.dtype]
    weight_grad: Union[FP8TensorRecipe, torch.dtype]
    # TODO(xrsrke): we don't need this, because the output gradients of a layer
    # is the input gradients of the other layer
    output_grad: Union[FP8TensorRecipe, torch.dtype]

    # TODO(xrsrke): this is a low-level implementation details
    # we should hide this from high-level apis later on
    split_accumulator: FP8SplitAccumulator
    accumulate: FP8Accumulate
    actsmooth: bool = False


@dataclass
class FP8OptimRecipe:
    # NOTE: these are just storage dtypes
    # not FP8Tensor that need to dynamically change
    # during training
    master_weight_dtype: Union[DTypes, torch.dtype]
    accum_dtype: torch.dtype

    exp_avg_dtype: Union[DTypes, torch.dtype]
    exp_avg_sq_dtype: Union[DTypes, torch.dtype]


@dataclass
class FP8TrainingRecipe:
    # TODO(xrsrke): add initialization dtype as a part of the recipe
    # currently we use float32 for initialization, then quantize it

    # TODO(xrsrke): allow disable FP8 for some specific layers like lm_head, mlp, etc.
    # maybe specify fp8 in the modeling code!

    # NOTE: precision dtype for non-fp8 modules
    linear: FP8LinearRecipe
    optim: FP8OptimRecipe
