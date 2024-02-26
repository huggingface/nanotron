from dataclasses import dataclass

from nanotron.fp8.dtypes import DTypes


# TODO(xrsrke): rename to LowPrecisionTensorRecipe or LPTensorRecipe
@dataclass
class FP8TensorRecipe:
    dtype: DTypes
    margin: int
    interval: int


@dataclass
class FP8LinearRecipe:
    input: FP8TensorRecipe
    weight: FP8TensorRecipe

    # NOTE: for the gradients
    input_grad: FP8TensorRecipe
    weight_grad: FP8TensorRecipe
    output_grad: FP8TensorRecipe


@dataclass
class FP8OptimRecipe:
    # NOTE: these are just storage dtypes
    # not FP8Tensor that need to dynamically change
    # during training
    master_weight_dtype: DTypes
    exp_avg_dtype: DTypes
    exp_avg_sq_dtype: DTypes


@dataclass
class FP8TrainingRecipe:
    linear: FP8LinearRecipe
    optim: FP8OptimRecipe
