from dataclasses import dataclass

import torch

from nanotron.fp8.dtypes import DTypes


@dataclass
class FP8TensorRecipe:
    dtype: DTypes
    margin: int
    interval: int


@dataclass
class FP8LinearRecipe:
    input_grad: FP8TensorRecipe
    weight_grad: FP8TensorRecipe
    output_grad: FP8TensorRecipe


@dataclass
class FP8OptimRecipe:
    master_weight_dtype: DTypes
    exp_avg_dtype: DTypes
    exp_avg_sq_dtype: DTypes


@dataclass
class FP8TrainingRecipe:
    linear: FP8LinearRecipe
    optim: FP8OptimRecipe


default_recipe = FP8TrainingRecipe(
    linear=FP8LinearRecipe(
        input_grad=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16),
        weight_grad=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16),
        output_grad=FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=1),
    ),
    optim=FP8OptimRecipe(
        master_weight_dtype=torch.float16,
        exp_avg_dtype=DTypes.FP8E4M3,
        exp_avg_sq_dtype=torch.float16,
    ),
)
