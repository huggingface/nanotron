from dataclasses import dataclass
from typing import Union

import torch

from nanotron.fp8.dtypes import DTypes


@dataclass
class FP8TensorRecipe:
    dtype: Union[DTypes, torch.dtype]
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
    master_weight_dtype: Union[DTypes, torch.dtype]
    exp_avg_dtype: Union[DTypes, torch.dtype]
    exp_avg_sq_dtype: Union[DTypes, torch.dtype]


@dataclass
class FP8TrainingRecipe:
    linear: FP8LinearRecipe
    optim: FP8OptimRecipe
