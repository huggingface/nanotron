import torch
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.recipe import FP8LinearRecipe, FP8OptimRecipe, FP8TensorRecipe, FP8TrainingRecipe


def test_fp8_training_recipe():
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

    assert default_recipe.linear.input_grad.dtype == DTypes.FP8E4M3
    assert default_recipe.linear.input_grad.margin == 0
    assert default_recipe.linear.input_grad.interval == 16

    assert default_recipe.linear.weight_grad.dtype == DTypes.FP8E4M3
    assert default_recipe.linear.weight_grad.margin == 0
    assert default_recipe.linear.weight_grad.interval == 16

    assert default_recipe.linear.output_grad.dtype == DTypes.FP8E5M2
    assert default_recipe.linear.output_grad.margin == 0
    assert default_recipe.linear.output_grad.interval == 1

    assert default_recipe.optim.master_weight_dtype == torch.float16
    assert default_recipe.optim.exp_avg_dtype == DTypes.FP8E4M3
    assert default_recipe.optim.exp_avg_sq_dtype == torch.float16
