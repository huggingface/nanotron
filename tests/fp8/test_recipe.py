from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.recipe import (
    FP8LinearRecipe,
    FP8OptimRecipe,
    FP8SplitAccumulator,
    FP8TensorRecipe,
    FP8TrainingRecipe,
)


def test_fp8_training_recipe():
    recipe = FP8TrainingRecipe(
        # linear=FP8LinearRecipe(
        #     accum_dtype=DTypes.KFLOAT16,
        #     input=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16, is_delayed_scaling=True),
        #     weight=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16, is_delayed_scaling=False),
        #     bias=FP8TensorRecipe(dtype=DTypes.KFLOAT16, margin=0, interval=16, is_delayed_scaling=False),
        #     input_grad=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16, is_delayed_scaling=False),
        #     weight_grad=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16, is_delayed_scaling=True),
        #     output_grad=FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=1, is_delayed_scaling=False),
        #     split_accumulator=FP8SplitAccumulator(output=False, input_grad=True, weight_grad=True),
        # ),
        linear=FP8LinearRecipe(
            accum_dtype=DTypes.KFLOAT16,
            input=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16),
            weight=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16),
            bias=FP8TensorRecipe(dtype=DTypes.KFLOAT16, margin=0, interval=16),
            input_grad=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16),
            weight_grad=FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=16),
            output_grad=FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=1),
            split_accumulator=FP8SplitAccumulator(output=False, input_grad=True, weight_grad=True),
            accumulate=FP8SplitAccumulator(output=True, input_grad=True, weight_grad=True),
        ),
        optim=FP8OptimRecipe(
            accum_dtype=DTypes.KFLOAT32,
            master_weight_dtype=DTypes.KFLOAT16,
            exp_avg_dtype=DTypes.FP8E4M3,
            exp_avg_sq_dtype=DTypes.KFLOAT16,
        ),
    )

    assert recipe.linear.accum_dtype == DTypes.KFLOAT16

    assert recipe.linear.input.dtype == DTypes.FP8E4M3
    assert recipe.linear.input.margin == 0
    assert recipe.linear.input.interval == 16
    # assert recipe.linear.input.is_delayed_scaling is True

    assert recipe.linear.weight.dtype == DTypes.FP8E4M3
    assert recipe.linear.weight.margin == 0
    assert recipe.linear.weight.interval == 16
    # assert recipe.linear.weight.is_delayed_scaling is False

    assert recipe.linear.bias.dtype == DTypes.KFLOAT16
    assert recipe.linear.bias.margin == 0
    assert recipe.linear.bias.interval == 16
    # assert recipe.linear.bias.is_delayed_scaling is False

    assert recipe.linear.input_grad.dtype == DTypes.FP8E4M3
    assert recipe.linear.input_grad.margin == 0
    assert recipe.linear.input_grad.interval == 16
    # assert recipe.linear.input_grad.is_delayed_scaling is False

    assert recipe.linear.weight_grad.dtype == DTypes.FP8E4M3
    assert recipe.linear.weight_grad.margin == 0
    assert recipe.linear.weight_grad.interval == 16
    # assert recipe.linear.weight_grad.is_delayed_scaling is True

    assert recipe.linear.output_grad.dtype == DTypes.FP8E5M2
    assert recipe.linear.output_grad.margin == 0
    assert recipe.linear.output_grad.interval == 1
    # assert recipe.linear.output_grad.is_delayed_scaling is True

    assert recipe.linear.split_accumulator.output is False
    assert recipe.linear.split_accumulator.input_grad is True
    assert recipe.linear.split_accumulator.weight_grad is True

    assert recipe.optim.accum_dtype == DTypes.KFLOAT32
    assert recipe.optim.master_weight_dtype == DTypes.KFLOAT16
    assert recipe.optim.exp_avg_dtype == DTypes.FP8E4M3
    assert recipe.optim.exp_avg_sq_dtype == DTypes.KFLOAT16
