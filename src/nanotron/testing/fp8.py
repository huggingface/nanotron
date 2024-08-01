import torch
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.recipe import FP8Accumulate, FP8LinearRecipe, FP8OptimRecipe, FP8SplitAccumulator, FP8TensorRecipe


def generate_linear_recipes():
    # accum_dtypes = [DTypes.KFLOAT16, DTypes.KFLOAT32]
    accum_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    fp8_recipes = [
        [
            FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
            FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
            FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
        ],
        [
            FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=1),
            FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=1),
            FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=1),
        ],
    ]

    split_accumulator_options = [
        FP8SplitAccumulator(output=True, input_grad=True, weight_grad=True),
        FP8SplitAccumulator(output=False, input_grad=False, weight_grad=False),
    ]

    accumulate_options = [
        FP8Accumulate(output=True, input_grad=True, weight_grad=True),
        FP8SplitAccumulator(output=False, input_grad=False, weight_grad=False),
    ]

    input_recipe = FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1)
    weight_recipe = FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1)
    bias_recipe = FP8TensorRecipe(dtype=DTypes.KFLOAT16, margin=0, interval=0)

    linear_recipes = []

    for accum_dtype in accum_dtypes:
        for input_grad_recipe, weight_grad_recipe, output_grad_recipe in fp8_recipes:
            for split_accumulator, accumulate in zip(split_accumulator_options, accumulate_options):
                recipe = FP8LinearRecipe(
                    accum_dtype=accum_dtype,
                    input=input_recipe,
                    weight=weight_recipe,
                    bias=bias_recipe,
                    input_grad=input_grad_recipe,
                    weight_grad=weight_grad_recipe,
                    output_grad=output_grad_recipe,
                    split_accumulator=split_accumulator,
                    accumulate=accumulate,
                )
                linear_recipes.append(recipe)

    return linear_recipes


LINEAR_RECIPES = generate_linear_recipes()
OPTIM_RECIPES = [
    FP8OptimRecipe(
        master_weight_dtype=torch.float32,
        accum_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
    ),
    FP8OptimRecipe(
        master_weight_dtype=DTypes.KFLOAT16,
        accum_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
    ),
]
