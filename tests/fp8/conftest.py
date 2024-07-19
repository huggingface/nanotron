# @pytest.fixture(params=[
#     {
#         "accum_dtype": accum_dtype,
#         "recipes": recipes,
#         "split_accumulator": split_accumulator,
#         "accumulate": accumulate
#     }
#     for accum_dtype in [DTypes.KFLOAT16, DTypes.KFLOAT32]
#     for recipes in [
#         [
#             FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
#             FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
#             FP8TensorRecipe(dtype=DTypes.FP8E4M3, margin=0, interval=1),
#         ],
#         [
#             FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=1),
#             FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=1),
#             FP8TensorRecipe(dtype=DTypes.FP8E5M2, margin=0, interval=1),
#         ],
#     ]
#     for split_accumulator, accumulate in [
#         [
#             FP8SplitAccumulator(output=True, input_grad=True, weight_grad=True),
#             FP8Accumulate(output=True, input_grad=True, weight_grad=True),
#         ],
#         [
#             FP8SplitAccumulator(output=False, input_grad=False, weight_grad=False),
#             FP8SplitAccumulator(output=False, input_grad=False, weight_grad=False),
#         ],
#     ]
# ])
# def linear_recipes(request):
#     return request.param


# @pytest.mark.parametrize("optim_recipe", [
#     FP8OptimRecipe(master_weight_dtype=torch.float32, accum_dtype=torch.float32, exp_avg_dtype=torch.float32, exp_avg_sq_dtype=torch.float32),
#     FP8OptimRecipe(master_weight_dtype=DTypes.KFLOAT16, accum_dtype=torch.float32, exp_avg_dtype=torch.float32, exp_avg_sq_dtype=torch.float32)
# ])
# def optim_recipe(optim_recipe):
#     return optim_recipe.params
