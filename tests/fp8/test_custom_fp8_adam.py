# from nanotron import distributed as dist
import nanotron.fp8.distributed as dist
import pytest
import torch
from nanotron.fp8 import constants
from nanotron.fp8.optim import FP8Adam
from nanotron.fp8.recipe import FP8OptimRecipe
from nanotron.fp8.tensor import convert_tensor_from_fp8
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import get_data_from_param
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import (
    FP8TensorParallelColumnLinear,
)
from nanotron.sanity_checks import assert_tensor_synced_across_pg
from nanotron.testing.fp8 import OPTIM_RECIPES
from nanotron.testing.parallel import init_distributed, rerun_if_address_is_in_use
from torch import nn
from torch.optim import Adam


# TODO(xrsrke): support gradient flow to bias
@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("lr", [0.001])  # [0.1, 0.001, 0.0004]
@pytest.mark.parametrize("steps", [1, 5, 10, 20, 50])
@pytest.mark.parametrize("optim_recipe", OPTIM_RECIPES)
# @pytest.mark.parametrize("linear_recipe", LINEAR_RECIPES)
@rerun_if_address_is_in_use()
def test_fp8_adam(tp: int, dp: int, pp: int, with_bias: bool, lr: float, steps: int, optim_recipe: FP8OptimRecipe):
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_fp8_adam)(
        with_bias=with_bias, lr=lr, steps=steps, optim_recipe=optim_recipe
    )


def _test_fp8_adam(
    parallel_context: ParallelContext, with_bias: bool, lr: float, steps: int, optim_recipe: FP8OptimRecipe
):
    # NOTE: divisible by 16 for TP
    in_features = 32
    out_features_per_tp_rank = 16

    out_features = parallel_context.tp_pg.size() * out_features_per_tp_rank

    column_linear = FP8TensorParallelColumnLinear(
        in_features=in_features,
        out_features=out_features,
        pg=parallel_context.tp_pg,
        mode=TensorParallelLinearMode.ALL_REDUCE,
        device="cuda",
        async_communication=False,
        bias=with_bias,
        # recipe=linear_recipe
    )
    reference_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=with_bias, device="cuda")

    with torch.inference_mode():
        dist.all_gather(
            tensor_list=list(reference_linear.weight.split(out_features_per_tp_rank, dim=0)),
            tensor=get_data_from_param(column_linear.weight),
            group=parallel_context.tp_pg,
        )

        if with_bias is True:
            bias = get_data_from_param(column_linear.bias)
            bias = bias.to(reference_linear.bias.dtype) if bias.dtype != reference_linear.bias.dtype else bias
            dist.all_gather(
                tensor_list=list(reference_linear.bias.split(out_features_per_tp_rank, dim=0)),
                tensor=bias,
                group=parallel_context.tp_pg,
            )

    batch_size = 16
    random_input = torch.randn(batch_size, in_features, device="cuda")
    dist.all_reduce(random_input, op=dist.ReduceOp.AVG, group=parallel_context.tp_pg)
    assert_tensor_synced_across_pg(random_input, pg=parallel_context.tp_pg)

    sharded_random_input = random_input.clone().contiguous()
    random_input.requires_grad = True
    sharded_random_input.requires_grad = True

    optim = FP8Adam(
        column_linear.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0, recipe=optim_recipe
    )
    ref_optim = Adam(reference_linear.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0)

    assert sum(len(group["params"]) for group in optim.param_groups) == sum(
        len(group["params"]) for group in ref_optim.param_groups
    )

    for _ in range(steps):
        optim.zero_grad()
        ref_optim.zero_grad()

        # Test that we get the same output after forward pass
        sharded_output = column_linear(sharded_random_input)
        reference_output = reference_linear(random_input)

        # TODO @thomasw21: Tune tolerance
        # try:
        #     torch.testing.assert_close(
        #         sharded_output,
        #         # TODO(xrsrke): retrieve accumulation precision from recipe
        #         reference_output[
        #             :,
        #             dist.get_rank(parallel_context.tp_pg)
        #             * out_features_per_tp_rank : (dist.get_rank(parallel_context.tp_pg) + 1)
        #             * out_features_per_tp_rank,
        #         ].to(torch.float16),
        #         rtol=0,
        #         atol=0.1,
        #     )
        # except BaseException as e:
        #     print(f"Rank {dist.get_rank(parallel_context.tp_pg)}: FAIL.")
        #     dist.barrier()
        #     raise e

        # print(f"Rank {dist.get_rank(parallel_context.tp_pg)}: SUCCESS.")
        # dist.barrier()

        # Test that we get the same gradient after backward pass
        sharded_output.sum().backward()
        reference_output.sum().backward()
        hidden_dim_slice = slice(
            dist.get_rank(parallel_context.tp_pg) * out_features_per_tp_rank,
            (dist.get_rank(parallel_context.tp_pg) + 1) * out_features_per_tp_rank,
        )

        # torch.testing.assert_close(
        #     # convert_tensor_from_fp8(column_linear.weight.data, column_linear.weight.data.fp8_meta, torch.float16),
        #     convert_tensor_from_fp8(
        #         get_data_from_param(column_linear.weight),
        #         get_data_from_param(column_linear.weight).fp8_meta,
        #         torch.float16,
        #     ),
        #     reference_linear.weight[hidden_dim_slice].to(torch.float16),
        #     rtol=0.1,
        #     atol=0.1,
        # )

        # # TODO(xrsrke): retrieve accumulation precision from recipe
        # assert sharded_output.dtype == torch.float16
        # # NOTE(xrsrke): we expect the output is a raw torch.Tensor, not FP8Paramter, or NanotronParameter
        # # assert isinstance(sharded_output, torch.Tensor)
        # assert sharded_output.__class__ == torch.Tensor
        # assert sharded_output.requires_grad is True

        # torch.testing.assert_close(sharded_random_input.grad, random_input.grad, rtol=0.1, atol=0.1)

        # if with_bias is True:
        #     torch.testing.assert_close(
        #         column_linear.bias.grad,
        #         reference_linear.bias.grad[hidden_dim_slice],
        #     )

        # if isinstance(get_data_from_param(column_linear.weight), FP8Tensor):
        #     # grad = column_linear.weight.data._temp_grad
        #     # grad = convert_tensor_from_fp8(grad, column_linear.weight.data._temp_grad.fp8_meta, torch.float16)
        #     grad = get_grad_from_parameter(column_linear.weight)
        #     grad = convert_tensor_from_fp8(grad, grad.fp8_meta, torch.float16)
        # else:
        #     # grad = column_linear.weight.grad
        #     grad = get_grad_from_parameter(column_linear.weight)

        # torch.testing.assert_close(
        #     grad,
        #     reference_linear.weight.grad[hidden_dim_slice].to(torch.float16),
        #     # rtol=0.1, atol=0.1
        #     rtol=0.2,
        #     atol=0.2,
        # )

        optim.step()
        ref_optim.step()

        assert id(optim.param_groups[0]["params"][0]) == id(column_linear.weight)
        assert id(get_data_from_param(optim.param_groups[0]["params"][0])) == id(
            get_data_from_param(column_linear.weight)
        )
        if with_bias is True:
            assert id(optim.param_groups[0]["params"][1]) == id(column_linear.bias)
            assert id(get_data_from_param(optim.param_groups[0]["params"][1])) == id(
                get_data_from_param(column_linear.bias)
            )

        dequant_w = convert_tensor_from_fp8(
            get_data_from_param(column_linear.weight),
            get_data_from_param(column_linear.weight).fp8_meta,
            torch.float32,
        )
        avg_diff = (dequant_w.abs() - reference_linear.weight[hidden_dim_slice].abs()).norm(p=2).mean()
        assert avg_diff > lr, f"diff: {avg_diff}"
        torch.testing.assert_close(
            dequant_w,
            reference_linear.weight[hidden_dim_slice],
            rtol=constants.FP8_WEIGHT_RTOL_THRESHOLD,
            atol=constants.FP8_WEIGHT_ATOL_THRESHOLD,
        )
        if with_bias is True:
            torch.testing.assert_close(
                get_data_from_param(column_linear.bias),
                reference_linear.bias[hidden_dim_slice].to(torch.float16),
                rtol=constants.FP8_WEIGHT_RTOL_THRESHOLD,
                atol=constants.FP8_WEIGHT_ATOL_THRESHOLD,
            )

        # for (_, states), (_, ref_states) in zip(optim.state.items(), ref_optim.state.items()):
        #     torch.testing.assert_allclose(states["step"].cpu(), ref_states["step"].cpu())
        #     torch.testing.assert_allclose(
        #         states["exp_avg"], ref_states["exp_avg"][hidden_dim_slice],
        #         # rtol=constants.FP8_1ST_OPTIM_STATE_RTOL_THRESHOLD,
        #         # atol=constants.FP8_1ST_OPTIM_STATE_ATOL_THRESHOLD,
        #         rtol=0.1,
        #         atol=0.1
        #     )
        #     torch.testing.assert_allclose(
        #         states["exp_avg_sq"],
        #         ref_states["exp_avg_sq"][hidden_dim_slice],
        #         # rtol=constants.FP8_2ND_OPTIM_STATE_RTOL_THRESHOLD,
        #         # atol=constants.FP8_2ND_OPTIM_STATE_ATOL_THRESHOLD,
        #         rtol=0.1,
        #         atol=0.1
        #     )

    parallel_context.destroy()
