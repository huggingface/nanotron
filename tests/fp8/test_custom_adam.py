import pytest
import torch
from nanotron import distributed as dist
from nanotron.fp8.optim import Adam as CustomAdam
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import get_grad_from_parameter
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
)
from nanotron.testing.parallel import init_distributed, rerun_if_address_is_in_use

# from helpers.utils import available_gpus, init_distributed, rerun_if_address_is_in_use
from nanotron.testing.utils import available_gpus
from torch import nn as torch_nn
from torch.optim import Adam


@pytest.mark.parametrize("tp,dp,pp", [pytest.param(i, 1, 1) for i in range(1, min(4, available_gpus()) + 1)])
# @pytest.mark.parametrize("tp_mode", list(TensorParallelLinearMode))
# @pytest.mark.parametrize("async_communication", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize("lr", [0.001, 1.0])
@pytest.mark.parametrize("steps", [1, 5, 10])
@rerun_if_address_is_in_use()
def test_column_linear(
    tp: int,
    dp: int,
    pp: int,
    weight_decay: float,
    lr: float,
    steps: int
    # tp_mode: TensorParallelLinearMode, async_communication: bool
):
    # if tp_mode is TensorParallelLinearMode.ALL_REDUCE and async_communication:
    #     pytest.skip("ALL_REDUCE mode does not support async communication")

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_column_linear)(
        steps=steps,
        lr=lr,
        weight_decay=weight_decay
        # tp_mode=tp_mode, async_communication=async_communication
    )


def _test_column_linear(
    parallel_context: ParallelContext,
    lr: float,
    weight_decay: float,
    steps: int
    # tp_mode: TensorParallelLinearMode, async_communication: bool
):
    # if async_communication:
    #     os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    in_features = 2
    out_features_per_tp_rank = 3
    out_features = parallel_context.tp_pg.size() * out_features_per_tp_rank

    column_linear = TensorParallelColumnLinear(
        in_features=in_features,
        out_features=out_features,
        pg=parallel_context.tp_pg,
        mode=TensorParallelLinearMode.ALL_REDUCE,
        device="cuda",
        async_communication=False,
    )
    reference_linear = torch_nn.Linear(in_features=in_features, out_features=out_features, device="cuda")

    # Copy weights/bias from sharded to un-sharded
    with torch.inference_mode():
        dist.all_gather(
            tensor_list=list(reference_linear.weight.split(out_features_per_tp_rank, dim=0)),
            tensor=column_linear.weight,
            group=parallel_context.tp_pg,
        )
        dist.all_gather(
            tensor_list=list(reference_linear.bias.split(out_features_per_tp_rank, dim=0)),
            tensor=column_linear.bias,
            group=parallel_context.tp_pg,
        )

    batch_size = 5
    random_input = torch.randn(batch_size, in_features, device="cuda")
    dist.all_reduce(random_input, op=dist.ReduceOp.AVG, group=parallel_context.tp_pg)
    sharded_random_input = random_input
    sharded_random_input = sharded_random_input.clone()
    # random_input.requires_grad = True
    # sharded_random_input.requires_grad = True

    optim = CustomAdam(column_linear.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay)
    ref_optim = Adam(reference_linear.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay)

    hidden_dim_slice = slice(
        dist.get_rank(parallel_context.tp_pg) * out_features_per_tp_rank,
        (dist.get_rank(parallel_context.tp_pg) + 1) * out_features_per_tp_rank,
    )

    for i in range(steps):
        optim.zero_grad()
        ref_optim.zero_grad()

        sharded_output = column_linear(sharded_random_input)
        reference_output = reference_linear(random_input)

        try:
            torch.testing.assert_close(
                sharded_output,
                reference_output[
                    :,
                    dist.get_rank(parallel_context.tp_pg)
                    * out_features_per_tp_rank : (dist.get_rank(parallel_context.tp_pg) + 1)
                    * out_features_per_tp_rank,
                ],
            )
        except BaseException as e:
            print(f"Rank {dist.get_rank(parallel_context.tp_pg)}: FAIL.")
            dist.barrier()
            raise e

        dist.barrier()

        sharded_output.sum().backward()
        reference_output.sum().backward()

        torch.testing.assert_close(
            get_grad_from_parameter(column_linear.weight),
            reference_linear.weight.grad[hidden_dim_slice],
        )
        torch.testing.assert_close(
            get_grad_from_parameter(column_linear.bias),
            reference_linear.bias.grad[hidden_dim_slice],
        )

        orig_weight = column_linear.weight.data.clone()

        torch.testing.assert_close(
            orig_weight,
            column_linear.weight.data,
        )

        optim.step()
        ref_optim.step()

        # assert (column_linear.weight.data.abs() - orig_weight.abs()).norm().item() > 1.0
        assert id(optim.param_groups[0]["params"][0]) == id(column_linear.weight)
        assert id(optim.param_groups[0]["params"][1]) == id(column_linear.bias)
        assert id(optim.param_groups[0]["params"][0].data) == id(column_linear.weight.data)
        assert id(optim.param_groups[0]["params"][1].data) == id(column_linear.bias.data)

        if i == 4:
            assert 1 == 1

        torch.testing.assert_close(
            column_linear.weight.data,
            reference_linear.weight[hidden_dim_slice],
        )
        torch.testing.assert_close(
            column_linear.bias.data,
            reference_linear.bias[hidden_dim_slice],
        )

        for (_, states), (_, ref_states) in zip(optim.state.items(), ref_optim.state.items()):
            torch.testing.assert_allclose(states["step"], ref_states["step"])
            torch.testing.assert_allclose(states["exp_avg"], ref_states["exp_avg"][hidden_dim_slice])
            torch.testing.assert_allclose(states["exp_avg_sq"], ref_states["exp_avg_sq"][hidden_dim_slice])

    parallel_context.destroy()
