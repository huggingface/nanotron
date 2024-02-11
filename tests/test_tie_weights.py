import torch
from helpers.distributed_tensor import assert_tensor_equal_over_group
from helpers.exception import assert_fail_with
from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron import distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.tied_parameters import (
    get_tied_id_to_param,
    sync_tied_weights_gradients,
    tie_parameters,
)
from torch import nn


@rerun_if_address_is_in_use()
def test_tie_weight_in_same_device():
    init_distributed(tp=1, dp=1, pp=1)(_test_tie_weight_in_same_device)()


def _test_tie_weight_in_same_device(parallel_context: ParallelContext):
    model = nn.ModuleDict({"dense0": nn.Linear(10, 10, device="cuda"), "dense1": nn.Linear(10, 10, device="cuda")})

    # Tie weights/bias
    tie_parameters(
        root_module=model,
        ties=[("dense0.weight", (0,)), ("dense1.weight", (0,))],
        parallel_context=parallel_context,
        reduce_op=dist.ReduceOp.SUM,
    )
    tie_parameters(
        root_module=model,
        ties=[("dense0.bias", (0,)), ("dense1.bias", (0,))],
        parallel_context=parallel_context,
        reduce_op=dist.ReduceOp.SUM,
    )

    weight0 = model.get_parameter("dense0.weight")
    weight1 = model.get_parameter("dense1.weight")
    bias0 = model.get_parameter("dense0.bias")
    bias1 = model.get_parameter("dense1.bias")

    # We check that we use the same parameter for both linear layers
    assert id(weight0) == id(weight1)
    assert id(bias0) == id(bias1)

    parallel_context.destroy()


@rerun_if_address_is_in_use()
def test_tie_weight_in_different_device():
    init_distributed(tp=1, dp=1, pp=2)(_test_tie_weight_in_different_device)()


def _test_tie_weight_in_different_device(parallel_context: ParallelContext):
    if dist.get_rank(parallel_context.pp_pg) == 0:
        model = nn.ModuleDict(
            {
                "dense0": nn.Linear(10, 10, device="cuda"),
            }
        )
    else:
        model = nn.ModuleDict(
            {
                "dense1": nn.Linear(10, 10, device="cuda"),
            }
        )

    # Tie weights/bias
    tie_parameters(
        root_module=model,
        ties=[("dense0.weight", (0,)), ("dense1.weight", (1,))],
        parallel_context=parallel_context,
        reduce_op=dist.ReduceOp.SUM,
    )
    tie_parameters(
        root_module=model,
        ties=[("dense0.bias", (0,)), ("dense1.bias", (1,))],
        parallel_context=parallel_context,
        reduce_op=dist.ReduceOp.SUM,
    )

    group = parallel_context.world_ranks_to_pg[(0, 1)]

    # Check that model weights are not in fact synchronized
    if dist.get_rank(parallel_context.pp_pg) == 0:
        weight = model.dense0.weight
        bias = model.dense0.bias
    else:
        weight = model.dense1.weight
        bias = model.dense1.bias

    # Make sure that weight/bias are NanotronParameter and that they are tied
    assert isinstance(weight, NanotronParameter)
    assert weight.is_tied
    assert isinstance(bias, NanotronParameter)
    assert bias.is_tied

    # Weights/bias are not synced yet
    assert not assert_tensor_equal_over_group(weight, group=group, assert_=False)
    assert not assert_tensor_equal_over_group(bias, group=group, assert_=False)

    # Manually sync weights
    for (_, group_ranks), param in sorted(
        get_tied_id_to_param(
            parameters=model.parameters(),
            root_module=model,
        ).items(),
        key=lambda x: x[0],
    ):
        group = parallel_context.world_ranks_to_pg[group_ranks]
        dist.all_reduce(param, op=dist.ReduceOp.AVG, group=group)

    # We check that we use the same parameter for both linear layers
    assert_tensor_equal_over_group(weight, group=group)
    assert_tensor_equal_over_group(bias, group=group)

    parallel_context.destroy()


@rerun_if_address_is_in_use()
def test_tie_weight_across_dp_is_impossible():
    init_distributed(tp=1, dp=2, pp=1)(_test_tie_weight_across_dp_is_impossible)()


def _test_tie_weight_across_dp_is_impossible(parallel_context: ParallelContext):
    if dist.get_rank(parallel_context.dp_pg) == 0:
        model = nn.ModuleDict(
            {
                "dense0": nn.Linear(10, 10, device="cuda"),
            }
        )
    else:
        model = nn.ModuleDict(
            {
                "dense1": nn.Linear(10, 10, device="cuda"),
            }
        )

    # Tie weights/bias
    with assert_fail_with(AssertionError):
        tie_parameters(
            root_module=model,
            ties=[("dense0.weight", (0,)), ("dense1.weight", (1,))],
            parallel_context=parallel_context,
            reduce_op=dist.ReduceOp.SUM,
        )
    with assert_fail_with(AssertionError):
        tie_parameters(
            root_module=model,
            ties=[("dense0.bias", (0,)), ("dense1.bias", (1,))],
            parallel_context=parallel_context,
            reduce_op=dist.ReduceOp.SUM,
        )

    parallel_context.destroy()


@rerun_if_address_is_in_use()
def test_tie_weight_in_different_device_have_gradients_synchronized():
    init_distributed(tp=1, dp=1, pp=2)(_test_tie_weight_in_different_device_have_gradients_synchronized)()


def _test_tie_weight_in_different_device_have_gradients_synchronized(parallel_context: ParallelContext):
    if dist.get_rank(parallel_context.pp_pg) == 0:
        model = nn.ModuleDict(
            {
                "dense0": nn.Linear(10, 10, device="cuda"),
            }
        )
    else:
        model = nn.ModuleDict(
            {
                "dense1": nn.Linear(10, 10, device="cuda"),
            }
        )

    # Tie weights/bias
    tie_parameters(
        root_module=model,
        ties=[("dense0.weight", (0,)), ("dense1.weight", (1,))],
        parallel_context=parallel_context,
        reduce_op=dist.ReduceOp.SUM,
    )
    tie_parameters(
        root_module=model,
        ties=[("dense0.bias", (0,)), ("dense1.bias", (1,))],
        parallel_context=parallel_context,
        reduce_op=dist.ReduceOp.SUM,
    )

    group = parallel_context.world_ranks_to_pg[(0, 1)]

    # Check that model weights are not in fact synchronized
    if dist.get_rank(parallel_context.pp_pg) == 0:
        weight = model.dense0.weight
        bias = model.dense0.bias
    else:
        weight = model.dense1.weight
        bias = model.dense1.bias

    # Make sure that weight/bias are NanotronParameter and that they are tied
    assert isinstance(weight, NanotronParameter)
    assert weight.is_tied
    assert isinstance(bias, NanotronParameter)
    assert bias.is_tied

    # Weights/bias are not synced yet
    assert not assert_tensor_equal_over_group(weight, group=group, assert_=False)
    assert not assert_tensor_equal_over_group(bias, group=group, assert_=False)

    # Compute gradient
    input_ = torch.randn(13, 10, device="cuda")
    if dist.get_rank(parallel_context.pp_pg) == 0:
        out = model.dense0(input_)
    else:
        out = model.dense1(input_)
    out.sum().backward()

    # sync gradients
    # TODO @thomasw21: This should be done in hooks
    sync_tied_weights_gradients(model, parallel_context=parallel_context, grad_accumulator=None)

    # Check that we have gradient
    assert weight.grad is not None
    assert bias.grad is not None

    # We check that we both gradients are synchronized
    assert_tensor_equal_over_group(weight.grad, group=group)
    assert_tensor_equal_over_group(bias.grad, group=group)

    parallel_context.destroy()
