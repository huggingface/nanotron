import os

import pytest
import torch
from helpers.distributed_tensor import assert_tensor_equal_over_group
from helpers.dummy import dummy_infinite_data_loader, init_dummy_model
from helpers.exception import assert_fail_with
from helpers.utils import available_gpus, init_distributed, rerun_if_address_is_in_use
from nanotron import distributed as dist
from nanotron.optim import NamedOptimizer, ZeroDistributedOptimizer
from nanotron.optim.zero import SlicedFlatTensor
from nanotron.parallel import ParallelContext
from nanotron.parallel.data_parallel.utils import sync_gradients_across_dp
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.engine import AllForwardAllBackwardPipelineEngine
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.tensor_parallel import nn
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tied_parameters import sync_tied_weights_gradients
from nanotron.random import RandomStates, branch_random_state, get_current_random_state, get_synced_random_state
from torch import nn as torch_nn
from torch.nn.parallel import DistributedDataParallel


@pytest.mark.parametrize("tp,dp,pp", [pytest.param(1, i, 1) for i in range(1, min(4, available_gpus()) + 1)])
@rerun_if_address_is_in_use()
def test_zero_optimizer(tp: int, dp: int, pp: int):
    init_distributed(pp=pp, dp=dp, tp=tp)(_test_zero_optimizer)()


def _test_zero_optimizer(parallel_context: ParallelContext):
    model = init_dummy_model(parallel_context=parallel_context)
    optimizer = ZeroDistributedOptimizer(
        named_params_or_groups=model.named_parameters(),
        optimizer_builder=lambda named_param_groups: NamedOptimizer(
            named_params_or_groups=named_param_groups,
            optimizer_builder=lambda param_groups: torch.optim.AdamW(param_groups),
        ),
        dp_pg=parallel_context.dp_pg,
    )
    index_to_name = [name for name, _ in model.named_parameters()]

    # reference model
    reference_model = init_dummy_model(parallel_context=parallel_context)
    reference_optimizer = torch.optim.AdamW(reference_model.parameters())

    # sync weights between reference_model and model
    with torch.no_grad():
        for (name, param), (ref_name, ref_param) in zip(model.named_parameters(), reference_model.named_parameters()):
            assert name == ref_name
            param.copy_(ref_param)

    # Get infinite dummy data iterator
    data_loader = iter(dummy_infinite_data_loader(pp_pg=parallel_context.pp_pg))
    nb_optim_steps = 3
    batches = [[next(data_loader)] for _ in range(nb_optim_steps)]
    pipeline_engine = AllForwardAllBackwardPipelineEngine()

    # Training loop
    for i, batch in enumerate(batches):
        # store original reference parameter
        old_named_params = {name: param.detach().clone() for name, param in model.named_parameters()}

        # Run forward/backward
        losses = pipeline_engine.train_batch_iter(
            model=model, pg=parallel_context.pp_pg, batch=batch, nb_microbatches=1, grad_accumulator=None
        )
        ref_losses = pipeline_engine.train_batch_iter(
            model=reference_model, pg=parallel_context.pp_pg, batch=batch, nb_microbatches=1, grad_accumulator=None
        )

        # Check loss match
        losses = list(losses)
        ref_losses = list(ref_losses)
        assert len(losses) == len(ref_losses)
        for loss, ref_loss in zip(losses, ref_losses):
            assert isinstance(loss["loss"], torch.Tensor)
            assert isinstance(ref_loss["loss"], torch.Tensor)
            torch.testing.assert_close(
                loss["loss"], ref_loss["loss"], atol=0, rtol=0, msg=lambda msg: f"At iteration {i}, {msg}"
            )

        # Manually sync tied parameters' gradients
        sync_tied_weights_gradients(module=model, parallel_context=parallel_context, grad_accumulator=None)
        sync_tied_weights_gradients(module=reference_model, parallel_context=parallel_context, grad_accumulator=None)

        # We rely on DDP to synchronize gradients across DP. We only need to manually synchronize them if we don't use DDP.
        if not isinstance(model, DistributedDataParallel):
            sync_gradients_across_dp(
                model, dp_pg=parallel_context.dp_pg, reduce_op=dist.ReduceOp.AVG, grad_accumulator=None
            )
        if not isinstance(reference_model, DistributedDataParallel):
            sync_gradients_across_dp(
                reference_model, dp_pg=parallel_context.dp_pg, reduce_op=dist.ReduceOp.AVG, grad_accumulator=None
            )

        # Check gradients are synced across DP
        for name, param in model.named_parameters():
            assert_tensor_equal_over_group(param.grad, group=parallel_context.dp_pg)
        for ref_name, ref_param in reference_model.named_parameters():
            assert_tensor_equal_over_group(ref_param.grad, group=parallel_context.dp_pg)

        # Check gradients are the same with reference_model
        for (name, param), (ref_name, ref_param) in zip(model.named_parameters(), reference_model.named_parameters()):
            assert name == ref_name
            torch.testing.assert_close(
                param.grad, ref_param.grad, atol=0, rtol=0, msg=lambda msg: f"At iteration {i}, {msg}"
            )

        assert len(optimizer.param_groups) == 1
        assert len(list(model.named_parameters())) == len(optimizer.param_groups[0]["params"])
        with torch.no_grad():
            for (name, param), sliced_param in zip(model.named_parameters(), optimizer.param_groups[0]["params"]):
                offsets = optimizer.param_name_to_dp_rank_offsets[name][dist.get_rank(parallel_context.dp_pg)]

                # Check that weights are the same
                expected_slice = param.view(-1)[slice(*offsets)].view_as(sliced_param)
                torch.testing.assert_close(
                    expected_slice,
                    sliced_param,
                    atol=0,
                    rtol=0,
                    msg=lambda msg: f"Weights don't match: {msg}\n - Expected slice: {expected_slice}\n - Got: {sliced_param}\n - Full gradient: {param}",
                )
                assert (
                    expected_slice.data_ptr() == sliced_param.data_ptr()
                ), "Parameters should actually share the same data pointer"

                # Check gradients is the view
                expected_slice = param.grad.view(-1)[slice(*offsets)].view_as(sliced_param.grad)
                assert (
                    expected_slice.data_ptr() == sliced_param.grad.data_ptr()
                ), "Parameters should actually share the same data pointer"
                torch.testing.assert_close(
                    expected_slice,
                    sliced_param.grad,
                    atol=0,
                    rtol=0,
                    msg=lambda msg: f"Gradients don't match: {msg}\n - Expected slice: {expected_slice}\n - Got: {sliced_param.grad}\n - Full gradient: {param.grad}",
                )

        # Optimizer steps
        optimizer.step()
        optimizer.zero_grad()
        reference_optimizer.step()
        reference_optimizer.zero_grad()

        # Check that params are synced across DP
        for name, param in model.named_parameters():
            assert_tensor_equal_over_group(param, group=parallel_context.dp_pg)
            assert param.grad is None

        # Check that gradients are reset
        for ref_name, ref_param in reference_model.named_parameters():
            assert_tensor_equal_over_group(ref_param, group=parallel_context.dp_pg)
            assert ref_param.grad is None
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                assert param.grad is None

        # Check params are the same with reference_model
        for (name, param), (ref_name, ref_param) in zip(model.named_parameters(), reference_model.named_parameters()):
            assert name == ref_name
            # TODO @thomasw21: Figure out how to make this pass at `atol`/`rtol` set to 0.
            torch.testing.assert_close(param, ref_param, msg=lambda msg: f"At iteration {i}, {msg}")

        # Check params have been updated correctly
        for (name, param) in model.named_parameters():
            old_param = old_named_params[name]
            assert not torch.allclose(param, old_param)

        # We need to check that the optimizer states are the same
        state_dict = optimizer.state_dict()
        reference_state_dict = reference_optimizer.state_dict()
        state = state_dict["state"]
        ref_state = reference_state_dict["state"]
        assert set(state) == set(ref_state)

        for index, optim_state in state.items():
            ref_optim_state = ref_state[index]

            name = index_to_name[index]
            offsets = optimizer.param_name_to_dp_rank_offsets[name][dist.get_rank(parallel_context.dp_pg)]

            assert set(optim_state) == set(ref_optim_state)

            for key in ["exp_avg", "exp_avg_sq"]:
                value = optim_state[key]
                ref_value = ref_optim_state[key]
                torch.testing.assert_close(
                    value,
                    ref_value.view(-1)[slice(*offsets)].view_as(value),
                    atol=0,
                    rtol=0,
                    msg=lambda msg: f"At iteration {i}, {msg}",
                )

    parallel_context.destroy()


@pytest.mark.parametrize("tp,dp,pp", [pytest.param(2, i, 1) for i in range(1, available_gpus() // 2 + 1)])
@pytest.mark.parametrize("tp_mode", list(TensorParallelLinearMode))
@pytest.mark.parametrize("async_communication", [False, True])
@rerun_if_address_is_in_use()
def test_zero_optimizer_with_tp(
    tp: int, dp: int, pp: int, tp_mode: TensorParallelLinearMode, async_communication: bool
):
    if tp_mode is TensorParallelLinearMode.ALL_REDUCE and async_communication:
        pytest.skip("ALL_REDUCE mode does not support async communication")
    init_distributed(pp=pp, dp=dp, tp=tp)(_test_zero_optimizer_with_tp)(
        tp_mode=tp_mode, async_communication=async_communication
    )


def _test_zero_optimizer_with_tp(
    parallel_context: ParallelContext, tp_mode: TensorParallelLinearMode, async_communication: bool
):
    if async_communication:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    model = torch_nn.Sequential(
        nn.TensorParallelColumnLinear(
            in_features=5,
            out_features=parallel_context.tp_pg.size(),
            mode=tp_mode,
            pg=parallel_context.tp_pg,
            device="cuda",
            async_communication=async_communication,
        ),
        # We choose `sigmoid` instead of `relu` since `relu` can result in a sparse gradient, causing no update to certain parameters
        torch_nn.Sigmoid(),
        nn.TensorParallelRowLinear(
            in_features=parallel_context.tp_pg.size(),
            out_features=3,
            mode=tp_mode,
            pg=parallel_context.tp_pg,
            device="cuda",
        ),
    )
    optimizer = ZeroDistributedOptimizer(
        named_params_or_groups=model.named_parameters(),
        optimizer_builder=lambda named_param_groups: NamedOptimizer(
            named_params_or_groups=named_param_groups,
            optimizer_builder=lambda param_groups: torch.optim.AdamW(param_groups),
        ),
        dp_pg=parallel_context.dp_pg,
    )
    optimizer_name_to_id = {v: k for k, v in optimizer.optimizer.id_to_name.items()}
    assert len(optimizer_name_to_id) == len(optimizer.id_to_name)

    # reference model
    reference_model = torch_nn.Sequential(
        torch_nn.Linear(in_features=5, out_features=parallel_context.tp_pg.size(), device="cuda"),
        torch_nn.Sigmoid(),
        torch_nn.Linear(in_features=parallel_context.tp_pg.size(), out_features=3, device="cuda"),
    )
    for module in reference_model.modules():
        for name, param in module.named_parameters(recurse=False):
            setattr(module, name, NanotronParameter(param))

    reference_optimizer = torch.optim.AdamW(reference_model.parameters())
    # TODO @thomasw21: This is a hack to obtain `AdamW` index in it's state.
    name_to_index = {name: index for index, (name, _) in enumerate(reference_model.named_parameters())}

    # sync parameters
    with torch.no_grad():
        for ref_name, ref_param in reference_model.named_parameters():
            dist.all_reduce(ref_param, op=dist.ReduceOp.AVG, group=parallel_context.world_pg)

        for (name, param), (ref_name, ref_param) in zip(model.named_parameters(), reference_model.named_parameters()):
            assert name == ref_name
            assert isinstance(param, NanotronParameter)

            if param.is_sharded:
                sharded_info = param.get_sharded_info()
                for local_global_slices_pair in sharded_info.local_global_slices_pairs:
                    local_slices = local_global_slices_pair.local_slices
                    global_slices = local_global_slices_pair.global_slices
                    param[local_slices].copy_(ref_param[global_slices])
            else:
                param.copy_(ref_param)

    # Get infinite dummy data iterator, it has to be synced across TP
    random_states = RandomStates(
        {
            "tp_synced": get_synced_random_state(random_state=get_current_random_state(), pg=parallel_context.tp_pg),
        }
    )
    batch_size = 2 * parallel_context.tp_pg.size() if tp_mode is TensorParallelLinearMode.REDUCE_SCATTER else 7
    with branch_random_state(random_states=random_states, key="tp_synced", enabled=True):
        nb_optim_steps = 3
        batches = [
            torch.randn(batch_size, 5, device="cuda")
            if dist.get_rank(parallel_context.pp_pg) == 0
            else TensorPointer(0)
            for _ in range(nb_optim_steps)
        ]

    # Model training loop
    for i, batch in enumerate(batches):
        # store original reference parameter
        old_named_params = {name: param.detach().clone() for name, param in model.named_parameters()}

        # Run forward pass
        if tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
            batch_size = batch.shape[0]
            assert batch_size % parallel_context.tp_pg.size() == 0
            step = batch_size // parallel_context.tp_pg.size()
            loss = model(
                batch[
                    dist.get_rank(parallel_context.tp_pg) * step : (dist.get_rank(parallel_context.tp_pg) + 1) * step
                ]
            )
        else:
            loss = model(batch)
        ref_loss = reference_model(batch)

        # Run backward pass
        loss.sum().backward()
        ref_loss.sum().backward()

        # Check loss is the same
        loss = loss.detach()
        ref_loss = ref_loss.detach()
        assert isinstance(loss, torch.Tensor)
        assert isinstance(ref_loss, torch.Tensor)
        if tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
            batch_size = batch.shape[0]
            assert batch_size % parallel_context.tp_pg.size() == 0
            step = batch_size // parallel_context.tp_pg.size()
            torch.testing.assert_close(
                loss,
                ref_loss[
                    dist.get_rank(parallel_context.tp_pg) * step : (dist.get_rank(parallel_context.tp_pg) + 1) * step
                ],
                msg=lambda msg: f"At iteration {i}, {msg}",
            )
        else:
            torch.testing.assert_close(loss, ref_loss, msg=lambda msg: f"At iteration {i}, {msg}")

        # Manually sync tied parameters
        sync_tied_weights_gradients(module=model, parallel_context=parallel_context, grad_accumulator=None)
        sync_tied_weights_gradients(module=reference_model, parallel_context=parallel_context, grad_accumulator=None)

        # We rely on DDP to synchronize gradients across DP. We only need to manually synchronize them if we don't use DDP.
        if not isinstance(model, DistributedDataParallel):
            sync_gradients_across_dp(
                model, dp_pg=parallel_context.dp_pg, reduce_op=dist.ReduceOp.AVG, grad_accumulator=None
            )
        if not isinstance(reference_model, DistributedDataParallel):
            sync_gradients_across_dp(
                reference_model, dp_pg=parallel_context.dp_pg, reduce_op=dist.ReduceOp.AVG, grad_accumulator=None
            )

        # Check gradients are synced across DP
        for name, param in model.named_parameters():
            assert_tensor_equal_over_group(param.grad, group=parallel_context.dp_pg)
        for ref_name, ref_param in reference_model.named_parameters():
            assert_tensor_equal_over_group(ref_param.grad, group=parallel_context.dp_pg)

        # Check gradients are the same with reference_model
        for (name, param), (ref_name, ref_param) in zip(model.named_parameters(), reference_model.named_parameters()):
            assert name == ref_name

            if param.is_sharded:
                sharded_info = param.get_sharded_info()
                for local_global_slices_pair in sharded_info.local_global_slices_pairs:
                    local_slices = local_global_slices_pair.local_slices
                    global_slices = local_global_slices_pair.global_slices
                    torch.testing.assert_close(
                        param.grad[local_slices],
                        ref_param.grad[global_slices],
                        msg=lambda msg: f"At iteration {i}, {msg}",
                    )
            else:
                torch.testing.assert_close(param.grad, ref_param.grad, msg=lambda msg: f"At iteration {i}, {msg}")

        with torch.no_grad():
            optim_param_id_to_param = {id(param): param for param in optimizer.param_groups[0]["params"]}
            assert len(optim_param_id_to_param) == len(optimizer.param_groups[0]["params"])
            for name, param in model.named_parameters():
                if dist.get_rank(parallel_context.dp_pg) not in optimizer.param_name_to_dp_rank_offsets[name]:
                    assert name not in optimizer_name_to_id
                    continue

                param_id = optimizer_name_to_id[name]
                sliced_param = optim_param_id_to_param[param_id]
                offsets = optimizer.param_name_to_dp_rank_offsets[name][dist.get_rank(parallel_context.dp_pg)]

                # Check that weights share the same storage
                expected_slice = param.view(-1)[slice(*offsets)].view_as(sliced_param)
                torch.testing.assert_close(
                    expected_slice,
                    sliced_param,
                    atol=0,
                    rtol=0,
                    msg=lambda msg: f"At iteration {i}, weights don't match: {msg}\n - Expected slice: {expected_slice}\n - Got: {sliced_param}\n - Full gradient: {param}",
                )
                assert (
                    expected_slice.data_ptr() == sliced_param.data_ptr()
                ), "Parameters should actually share the same data pointer"

                # Check that gradients share the same storage
                expected_slice = param.grad.view(-1)[slice(*offsets)].view_as(sliced_param.grad)
                assert (
                    expected_slice.data_ptr() == sliced_param.grad.data_ptr()
                ), "Parameters should actually share the same data pointer"
                torch.testing.assert_close(
                    expected_slice,
                    sliced_param.grad,
                    atol=0,
                    rtol=0,
                    msg=lambda msg: f"At iteration {i}, gradients don't match: {msg}\n - Expected slice: {expected_slice}\n - Got: {sliced_param.grad}\n - Full gradient: {param.grad}",
                )

        # Optimizer steps
        optimizer.step()
        optimizer.zero_grad()
        reference_optimizer.step()
        reference_optimizer.zero_grad()

        # Check that params are synced across DP
        for name, param in model.named_parameters():
            assert_tensor_equal_over_group(param, group=parallel_context.dp_pg)
            assert param.grad is None

        # Check that gradients are reset
        for ref_name, ref_param in reference_model.named_parameters():
            assert_tensor_equal_over_group(ref_param, group=parallel_context.dp_pg)
            assert ref_param.grad is None
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                assert param.grad is None

        # Check params are the same with reference_model
        for (name, param), (ref_name, ref_param) in zip(model.named_parameters(), reference_model.named_parameters()):
            assert name == ref_name
            if param.is_sharded:
                sharded_info = param.get_sharded_info()
                for local_global_slices_pair in sharded_info.local_global_slices_pairs:
                    local_slices = local_global_slices_pair.local_slices
                    global_slices = local_global_slices_pair.global_slices
                    torch.testing.assert_close(
                        param[local_slices], ref_param[global_slices], msg=lambda msg: f"At iteration {i}, {msg}"
                    )
            else:
                torch.testing.assert_close(param, ref_param, msg=lambda msg: f"At iteration {i}, {msg}")

        # Check params have been updated correctly:
        for (name, param) in model.named_parameters():
            old_param = old_named_params[name]
            assert not torch.allclose(param, old_param)

        # We need to check that the optimizer states are the same
        state_dict = optimizer.state_dict()
        reference_state_dict = reference_optimizer.state_dict()
        state = state_dict["state"]
        ref_state = reference_state_dict["state"]

        assert "names" in state_dict
        state_index_to_name = state_dict["names"]
        state_name_to_index = {name: index for index, name in state_index_to_name.items()}
        # Check that this is a bijection
        assert len(state_index_to_name) == len(state_name_to_index)

        for name, param in model.named_parameters():
            if name not in state_name_to_index:
                # Parameters is not passed to optimizer, mainly due to zero sharding strategy
                continue

            index = state_name_to_index[name]
            optim_state = state[index]

            ref_optim_state = ref_state[name_to_index[name]]

            offsets = optimizer.param_name_to_dp_rank_offsets[name][dist.get_rank(parallel_context.dp_pg)]

            assert set(optim_state) == set(ref_optim_state)
            assert isinstance(param, NanotronParameter)
            for key in ["exp_avg", "exp_avg_sq"]:
                value = optim_state[key]
                ref_value = ref_optim_state[key]
                if param.is_sharded:
                    sharded_info = param.get_sharded_info()

                    for local_global_slices_pair in sharded_info.local_global_slices_pairs:
                        global_slices = local_global_slices_pair.global_slices
                        torch.testing.assert_close(
                            # TODO @thomasw21: We can't add any information about `local_slices` to `value` because it's already flattened
                            #  For now, we're going to assume that sharded parameters are contiguous, and `local_slices` are trivial all none slice
                            value,
                            ref_value[global_slices].view(-1)[slice(*offsets)],
                            msg=lambda msg: f"At iteration {i}, {msg}",
                        )
                else:
                    torch.testing.assert_close(
                        value,
                        ref_value.view(-1)[slice(*offsets)].view_as(value),
                        msg=lambda msg: f"At iteration {i}, {msg}",
                    )

    parallel_context.destroy()


@rerun_if_address_is_in_use()
def test_sliced_flat_tensor():
    init_distributed(1, 1, 1)(_test_sliced_flat_tensor)()


def _test_sliced_flat_tensor(parallel_context: ParallelContext):
    a = torch.randn(2, 3, requires_grad=True)
    grad = torch.randn(2, 3)
    a.grad = grad

    start_offset, end_offset = 1, 5
    b = SlicedFlatTensor(a, start_offset=start_offset, end_offset=end_offset)

    torch.testing.assert_close(a.grad, grad, atol=0, rtol=0)
    torch.testing.assert_close(b.grad, grad.view(-1)[start_offset:end_offset])

    # Deallocate the gradient by setting it to None
    a.grad = None

    assert a.grad is None
    assert b.grad is None

    # Setting gradient to None on the sliced tensor works
    a.grad = grad
    assert a.grad is not None
    assert b.grad is not None
    b.grad = None
    assert b.grad is None
    assert a.grad is None

    with assert_fail_with(NotImplementedError):
        b.grad = torch.randn(1, 5)

    with assert_fail_with(NotImplementedError):
        del b.grad

    c = b[:3]
    # It's important not to contaminate everyone.
    assert not isinstance(c, SlicedFlatTensor)

    parallel_context.destroy()
