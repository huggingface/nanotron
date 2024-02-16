import math
import os

import pytest
import torch
from helpers.dummy import DummyModel, dummy_infinite_data_loader
from helpers.utils import available_gpus, init_distributed, rerun_if_address_is_in_use
from nanotron import distributed as dist
from nanotron.models import init_on_device_and_dtype
from nanotron.optim.clip_grads import clip_grad_norm
from nanotron.optim.gradient_accumulator import (
    FP32GradientAccumulator,
)
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter, sanity_check
from nanotron.parallel.pipeline_parallel.engine import (
    AllForwardAllBackwardPipelineEngine,
)
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
)
from nanotron.parallel.tied_parameters import (
    sync_tied_weights_gradients,
    tie_parameters,
)
from nanotron.parallel.utils import initial_sync
from nanotron.sanity_checks import assert_tensor_synced_across_pg
from torch import nn


@pytest.mark.skipif(available_gpus() < 2, reason="test_clip_grads_with_pp requires at least 2 gpus")
@pytest.mark.parametrize("norm_type", [math.inf, 1.0, 2.0])
@rerun_if_address_is_in_use()
def test_clip_grads_with_pp(norm_type: float):
    init_distributed(tp=1, dp=1, pp=2)(_test_clip_grads_with_pp)(norm_type=norm_type)


def _test_clip_grads_with_pp(parallel_context: ParallelContext, norm_type: float):
    device = torch.device("cuda")
    p2p = P2P(parallel_context.pp_pg, device=device)
    reference_rank = 0
    has_reference_model = dist.get_rank(parallel_context.pp_pg) == reference_rank
    pipeline_engine = AllForwardAllBackwardPipelineEngine()
    current_pp_rank = dist.get_rank(parallel_context.pp_pg)

    # spawn model
    model = DummyModel(p2p=p2p)
    if has_reference_model:
        reference_model = DummyModel(p2p=p2p)

    # Set the ranks
    assert len(model.mlp) == parallel_context.pp_pg.size()
    with init_on_device_and_dtype(device):
        for pp_rank, non_linear in zip(range(parallel_context.pp_pg.size()), model.mlp):
            non_linear.linear.build_and_set_rank(pp_rank=pp_rank)
            non_linear.activation.build_and_set_rank(pp_rank=pp_rank)
        model.loss.build_and_set_rank(pp_rank=parallel_context.pp_pg.size() - 1)

        # build reference model
        if has_reference_model:
            for non_linear in reference_model.mlp:
                non_linear.linear.build_and_set_rank(pp_rank=reference_rank)
                non_linear.activation.build_and_set_rank(pp_rank=reference_rank)
            reference_model.loss.build_and_set_rank(pp_rank=reference_rank)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            setattr(module, "weight", NanotronParameter(module.weight))
            setattr(module, "bias", NanotronParameter(module.bias))

    # synchronize weights
    if has_reference_model:
        with torch.inference_mode():
            for pp_rank in range(parallel_context.pp_pg.size()):
                reference_non_linear = reference_model.mlp[pp_rank].linear.pp_block
                if pp_rank == current_pp_rank:
                    # We already have the weights locally
                    non_linear = model.mlp[pp_rank].linear.pp_block
                    reference_non_linear.weight.data.copy_(non_linear.weight.data)
                    reference_non_linear.bias.data.copy_(non_linear.bias.data)
                    continue

                weight, bias = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
                reference_non_linear.weight.data.copy_(weight.data)
                reference_non_linear.bias.data.copy_(bias.data)
    else:
        p2p.send_tensors(
            [model.mlp[current_pp_rank].linear.pp_block.weight, model.mlp[current_pp_rank].linear.pp_block.bias],
            to_rank=reference_rank,
        )

    # Get infinite dummy data iterator
    data_iterator = dummy_infinite_data_loader(pp_pg=parallel_context.pp_pg)  # First rank receives data

    n_micro_batches_per_batch = 5
    batch = [next(data_iterator) for _ in range(n_micro_batches_per_batch)]
    pipeline_engine.train_batch_iter(
        model, pg=parallel_context.pp_pg, batch=batch, nb_microbatches=n_micro_batches_per_batch, grad_accumulator=None
    )

    # Equivalent on the reference model
    if has_reference_model:
        for micro_batch in batch:
            loss = reference_model(**micro_batch)
            loss /= n_micro_batches_per_batch
            loss.backward()

    # Check that gradient are the same as reference
    pp_rank = dist.get_rank(parallel_context.pp_pg)
    if has_reference_model:
        for pp_rank in range(parallel_context.pp_pg.size()):
            reference_non_linear = reference_model.mlp[pp_rank].linear.pp_block
            if pp_rank == current_pp_rank:
                # We already have the gradients locally
                non_linear = model.mlp[pp_rank].linear.pp_block
                torch.testing.assert_close(
                    non_linear.weight.grad,
                    reference_non_linear.weight.grad,
                    atol=1e-6,
                    rtol=1e-7,
                )
                torch.testing.assert_close(non_linear.bias.grad, reference_non_linear.bias.grad, atol=1e-6, rtol=1e-7)
                continue

            weight_grad, bias_grad = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
            torch.testing.assert_close(weight_grad, reference_non_linear.weight.grad, atol=1e-6, rtol=1e-7)
            torch.testing.assert_close(bias_grad, reference_non_linear.bias.grad, atol=1e-6, rtol=1e-7)
    else:
        p2p.send_tensors(
            [model.mlp[pp_rank].linear.pp_block.weight.grad, model.mlp[pp_rank].linear.pp_block.bias.grad],
            to_rank=reference_rank,
        )

    non_linear = model.mlp[current_pp_rank].linear.pp_block
    old_weight_grad = non_linear.weight.grad.clone()
    old_bias_grad = non_linear.bias.grad.clone()
    # Clip grads
    total_norm = clip_grad_norm(
        mp_pg=parallel_context.mp_pg,
        named_parameters=model.named_parameters(),
        grad_accumulator=None,
        max_norm=1.0,
        norm_type=norm_type,
    )
    if has_reference_model:
        reference_total_norm = torch.nn.utils.clip_grad_norm_(
            reference_model.parameters(), max_norm=1.0, norm_type=norm_type
        )
        torch.testing.assert_close(total_norm, reference_total_norm, atol=1e-6, rtol=1e-7)

    # Check that grad changed
    assert not torch.allclose(old_weight_grad, non_linear.weight.grad), "Grad should have changed"
    assert not torch.allclose(old_bias_grad, non_linear.weight.grad), "Grad should have changed"

    # Check that gradient are the same as reference
    if has_reference_model:
        for pp_rank in range(parallel_context.pp_pg.size()):
            reference_non_linear = reference_model.mlp[pp_rank].linear.pp_block
            if pp_rank == current_pp_rank:
                # We already have the gradients locally
                non_linear = model.mlp[pp_rank].linear.pp_block
                torch.testing.assert_close(
                    non_linear.weight.grad,
                    reference_non_linear.weight.grad,
                    atol=1e-6,
                    rtol=1e-7,
                )
                torch.testing.assert_close(
                    non_linear.bias.grad,
                    reference_non_linear.bias.grad,
                    atol=1e-6,
                    rtol=1e-7,
                )
                continue

            weight_grad, bias_grad = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
            torch.testing.assert_close(weight_grad, reference_non_linear.weight.grad, atol=1e-6, rtol=1e-7)
            torch.testing.assert_close(bias_grad, reference_non_linear.bias.grad, atol=1e-6, rtol=1e-7)
    else:
        p2p.send_tensors(
            [
                model.mlp[current_pp_rank].linear.pp_block.weight.grad,
                model.mlp[current_pp_rank].linear.pp_block.bias.grad,
            ],
            to_rank=reference_rank,
        )

    print(parallel_context.__dir__())

    parallel_context.destroy()


@pytest.mark.skipif(available_gpus() < 2, reason="test_clip_grads_with_tp requires at least 2 gpus")
@pytest.mark.parametrize(
    "tp_mode,async_communication",
    [
        pytest.param(TensorParallelLinearMode.ALL_REDUCE, False),
        pytest.param(TensorParallelLinearMode.REDUCE_SCATTER, True),
    ],
)
@pytest.mark.parametrize("norm_type", [math.inf, 1.0, 2.0])
@rerun_if_address_is_in_use()
def test_clip_grads_with_tp(tp_mode: TensorParallelLinearMode, async_communication: bool, norm_type: float):
    init_distributed(tp=2, dp=1, pp=1)(_test_clip_grads_with_tp)(
        tp_mode=tp_mode, async_communication=async_communication, norm_type=norm_type
    )


def _test_clip_grads_with_tp(
    parallel_context: ParallelContext, tp_mode: TensorParallelLinearMode, async_communication: bool, norm_type: float
):
    if async_communication:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    in_features = 2
    out_features_per_tp_rank = 3
    out_features = parallel_context.tp_pg.size() * out_features_per_tp_rank

    # Sharded
    column_linear = TensorParallelColumnLinear(
        in_features=in_features,
        out_features=out_features,
        pg=parallel_context.tp_pg,
        mode=tp_mode,
        device="cuda",
        async_communication=async_communication,
    )

    # Un-sharded
    reference_linear = nn.Linear(in_features=in_features, out_features=out_features, device="cuda")

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

    # Generate random input
    random_input: torch.Tensor
    sharded_random_input: torch.Tensor
    if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        batch_size = 5
        random_input = torch.randn(batch_size, in_features, device="cuda")
        # synchronize random_input across tp
        dist.all_reduce(random_input, op=dist.ReduceOp.AVG, group=parallel_context.tp_pg)
        sharded_random_input = random_input
    elif tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
        sharded_batch_size = 5
        sharded_random_input = torch.randn(sharded_batch_size, in_features, device="cuda")
        random_input = torch.empty(
            sharded_batch_size * parallel_context.tp_pg.size(),
            *(sharded_random_input.shape[1:]),
            device=sharded_random_input.device,
            dtype=sharded_random_input.dtype,
        )
        dist.all_gather_into_tensor(random_input, sharded_random_input, group=parallel_context.tp_pg)
    else:
        ValueError(f"Unsupported mode: {tp_mode}")

    # Test that we get the same output after forward pass
    sharded_output = column_linear(sharded_random_input)
    reference_output = reference_linear(random_input)
    # TODO @thomasw21: Tune tolerance
    torch.testing.assert_close(
        sharded_output,
        reference_output[
            :,
            dist.get_rank(parallel_context.tp_pg)
            * out_features_per_tp_rank : (dist.get_rank(parallel_context.tp_pg) + 1)
            * out_features_per_tp_rank,
        ],
        atol=1e-6,
        rtol=1e-7,
    )

    # Test that we get the same gradient after backward pass
    sharded_output.sum().backward()
    reference_output.sum().backward()
    torch.testing.assert_close(
        column_linear.weight.grad,
        reference_linear.weight.grad[
            dist.get_rank(parallel_context.tp_pg)
            * out_features_per_tp_rank : (dist.get_rank(parallel_context.tp_pg) + 1)
            * out_features_per_tp_rank
        ],
        atol=1e-6,
        rtol=1e-7,
    )
    torch.testing.assert_close(
        column_linear.bias.grad,
        reference_linear.bias.grad[
            dist.get_rank(parallel_context.tp_pg)
            * out_features_per_tp_rank : (dist.get_rank(parallel_context.tp_pg) + 1)
            * out_features_per_tp_rank
        ],
        atol=1e-6,
        rtol=1e-7,
    )

    old_grad = column_linear.weight.grad.clone()
    # Clip grads
    total_norm = clip_grad_norm(
        mp_pg=parallel_context.mp_pg,
        named_parameters=column_linear.named_parameters(),
        grad_accumulator=None,
        max_norm=1.0,
        norm_type=norm_type,
    )
    ref_total_norm = torch.nn.utils.clip_grad_norm_(reference_linear.parameters(), max_norm=1.0, norm_type=norm_type)

    # Check that the gradients have changed
    assert not torch.allclose(old_grad, column_linear.weight.grad), "Gradients should have changed after clipping"

    # Test that we get the same gradient after clipping
    torch.testing.assert_close(
        column_linear.weight.grad,
        reference_linear.weight.grad[
            dist.get_rank(parallel_context.tp_pg)
            * out_features_per_tp_rank : (dist.get_rank(parallel_context.tp_pg) + 1)
            * out_features_per_tp_rank
        ],
    )
    torch.testing.assert_close(
        column_linear.bias.grad,
        reference_linear.bias.grad[
            dist.get_rank(parallel_context.tp_pg)
            * out_features_per_tp_rank : (dist.get_rank(parallel_context.tp_pg) + 1)
            * out_features_per_tp_rank
        ],
    )
    torch.testing.assert_close(total_norm, ref_total_norm)

    parallel_context.destroy()


@pytest.mark.skipif(available_gpus() < 2, reason="test_clip_grads_tied_weights requires at least 2 gpus")
@pytest.mark.parametrize("norm_type", [math.inf, 1.0, 2.0])
@rerun_if_address_is_in_use()
def test_clip_grads_tied_weights(norm_type: float):
    init_distributed(tp=1, dp=1, pp=2)(_test_clip_grads_tied_weights)(norm_type=norm_type)


def _test_clip_grads_tied_weights(parallel_context: ParallelContext, norm_type: float):
    if dist.get_rank(parallel_context.pp_pg) == 0:
        model = nn.ModuleDict({"dense0": nn.Linear(10, 10, device="cuda")})
    else:
        model = nn.ModuleDict({"dense1": nn.Linear(10, 10, device="cuda")})

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

    # Sync tied weights: basic assumption
    initial_sync(model=model, parallel_context=parallel_context)

    # Check that weights are now synced
    assert_tensor_synced_across_pg(weight, group)
    assert_tensor_synced_across_pg(bias, group)

    # Compute gradient
    input_ = torch.randn(13, 10, device="cuda")
    if dist.get_rank(parallel_context.pp_pg) == 0:
        out = model.dense0(input_)
    else:
        out = model.dense1(input_)
    out.sum().backward()

    # sync gradients
    sync_tied_weights_gradients(model, parallel_context=parallel_context, grad_accumulator=None)

    # We check that we both gradients are synchronized
    assert_tensor_synced_across_pg(weight.grad, group)
    assert_tensor_synced_across_pg(bias.grad, group)

    # Save grads as reference
    ref_weight = weight.clone()
    ref_weight.grad = weight.grad.clone()
    ref_bias = bias.clone()
    ref_bias.grad = bias.grad.clone()

    old_grad = weight.grad.clone()
    # Clip grads
    total_norm = clip_grad_norm(
        mp_pg=parallel_context.mp_pg,
        named_parameters=model.named_parameters(),
        grad_accumulator=None,
        max_norm=1.0,
        norm_type=norm_type,
    )
    ref_total_norm = torch.nn.utils.clip_grad_norm_([ref_weight, ref_bias], max_norm=1.0, norm_type=norm_type)

    # Check that the gradients have changed
    assert not torch.allclose(old_grad, weight.grad), "Gradients should have changed after clipping"

    # Test that we get the same gradient after clipping
    assert torch.allclose(weight.grad, ref_weight.grad, rtol=1e-7, atol=1e-6)
    assert torch.allclose(bias.grad, ref_bias.grad, rtol=1e-7, atol=1e-6)
    assert torch.allclose(total_norm, ref_total_norm, rtol=0, atol=0), f"Got {total_norm} and {ref_total_norm}"

    parallel_context.destroy()


@pytest.mark.parametrize("half_precision", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("norm_type", [math.inf, 1.0, 2.0])
@rerun_if_address_is_in_use()
def test_clip_grads_fp32_accumulator(norm_type: float, half_precision: torch.dtype):
    init_distributed(tp=1, dp=1, pp=2)(_test_clip_grads_fp32_accumulator)(
        norm_type=norm_type, half_precision=half_precision
    )


def _test_clip_grads_fp32_accumulator(
    parallel_context: ParallelContext, norm_type: float, half_precision: torch.dtype
):
    device = torch.device("cuda")
    p2p = P2P(parallel_context.pp_pg, device=device)
    reference_rank = 0
    has_reference_model = dist.get_rank(parallel_context.pp_pg) == reference_rank
    pipeline_engine = AllForwardAllBackwardPipelineEngine()
    current_pp_rank = dist.get_rank(parallel_context.pp_pg)

    # spawn model
    model = DummyModel(p2p=p2p)
    if has_reference_model:
        reference_model = DummyModel(p2p=p2p).to(torch.float)

    # Set the ranks
    assert len(model.mlp) == parallel_context.pp_pg.size()
    with init_on_device_and_dtype(device):
        for pp_rank, non_linear in zip(range(parallel_context.pp_pg.size()), model.mlp):
            non_linear.linear.build_and_set_rank(pp_rank=pp_rank)
            non_linear.activation.build_and_set_rank(pp_rank=pp_rank)
        model.loss.build_and_set_rank(pp_rank=parallel_context.pp_pg.size() - 1)

        if has_reference_model:
            for non_linear in reference_model.mlp:
                non_linear.linear.build_and_set_rank(pp_rank=reference_rank)
                non_linear.activation.build_and_set_rank(pp_rank=reference_rank)
            reference_model.loss.build_and_set_rank(pp_rank=reference_rank)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            setattr(module, "weight", NanotronParameter(module.weight))
            setattr(module, "bias", NanotronParameter(module.bias))

    # model goes to half precision
    model = model.to(half_precision)

    # synchronize weights
    if has_reference_model:
        with torch.inference_mode():
            for pp_rank in range(parallel_context.pp_pg.size()):
                reference_non_linear = reference_model.mlp[pp_rank].linear.pp_block
                if pp_rank == current_pp_rank:
                    # We already have the weights locally
                    non_linear = model.mlp[pp_rank].linear.pp_block
                    reference_non_linear.weight.data.copy_(non_linear.weight.data)
                    reference_non_linear.bias.data.copy_(non_linear.bias.data)
                    continue

                weight, bias = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
                reference_non_linear.weight.data.copy_(weight.data)
                reference_non_linear.bias.data.copy_(bias.data)
    else:
        p2p.send_tensors(
            [model.mlp[current_pp_rank].linear.pp_block.weight, model.mlp[current_pp_rank].linear.pp_block.bias],
            to_rank=reference_rank,
        )

    # Add gradient accumulator
    grad_accumulator = FP32GradientAccumulator(model.named_parameters())

    # Check that our model is a valid model
    sanity_check(model)

    # Compute backward
    # Get infinite dummy data iterator
    data_iterator = dummy_infinite_data_loader(
        pp_pg=parallel_context.pp_pg, dtype=half_precision
    )  # First rank receives data

    n_micro_batches_per_batch = 5
    batch = [next(data_iterator) for _ in range(n_micro_batches_per_batch)]
    pipeline_engine.train_batch_iter(
        model,
        pg=parallel_context.pp_pg,
        batch=batch,
        nb_microbatches=n_micro_batches_per_batch,
        grad_accumulator=grad_accumulator,
    )

    # We're going to copy the model gradients to the reference model gradient
    # The reason why we do this, instead of computing backward using autograd is because of numerical precisions
    if has_reference_model:
        for pp_rank in range(parallel_context.pp_pg.size()):
            reference_non_linear = reference_model.mlp[pp_rank].linear.pp_block
            prefix_name = f"mlp.{pp_rank}.linear.pp_block"
            if pp_rank == current_pp_rank:
                # We already have the gradients locally
                reference_non_linear.weight.grad = grad_accumulator.get_grad_buffer(f"{prefix_name}.weight").clone()
                reference_non_linear.bias.grad = grad_accumulator.get_grad_buffer(f"{prefix_name}.bias").clone()
                continue

            weight_grad, bias_grad = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
            reference_non_linear.weight.grad = weight_grad
            reference_non_linear.bias.grad = bias_grad
    else:
        p2p.send_tensors(
            [
                grad_accumulator.get_grad_buffer(f"mlp.{current_pp_rank}.linear.pp_block.weight"),
                grad_accumulator.get_grad_buffer(f"mlp.{current_pp_rank}.linear.pp_block.bias"),
            ],
            to_rank=reference_rank,
        )

    old_fp32_grads = {
        name: grad_accumulator.get_grad_buffer(name=name).clone() for name, _ in model.named_parameters()
    }

    # Clip grads
    total_norm = clip_grad_norm(
        mp_pg=parallel_context.mp_pg,
        named_parameters=model.named_parameters(),
        grad_accumulator=grad_accumulator,
        max_norm=1.0,
        norm_type=norm_type,
    )
    if has_reference_model:
        ref_total_norm = torch.nn.utils.clip_grad_norm_(
            reference_model.parameters(), max_norm=1.0, norm_type=norm_type
        )

    # Check that the gradients have changed
    for name, _ in model.named_parameters():
        new_fp32_grad = grad_accumulator.get_grad_buffer(name=name)
        assert not torch.allclose(old_fp32_grads[name], new_fp32_grad), "Gradients should have changed after clipping"

    # We check that we get the same gradient accumulation. In theory we do get more precision by promoting gradients to fp32.
    if has_reference_model:
        torch.testing.assert_close(
            total_norm.view(1),
            ref_total_norm.view(1),
            atol=1e-6,
            rtol=1e-7,
            msg=lambda msg: f"Expected {total_norm} to match {ref_total_norm}.\n{msg}",
        )
        for pp_rank in range(parallel_context.pp_pg.size()):
            reference_non_linear = reference_model.mlp[pp_rank].linear.pp_block
            prefix_name = f"mlp.{pp_rank}.linear.pp_block"
            if pp_rank == current_pp_rank:
                # We already have the gradients locally
                torch.testing.assert_close(
                    reference_non_linear.weight.grad,
                    grad_accumulator.get_grad_buffer(f"{prefix_name}.weight"),
                    atol=1e-6,
                    rtol=1e-7,
                )
                torch.testing.assert_close(
                    reference_non_linear.bias.grad,
                    grad_accumulator.get_grad_buffer(f"{prefix_name}.bias"),
                    atol=1e-6,
                    rtol=1e-7,
                )
                continue

            weight_grad, bias_grad = p2p.recv_tensors(num_tensors=2, from_rank=pp_rank)
            torch.testing.assert_close(
                reference_non_linear.weight.grad,
                weight_grad,
                atol=1e-6,
                rtol=1e-7,
            )
            torch.testing.assert_close(
                reference_non_linear.bias.grad,
                bias_grad,
                atol=1e-6,
                rtol=1e-7,
            )
    else:
        p2p.send_tensors(
            [
                grad_accumulator.get_grad_buffer(f"mlp.{current_pp_rank}.linear.pp_block.weight"),
                grad_accumulator.get_grad_buffer(f"mlp.{current_pp_rank}.linear.pp_block.bias"),
            ],
            to_rank=reference_rank,
        )

    parallel_context.destroy()
