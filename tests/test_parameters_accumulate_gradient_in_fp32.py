import copy

import nanotron.distributed as dist
import pytest
import torch
from helpers.dummy import DummyModel, dummy_infinite_data_loader
from helpers.exception import assert_fail_except_rank_with, timeout_after
from helpers.utils import available_gpus, init_distributed, rerun_if_address_is_in_use
from nanotron.models import init_on_device_and_dtype
from nanotron.optim import ZeroDistributedOptimizer
from nanotron.optim.gradient_accumulator import FP32GradBucketManager, FP32GradientAccumulator, get_fp32_accum_hook
from nanotron.optim.named_optimizer import NamedOptimizer
from nanotron.optim.optimizer_from_gradient_accumulator import (
    OptimizerFromGradientAccumulator,
)
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter, sanity_check
from nanotron.parallel.pipeline_parallel.engine import (
    AllForwardAllBackwardPipelineEngine,
    OneForwardOneBackwardPipelineEngine,
    PipelineEngine,
)
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.pipeline_parallel.utils import get_pp_rank_of
from nanotron.parallel.tied_parameters import (
    get_tied_id_to_param,
    sync_tied_weights_gradients,
    tie_parameters,
)
from nanotron.parallel.utils import initial_sync
from nanotron.sanity_checks import assert_tensor_synced_across_pg
from nanotron.utils import ContextManagers
from torch import nn


@pytest.mark.parametrize("half_precision", [torch.float16, torch.bfloat16])
def test_gradient_promoting_in_fp32(half_precision: torch.dtype):
    model = nn.Linear(3, 2, bias=False, dtype=half_precision, device="cuda")

    # Create Nanotron Parameter
    model.weight = NanotronParameter(model.weight)

    # Add gradient accumulator
    accumulator = FP32GradientAccumulator(model.named_parameters())

    # Check that our model is a valid model
    sanity_check(model)

    # Compute backward
    input = torch.randn(5, 3, dtype=half_precision, device="cuda")
    accumulator.backward(model(input).sum())

    # Check that we have an high precision gradient and that the low precision one is cleared
    assert accumulator.parameters["weight"]["fp32"].grad.dtype == torch.float
    if model.weight.grad is not None:
        # We check that it's zero
        torch.testing.assert_close(model.weight.grad, torch.zeros_like(model.weight.grad), atol=1e-6, rtol=1e-7)


@pytest.mark.parametrize("half_precision", [torch.float16, torch.bfloat16])
def test_gradient_accumulated_in_fp32(half_precision: torch.dtype):
    model = nn.Linear(3, 2, bias=False, dtype=half_precision, device="cuda")
    ref_model = nn.Linear(3, 2, bias=False, dtype=half_precision, device="cuda")
    with torch.inference_mode():
        ref_model.weight.copy_(model.weight)

    # Create Nanotron Parameter
    model.weight = NanotronParameter(model.weight)

    # Add gradient accumulator
    accumulator = FP32GradientAccumulator(model.named_parameters())

    # Check that our model is a valid model
    sanity_check(model)

    # Compute backward
    grad_accumulation_steps = 2
    for _ in range(grad_accumulation_steps):
        # We want large input to have large gradients.
        input = (torch.randn(5, 3, dtype=half_precision, device="cuda") ** 2 + 1) * 100

        # Compute backwards
        accumulator.backward(model(input).sum())
        ref_model(input).sum().backward()

    # We check that we get the same gradient accumulation. In theory we do get more precision by promoting gradients to fp32.
    torch.testing.assert_close(
        accumulator.parameters["weight"]["fp32"].grad.to(half_precision),
        ref_model.weight.grad,
    )


@pytest.mark.parametrize("half_precision", [torch.float16, torch.bfloat16])
def test_optimizer_can_step_gradient_in_fp32(half_precision: torch.dtype):
    model = nn.Linear(3, 2, bias=False, dtype=half_precision, device="cuda")
    original_weight = model.weight.detach().clone()

    # Create Nanotron Parameter
    model.weight = NanotronParameter(model.weight)

    # Add optimizer
    optimizer = OptimizerFromGradientAccumulator(
        gradient_accumulator_builder=lambda named_params: FP32GradientAccumulator(named_parameters=named_params),
        named_params_or_groups=model.named_parameters(),
        optimizer_builder=lambda named_param_groups: NamedOptimizer(
            named_params_or_groups=named_param_groups,
            optimizer_builder=lambda param_groups: torch.optim.AdamW(param_groups),
        ),
    )
    accumulator = optimizer.gradient_accumulator

    # Check that our model is a valid model
    sanity_check(model)

    # Compute backward
    input = torch.randn(5, 3, dtype=half_precision, device="cuda")
    accumulator.backward(model(input).sum())

    # Check that we have an high precision gradient and that the low precision one is cleared
    assert accumulator.parameters["weight"]["fp32"].grad.dtype == torch.float
    if model.weight.grad is not None:
        # We check that it's zero
        torch.testing.assert_close(model.weight.grad, torch.zeros_like(model.weight.grad), atol=1e-6, rtol=1e-7)

    optimizer.step()
    optimizer.zero_grad()

    # Check that we don't have gradients anymore and that it's set to `None`
    assert accumulator.parameters["weight"]["fp32"].grad is None
    assert model.weight.grad is None

    # Check that gradients have been set to zero
    fp32_grad = accumulator.get_grad_buffer(name="weight")
    torch.testing.assert_close(fp32_grad, torch.zeros_like(fp32_grad), atol=1e-6, rtol=1e-7)

    # weights has been updates
    assert not torch.allclose(original_weight, model.weight)


@pytest.mark.skipif(available_gpus() < 2, reason="Testing ddp_hook_allreduce requires at least 2 gpus")
@pytest.mark.parametrize("half_precision", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("accumulation_steps", [1, 10])
@pytest.mark.parametrize("train_iterations", [1, 3])
@rerun_if_address_is_in_use()
def test_ddp_with_grad_accum_in_fp32(half_precision: torch.dtype, accumulation_steps: int, train_iterations: int):
    init_distributed(tp=1, dp=2, pp=1)(_test_ddp_with_grad_accum_in_fp32)(
        half_precision=half_precision,
        accumulation_steps=accumulation_steps,
        train_iterations=train_iterations,
    )


def _test_ddp_with_grad_accum_in_fp32(
    parallel_context: ParallelContext,
    half_precision: torch.dtype,
    accumulation_steps: int,
    train_iterations: int,
):
    hidden_size = 32
    n_layers = 3
    model = nn.Sequential(
        nn.Linear(3, hidden_size, bias=False, dtype=half_precision, device="cuda"),
        *(
            nn.Linear(hidden_size, hidden_size, bias=False, dtype=half_precision, device="cuda")
            for _ in range(n_layers - 1)
        ),
    )
    model_hook = copy.deepcopy(model)

    # Create Nanotron Parameters
    for module in model.modules():
        if isinstance(module, nn.Linear):
            setattr(module, "weight", NanotronParameter(module.weight))
    for module in model_hook.modules():
        if isinstance(module, nn.Linear):
            setattr(module, "weight", NanotronParameter(module.weight))

    # Needed in order to obtain smaller gradient buckets when using `DistributedDataParallel`
    model_ddp = torch.nn.parallel.DistributedDataParallel(
        model,
        process_group=parallel_context.dp_pg,
    )  # we won't actually use DDP anywhere, it's just to have same module names
    model_ddp_accum_ref = {}
    model_ddp_fp32_accum = torch.nn.parallel.DistributedDataParallel(
        model_hook,
        process_group=parallel_context.dp_pg,
    )

    # Add gradient accumulator
    accumulator = FP32GradientAccumulator(model_ddp_fp32_accum.named_parameters())

    # Register DDP hook
    state = FP32GradBucketManager(
        dp_pg=parallel_context.dp_pg,
        accumulator=accumulator,
        param_id_to_name={id(param): name for name, param in model_ddp_fp32_accum.named_parameters()},
    )
    model_ddp_fp32_accum.register_comm_hook(
        state=state,
        hook=get_fp32_accum_hook(
            reduce_scatter=False,
            reduce_op=dist.ReduceOp.AVG,
        ),
    )

    for train_iter in range(train_iterations):
        # Gradient accumulation steps
        for accum_step in range(accumulation_steps - 1):
            # Forward-Backward
            input = torch.randn(10, 3, dtype=half_precision, device="cuda")
            loss = model_ddp.module(input).sum()
            assert not torch.isinf(loss).any(), "loss is inf"
            loss.backward()
            with ContextManagers([model_ddp_fp32_accum.no_sync(), accumulator.no_sync()]):
                loss_fp32_accum = model_ddp_fp32_accum(input).sum()
                accumulator.backward(loss_fp32_accum)

            for name, param in model_ddp.named_parameters():
                grad = param.grad
                grad_fp32_accum = accumulator.parameters[name]["fp32"].grad
                fp32_grad_bucket = accumulator.get_grad_buffer(name=name)

                # Check that FP32GradAccum+DDP+hook gives close gradients to DDP
                model_ddp_accum_ref[name] = (
                    grad.float() if accum_step == 0 else model_ddp_accum_ref[name] + grad.float()
                )

                dist.barrier()
                torch.testing.assert_close(model_ddp_accum_ref[name], fp32_grad_bucket, atol=1e-6, rtol=1e-7)

                dist.barrier()
                # Check that we correctly copied grads from buckets to params (`copy_buckets_to_grads`)
                torch.testing.assert_close(fp32_grad_bucket, grad_fp32_accum, atol=1e-6, rtol=1e-7)

                # Check that the gradients are not synchronized across DP
                with assert_fail_except_rank_with(AssertionError, rank_exception=0, pg=parallel_context.dp_pg):
                    assert_tensor_synced_across_pg(grad, parallel_context.dp_pg)
                with assert_fail_except_rank_with(AssertionError, rank_exception=0, pg=parallel_context.dp_pg):
                    assert_tensor_synced_across_pg(fp32_grad_bucket, parallel_context.dp_pg)

            # We zero out half grads for `model_ddp` because we're accumulating grads manually in `model_ddp_accum_ref`
            model_ddp.zero_grad()

        # Last accumulation step (Sync grads across DDP)
        input = torch.randn(10, 3, dtype=half_precision, device="cuda")
        loss = model_ddp.module(input).sum()
        loss.backward()
        # manually reduce grads across DDP
        for name, param in model_ddp.named_parameters():
            grad = param.grad
            model_ddp_accum_ref[name] = (
                model_ddp_accum_ref[name] + grad.float() if name in model_ddp_accum_ref else grad.float()
            )
            dist.all_reduce(model_ddp_accum_ref[name], group=parallel_context.dp_pg, op=dist.ReduceOp.AVG)

        loss_fp32_accum = model_ddp_fp32_accum(input).sum()
        accumulator.backward(loss_fp32_accum)

        for name, param in model_ddp_fp32_accum.named_parameters():
            # Check that half grads has been set to None in sync step, to avoid it being uncorrectly used
            half_grad = param.grad
            assert half_grad is None, f"{half_grad} != None"

            grad = model_ddp_accum_ref[name]
            grad_fp32_accum = accumulator.parameters[name]["fp32"].grad
            fp32_grad_bucket = accumulator.get_grad_buffer(name=name)

            # Check that FP32GradAccum+DDP+hook gives close gradients to DDP
            dist.barrier()
            torch.testing.assert_close(grad, fp32_grad_bucket, atol=1e-6, rtol=1e-7)

            # Check that grad points to the same memory as the bucket
            assert grad_fp32_accum.data_ptr() == fp32_grad_bucket.data_ptr()

            # Check that the gradients are synchronized across DP
            assert_tensor_synced_across_pg(grad, parallel_context.dp_pg)
            assert_tensor_synced_across_pg(grad_fp32_accum, parallel_context.dp_pg)

        # Zero out gradients (Usually it's the optimizer that does this)
        model_ddp.zero_grad()
        model_ddp_accum_ref = {}
        accumulator.zero_grad()  # Sets half grads to None and zeroes out fp32 grad buckets
        for name, elt in accumulator.parameters.items():
            fp32_param = elt["fp32"]
            fp32_param.grad = None

        # Check that fp32 grad buckets are zeroed out and `param.grad` is set to None
        for name, param in model_ddp_fp32_accum.named_parameters():
            assert param.grad is None
            fp32_grad_bucket = accumulator.get_grad_buffer(name=name)
            dist.barrier()
            torch.testing.assert_close(fp32_grad_bucket, torch.zeros_like(fp32_grad_bucket), atol=1e-6, rtol=1e-7)

        # Check that all fp32 grad buckets are zeroed out
        for _, elt in accumulator.fp32_grad_buffers.items():
            fp32_grad = elt["fp32_grad"]
            # This is important as we assume grad buckets to be zeroed out at the first accumulation step
            dist.barrier()
            torch.testing.assert_close(fp32_grad, torch.zeros_like(fp32_grad), atol=1e-6, rtol=1e-7)

    parallel_context.destroy()


@pytest.mark.skipif(
    available_gpus() < 4, reason="Testing test_tied_weights_sync_with_grad_accum_in_fp32 requires at least 4 gpus"
)
@pytest.mark.parametrize(
    "pipeline_engine", [AllForwardAllBackwardPipelineEngine(), OneForwardOneBackwardPipelineEngine()]
)
@pytest.mark.parametrize("reduce_scatter", [True, False])
@rerun_if_address_is_in_use()
def test_tied_weights_sync_with_grad_accum_in_fp32(pipeline_engine: PipelineEngine, reduce_scatter: bool):
    init_distributed(tp=1, dp=2, pp=2)(_test_tied_weights_sync_with_grad_accum_in_fp32)(
        pipeline_engine=pipeline_engine, reduce_scatter=reduce_scatter
    )


def _test_tied_weights_sync_with_grad_accum_in_fp32(
    parallel_context: ParallelContext, pipeline_engine: PipelineEngine, reduce_scatter: bool
):
    # We init two replicas of 2 denses. Each dense is on a device.
    dtype = torch.float16
    device = torch.device("cuda")
    p2p = P2P(pg=parallel_context.pp_pg, device=device)

    model = DummyModel(p2p=p2p)
    reference_model = DummyModel(p2p=p2p)
    reference_model_accum_ref = {}

    for mdl in [model, reference_model]:
        # Set the ranks
        with init_on_device_and_dtype(device, dtype):
            assert parallel_context.pp_pg.size() == len(mdl.mlp)
            for pp_rank, non_linear in zip(range(parallel_context.pp_pg.size()), mdl.mlp):
                non_linear.linear.build_and_set_rank(pp_rank=pp_rank)
                non_linear.activation.build_and_set_rank(pp_rank=pp_rank)
            mdl.loss.build_and_set_rank(pp_rank=parallel_context.pp_pg.size() - 1)

        # Tie all dense weights across PP
        tie_parameters(
            root_module=mdl,
            ties=[
                (
                    target,
                    (
                        parallel_context.world_rank_matrix[
                            dist.get_rank(parallel_context.expert_pg),
                            get_pp_rank_of(target, module=mdl),
                            dist.get_rank(parallel_context.dp_pg),
                            dist.get_rank(parallel_context.tp_pg),
                        ],
                    ),
                )
                for target in [
                    f"mlp.{pp_rank}.linear.pp_block.weight" for pp_rank in range(parallel_context.pp_pg.size())
                ]
            ],
            parallel_context=parallel_context,
            reduce_op=dist.ReduceOp.SUM,
        )

        for name, module in mdl.named_modules():
            if isinstance(module, nn.Linear):
                module.bias = NanotronParameter(module.bias)

        # Sync DP and tied weights: basic assumption
        initial_sync(model=mdl, parallel_context=parallel_context)

    # Sync params between `model` and `reference_model`
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(reference_model.get_parameter(name))

    # DDP
    model_ddp = torch.nn.parallel.DistributedDataParallel(model, process_group=parallel_context.dp_pg)
    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
    reference_module_id_to_prefix = {
        id(module): f"{module_name}." for module_name, module in reference_model.named_modules()
    }
    # Fix the root_model
    module_id_to_prefix[id(model)] = ""
    reference_module_id_to_prefix[id(reference_model)] = ""

    # named parameters
    named_parameters = [
        (
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            if param.is_tied
            else name,
            param,
        )
        for name, param in model.named_parameters()
    ]

    # Optimizer: We don't actually run the optimizer, we just use it to build the gradient accumulator
    optimizer = ZeroDistributedOptimizer(
        dp_pg=parallel_context.dp_pg,
        named_params_or_groups=named_parameters,
        optimizer_builder=lambda named_param_groups_1: OptimizerFromGradientAccumulator(
            gradient_accumulator_builder=lambda named_params: FP32GradientAccumulator(
                named_parameters=named_params,
                grad_buckets_named_params=named_parameters,
            ),
            named_params_or_groups=named_param_groups_1,
            optimizer_builder=lambda named_param_groups_2: NamedOptimizer(
                named_params_or_groups=named_param_groups_2,
                optimizer_builder=lambda param_groups: torch.optim.AdamW(param_groups),
            ),
        ),
    )
    param_id_to_name = {
        id(param): param.get_tied_info().get_full_name_from_module_id_to_prefix(
            module_id_to_prefix=module_id_to_prefix
        )
        if param.is_tied
        else name
        for name, param in model.named_parameters()
    }

    # Add gradient accumulator
    # We use `model_ddp.module` in order ta have the parameter names without the `module.` prefix
    accumulator = optimizer.optimizer.gradient_accumulator
    accumulator.assign_param_offsets(
        dp_rank=dist.get_rank(parallel_context.dp_pg),
        param_name_to_offsets=optimizer.param_name_to_dp_rank_offsets,
    )
    model_ddp.register_comm_hook(
        state=FP32GradBucketManager(
            dp_pg=parallel_context.dp_pg,
            accumulator=accumulator,
            param_id_to_name=param_id_to_name,
        ),
        hook=get_fp32_accum_hook(reduce_scatter=reduce_scatter, reduce_op=dist.ReduceOp.AVG),
    )

    # Get infinite dummy data iterator
    data_iterator = dummy_infinite_data_loader(pp_pg=parallel_context.pp_pg, dtype=dtype)  # First rank receives data

    n_micro_batches_per_batch = 2
    batch = [next(data_iterator) for _ in range(n_micro_batches_per_batch)]

    ## Reference model iteration step
    def forward_backward_reference(mdl, micro_batch):
        pipeline_engine.train_batch_iter(
            mdl, pg=parallel_context.pp_pg, batch=[micro_batch], nb_microbatches=1, grad_accumulator=None
        )

    for accum_step in range(n_micro_batches_per_batch - 1):
        # Forward-Backward
        forward_backward_reference(reference_model, batch[accum_step])
        # Accumulate grads
        for name, param in reference_model.named_parameters():
            grad = param.grad
            if param.is_tied:
                tied_info = param.get_tied_info()
                name = tied_info.get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=reference_module_id_to_prefix
                )
            reference_model_accum_ref[name] = (
                grad.float() if accum_step == 0 else reference_model_accum_ref[name] + grad.float()
            )

        # We zero out half grads for `reference_model` because we're accumulating grads manually in `reference_model_accum_ref`
        reference_model.zero_grad()

    # Last accumulation step (Sync grads across DDP)
    forward_backward_reference(reference_model, batch[-1])
    # manually reduce grads across DDP
    for name, param in reference_model.named_parameters():
        grad = param.grad
        if param.is_tied:
            tied_info = param.get_tied_info()
            name = tied_info.get_full_name_from_module_id_to_prefix(module_id_to_prefix=reference_module_id_to_prefix)
        reference_model_accum_ref[name] = (
            reference_model_accum_ref[name] + grad.float() if name in reference_model_accum_ref else grad.float()
        )
        dist.all_reduce(reference_model_accum_ref[name], group=parallel_context.dp_pg, op=dist.ReduceOp.AVG)

    ## Model iteration step
    pipeline_engine.train_batch_iter(
        model_ddp,
        pg=parallel_context.pp_pg,
        batch=batch,
        nb_microbatches=n_micro_batches_per_batch,
        grad_accumulator=accumulator,
    )
    for name, param in model_ddp.module.named_parameters():
        if param.is_tied:
            tied_info = param.get_tied_info()
            name = tied_info.get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)

        # Each parameter is sharded across DP.
        assert (
            name in accumulator.parameters
        ), f"`accumulator.parameters` must have all params {name} not in `accumulator.parameters`. Existing keys are: {accumulator.parameters}"

        fp32_grad = accumulator.get_grad_buffer(name=name)

        if not reduce_scatter:
            # Check that the gradients are synchronized across DP
            assert_tensor_synced_across_pg(fp32_grad, parallel_context.dp_pg)

        fp32_grad_ref = reference_model_accum_ref[name]
        dist.barrier()

        if reduce_scatter:
            slice_ = slice(*accumulator.param_name_to_offsets[name])
            # Check that gradients are correct
            torch.testing.assert_close(
                fp32_grad_ref.view(-1)[slice_] / n_micro_batches_per_batch,
                fp32_grad.view(-1)[slice_],
                rtol=1e-7,
                atol=1e-6,
                msg=lambda msg: f"FP32 Gradients at `{name}` don't match\n - Expected: {fp32_grad_ref.view(-1)[slice_] / n_micro_batches_per_batch}\n - Got: {fp32_grad.view(-1)[slice_]}",
            )
        else:
            # Check that gradients are correct
            torch.testing.assert_close(fp32_grad_ref / n_micro_batches_per_batch, fp32_grad, rtol=1e-7, atol=1e-6)

    # Check that tied weights grads are not synchronized yet
    for (name, group_ranks), param in sorted(
        get_tied_id_to_param(parameters=model_ddp.parameters(), root_module=model_ddp.module).items(),
        key=lambda x: x[0],
    ):
        if not (isinstance(param, NanotronParameter) and param.is_tied):
            continue

        group = parallel_context.world_ranks_to_pg[group_ranks]
        fp32_grad = accumulator.get_grad_buffer(name=name)

        with assert_fail_except_rank_with(AssertionError, rank_exception=0, pg=group):
            assert_tensor_synced_across_pg(
                tensor=fp32_grad,
                pg=group,
                msg=lambda err: f"Tied weights's grads {name} are not synchronized. {err}",
            )
    # Sync tied weights grads (e.g. sync dense1 and dense2 grads in DP=0, but the problem is that DP=0 has only optim states for dense1)
    # - Translate tied ranks along DP axis to find the DP rank that has the tied weights
    # - accumulator keeps grads for all DPs, so we can just sync the grads
    with timeout_after():
        sync_tied_weights_gradients(
            module=model_ddp.module, parallel_context=parallel_context, grad_accumulator=accumulator
        )

    tied_infos_dict = {
        (
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix),
            param.get_tied_info().global_ranks,
            param.get_tied_info().reduce_op,
        ): param
        for name, param in model_ddp.module.named_parameters()
        if param.is_tied
    }

    # Check that tied weights grads are synchronized
    for (name, group_ranks, reduce_op), param in sorted(tied_infos_dict.items(), key=lambda x: x[0]):
        # Make sure we don't get None for reduce_op
        assert reduce_op == dist.ReduceOp.SUM

        fp32_grad_buffer = accumulator.get_grad_buffer(name=name)

        # Grad buffers are only attached to param.grad on ranks that are sharded depending on `param_to_dprank`
        fp32_grad = accumulator.parameters[name]["fp32"].grad
        # Tied weights are synced using the fp32 grad buffers. Let's make sure they still point to the same memory
        # When using ZeRODistributedOptimizer gradients are slices across dp
        dp_slice_fp_32_grad_buffer = fp32_grad_buffer.view(-1)[slice(*accumulator.param_name_to_offsets[name])]
        assert (
            dp_slice_fp_32_grad_buffer.data_ptr() == fp32_grad.data_ptr()
        ), "dp_slice_fp_32_grad_buffer and fp32_grad should point to the same memory"

        group = parallel_context.world_ranks_to_pg[group_ranks]

        # Check that fp32 grads for tied weights are synced (Used in optimizer step)
        # Since we use `reduce_scatter = False` the entire gradient buffer is all reduced, causing it to be synced
        if reduce_scatter:
            assert_tensor_synced_across_pg(
                tensor=dp_slice_fp_32_grad_buffer,
                pg=group,
                msg=lambda err: f"Tied weights's fp32 grads {name} are not synchronized. {err}",
            )
        else:
            assert_tensor_synced_across_pg(
                tensor=fp32_grad_buffer,
                pg=group,
                msg=lambda err: f"Tied weights's fp32 grads {name} are not synchronized. {err}",
            )

        # Manually sync reference model's tied weights grads
        dist.all_reduce(reference_model_accum_ref[name], group=group, op=reduce_op)

    # Check that accumulated grads are correct
    for name, elt in accumulator.fp32_grad_buffers.items():
        fp32_grad = elt["fp32_grad"]

        dist.barrier()
        if reduce_scatter:
            slice_ = slice(*accumulator.param_name_to_offsets[name])
            torch.testing.assert_close(
                reference_model_accum_ref[name].view(-1)[slice_] / n_micro_batches_per_batch,
                fp32_grad.view(-1)[slice_],
                atol=1e-6,
                rtol=1e-7,
                msg=lambda msg: f"Grad for {name} is not correct.\n{msg}",
            )
        else:
            torch.testing.assert_close(
                reference_model_accum_ref[name] / n_micro_batches_per_batch,
                fp32_grad,
                atol=1e-6,
                rtol=1e-7,
                msg=lambda msg: f"Grad for {name} is not correct.\n{msg}",
            )

    parallel_context.destroy()
