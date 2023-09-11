import copy

import pytest
import torch
from helpers.dummy import DummyModel, dummy_infinite_data_loader
from helpers.exception import assert_fail_except_rank_with, timeout_after
from helpers.utils import available_gpus, init_distributed
from torch import nn

import brrr.core.distributed as dist
from brrr.core.dataclass import DistributedProcessGroups
from brrr.core.gradient_accumulator import FP32GradBucketManager, FP32GradientAccumulator, get_fp32_accum_hook
from brrr.core.optimizer import ZeroDistributedOptimizer
from brrr.core.optimizer.named_optimizer import NamedOptimizer
from brrr.core.optimizer.optimizer_from_gradient_accumulator import (
    OptimizerFromGradientAccumulator,
)
from brrr.core.parallelism.model import initial_sync
from brrr.core.parallelism.parameters import BRRRParameter, sanity_check
from brrr.core.parallelism.pipeline_parallelism.engine import (
    AllForwardAllBackwardPipelineEngine,
    OneForwardOneBackwardPipelineEngine,
    PipelineEngine,
)
from brrr.core.parallelism.pipeline_parallelism.p2p import P2P
from brrr.core.parallelism.pipeline_parallelism.utils import get_pp_rank_of
from brrr.core.parallelism.tied_parameters import (
    get_tied_id_to_param,
    sync_tied_weights_gradients,
    tie_parameters,
)
from brrr.core.utils import ContextManagers, assert_tensor_synced_across_pg, init_on_device_and_dtype


@pytest.mark.parametrize("half_precision", [torch.float16, torch.bfloat16])
def test_gradient_promoting_in_fp32(half_precision: torch.dtype):
    model = nn.Linear(3, 2, bias=False, dtype=half_precision, device="cuda")

    # Create BRRR Parameter
    model.weight = BRRRParameter(model.weight)

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
        torch.testing.assert_close(model.weight.grad, torch.zeros_like(model.weight.grad), atol=0, rtol=0)


@pytest.mark.parametrize("half_precision", [torch.float16, torch.bfloat16])
def test_gradient_accumulated_in_fp32(half_precision: torch.dtype):
    model = nn.Linear(3, 2, bias=False, dtype=half_precision, device="cuda")
    ref_model = nn.Linear(3, 2, bias=False, dtype=half_precision, device="cuda")
    with torch.inference_mode():
        ref_model.weight.copy_(model.weight)

    # Create BRRR Parameter
    model.weight = BRRRParameter(model.weight)

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

    # Create BRRR Parameter
    model.weight = BRRRParameter(model.weight)

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
        torch.testing.assert_close(model.weight.grad, torch.zeros_like(model.weight.grad), atol=0, rtol=0)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Check that we don't have gradients anymore and that it's set to `None`
    assert accumulator.parameters["weight"]["fp32"].grad is None
    assert model.weight.grad is None

    # Check that gradients have been set to zero
    fp32_grad = accumulator.get_grad_buffer(name="weight")
    torch.testing.assert_close(fp32_grad, torch.zeros_like(fp32_grad), atol=0, rtol=0)

    # weights has been updates
    assert not torch.allclose(original_weight, model.weight)


@pytest.mark.skipif(available_gpus() < 2, reason="Testing ddp_hook_allreduce requires at least 2 gpus")
@pytest.mark.parametrize("half_precision", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("accumulation_steps", [1, 10])
@pytest.mark.parametrize("train_iterations", [1, 3])
def test_ddp_with_grad_accum_in_fp32(half_precision: torch.dtype, accumulation_steps: int, train_iterations: int):
    init_distributed(tp=1, dp=2, pp=1)(_test_ddp_with_grad_accum_in_fp32)(
        half_precision=half_precision,
        accumulation_steps=accumulation_steps,
        train_iterations=train_iterations,
    )


def _test_ddp_with_grad_accum_in_fp32(
    dpg: DistributedProcessGroups,
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

    # Create BRRR Parameters
    for module in model.modules():
        if isinstance(module, nn.Linear):
            setattr(module, "weight", BRRRParameter(module.weight))
    for module in model_hook.modules():
        if isinstance(module, nn.Linear):
            setattr(module, "weight", BRRRParameter(module.weight))

    # Needed in order to obtain smaller gradient buckets when using `DistributedDataParallel`
    model_ddp = torch.nn.parallel.DistributedDataParallel(
        model,
        process_group=dpg.dp_pg,
    )  # we won't actually use DDP anywhere, it's just to have same module names
    model_ddp_accum_ref = {}
    model_ddp_fp32_accum = torch.nn.parallel.DistributedDataParallel(
        model_hook,
        process_group=dpg.dp_pg,
    )

    # Add gradient accumulator
    accumulator = FP32GradientAccumulator(model_ddp_fp32_accum.named_parameters())

    # Register DDP hook
    state = FP32GradBucketManager(
        dp_pg=dpg.dp_pg,
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
                torch.testing.assert_close(model_ddp_accum_ref[name], fp32_grad_bucket, atol=0, rtol=0)

                # Check that we correctly copied grads from buckets to params (`copy_buckets_to_grads`)
                torch.testing.assert_close(fp32_grad_bucket, grad_fp32_accum, atol=0, rtol=0)

                # Check that the gradients are not synchronized across DP
                with assert_fail_except_rank_with(AssertionError, rank_exception=0, pg=dpg.dp_pg):
                    assert_tensor_synced_across_pg(grad, dpg.dp_pg)
                with assert_fail_except_rank_with(AssertionError, rank_exception=0, pg=dpg.dp_pg):
                    assert_tensor_synced_across_pg(fp32_grad_bucket, dpg.dp_pg)

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
            dist.all_reduce(model_ddp_accum_ref[name], group=dpg.dp_pg, op=dist.ReduceOp.AVG)

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
            torch.testing.assert_close(grad, fp32_grad_bucket, atol=0, rtol=0)

            # Check that grad points to the same memory as the bucket
            assert grad_fp32_accum.data_ptr() == fp32_grad_bucket.data_ptr()

            # Check that the gradients are synchronized across DP
            assert_tensor_synced_across_pg(grad, dpg.dp_pg)
            assert_tensor_synced_across_pg(grad_fp32_accum, dpg.dp_pg)

        # Zero out gradients (Usually it's the optimizer that does this)
        model_ddp.zero_grad(set_to_none=True)
        model_ddp_accum_ref = {}
        accumulator.zero_grad(set_to_none=True)  # Sets half grads to None and zeroes out fp32 grad buckets
        for name, elt in accumulator.parameters.items():
            fp32_param = elt["fp32"]
            fp32_param.grad = None

        # Check that fp32 grad buckets are zeroed out and `param.grad` is set to None
        for name, param in model_ddp_fp32_accum.named_parameters():
            assert param.grad is None
            fp32_grad_bucket = accumulator.get_grad_buffer(name=name)
            torch.testing.assert_close(fp32_grad_bucket, torch.zeros_like(fp32_grad_bucket), atol=0, rtol=0)

        # Check that all fp32 grad buckets are zeroed out
        for _, elt in accumulator.fp32_grad_buffers.items():
            fp32_grad = elt["fp32_grad"]
            # This is important as we assume grad buckets to be zeroed out at the first accumulation step
            torch.testing.assert_close(fp32_grad, torch.zeros_like(fp32_grad), atol=0, rtol=0)


@pytest.mark.skipif(
    available_gpus() < 4, reason="Testing test_tied_weights_sync_with_grad_accum_in_fp32 requires at least 4 gpus"
)
@pytest.mark.parametrize(
    "pipeline_engine", [AllForwardAllBackwardPipelineEngine(), OneForwardOneBackwardPipelineEngine()]
)
@pytest.mark.parametrize("reduce_scatter", [True, False])
def test_tied_weights_sync_with_grad_accum_in_fp32(pipeline_engine: PipelineEngine, reduce_scatter: bool):
    init_distributed(tp=1, dp=2, pp=2)(_test_tied_weights_sync_with_grad_accum_in_fp32)(
        pipeline_engine=pipeline_engine, reduce_scatter=reduce_scatter
    )


def _test_tied_weights_sync_with_grad_accum_in_fp32(
    dpg: DistributedProcessGroups, pipeline_engine: PipelineEngine, reduce_scatter: bool
):
    # We init two replicas of 2 denses. Each dense is on a device.
    dtype = torch.float16
    device = torch.device("cuda")
    p2p = P2P(pg=dpg.pp_pg, device=device)

    model = DummyModel(p2p=p2p)
    reference_model = DummyModel(p2p=p2p)
    reference_model_accum_ref = {}

    for mdl in [model, reference_model]:
        # Set the ranks
        with init_on_device_and_dtype(device, dtype):
            assert dpg.pp_pg.size() == len(mdl.mlp)
            for pp_rank, non_linear in zip(range(dpg.pp_pg.size()), mdl.mlp):
                non_linear.linear.build_and_set_rank(pp_rank=pp_rank)
                non_linear.activation.build_and_set_rank(pp_rank=pp_rank)
            mdl.loss.build_and_set_rank(pp_rank=dpg.pp_pg.size() - 1)

        # Tie all dense weights across PP
        tie_parameters(
            root_module=mdl,
            ties=[
                (
                    target,
                    (
                        dpg.world_rank_matrix[
                            get_pp_rank_of(target, module=mdl), dist.get_rank(dpg.dp_pg), dist.get_rank(dpg.tp_pg)
                        ],
                    ),
                )
                for target in [f"mlp.{pp_rank}.linear.pp_block.weight" for pp_rank in range(dpg.pp_pg.size())]
            ],
            dpg=dpg,
            reduce_op=dist.ReduceOp.SUM,
        )

        for name, module in mdl.named_modules():
            if isinstance(module, nn.Linear):
                module.bias = BRRRParameter(module.bias)

        # Sync DP and tied weights: basic assumption
        initial_sync(model=mdl, dpg=dpg)

    # Sync params between `model` and `reference_model`
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(reference_model.get_parameter(name))

    # DDP
    model_ddp = torch.nn.parallel.DistributedDataParallel(model, process_group=dpg.dp_pg)
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
        dp_pg=dpg.dp_pg,
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
        dp_rank=dist.get_rank(dpg.dp_pg),
        param_name_to_offsets=optimizer.param_name_to_dp_rank_offsets,
    )
    model_ddp.register_comm_hook(
        state=FP32GradBucketManager(
            dp_pg=dpg.dp_pg,
            accumulator=accumulator,
            param_id_to_name=param_id_to_name,
        ),
        hook=get_fp32_accum_hook(reduce_scatter=reduce_scatter, reduce_op=dist.ReduceOp.AVG),
    )

    # Get infinite dummy data iterator
    data_iterator = dummy_infinite_data_loader(pp_pg=dpg.pp_pg, dtype=dtype)  # First rank receives data

    n_micro_batches_per_batch = 2
    batch = [next(data_iterator) for _ in range(n_micro_batches_per_batch)]

    ## Reference model iteration step
    def forward_backward_reference(mdl, micro_batch):
        pipeline_engine.train_batch_iter(mdl, pg=dpg.pp_pg, batch=[micro_batch], grad_accumulator=None)

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
        dist.all_reduce(reference_model_accum_ref[name], group=dpg.dp_pg, op=dist.ReduceOp.AVG)

    ## Model iteration step
    pipeline_engine.train_batch_iter(model_ddp, pg=dpg.pp_pg, batch=batch, grad_accumulator=accumulator)

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
            assert_tensor_synced_across_pg(fp32_grad, dpg.dp_pg)

        fp32_grad_ref = reference_model_accum_ref[name]
        if reduce_scatter:
            slice_ = slice(*accumulator.param_name_to_offsets[name])
            # Check that gradients are correct
            torch.testing.assert_close(
                fp32_grad_ref.view(-1)[slice_],
                fp32_grad.view(-1)[slice_],
                rtol=0,
                atol=0,
                msg=lambda msg: f"FP32 Gradients at `{name}` don't match\n - Expected: {fp32_grad_ref.view(-1)[slice_]}\n - Got: {fp32_grad.view(-1)[slice_]}",
            )
        else:
            # Check that gradients are correct
            torch.testing.assert_close(fp32_grad_ref, fp32_grad, rtol=0, atol=0)

    # Check that tied weights grads are not synchronized yet
    for (name, group_ranks), param in sorted(
        get_tied_id_to_param(parameters=model_ddp.parameters(), root_module=model_ddp.module).items(),
        key=lambda x: x[0],
    ):
        if not (isinstance(param, BRRRParameter) and param.is_tied):
            continue

        group = dpg.world_ranks_to_pg[group_ranks]
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
        sync_tied_weights_gradients(module=model_ddp.module, dpg=dpg, grad_accumulator=accumulator)

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

        group = dpg.world_ranks_to_pg[group_ranks]

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

        if reduce_scatter:
            slice_ = slice(*accumulator.param_name_to_offsets[name])
            torch.testing.assert_close(
                reference_model_accum_ref[name].view(-1)[slice_],
                fp32_grad.view(-1)[slice_],
                atol=0,
                rtol=0,
                msg=lambda msg: f"Grad for {name} is not correct.\n{msg}",
            )
        else:
            torch.testing.assert_close(
                reference_model_accum_ref[name],
                fp32_grad,
                atol=0,
                rtol=0,
                msg=lambda msg: f"Grad for {name} is not correct.\n{msg}",
            )
