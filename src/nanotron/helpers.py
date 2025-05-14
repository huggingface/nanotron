import contextlib
import csv
import gc
import math
import os
import re
import time
from datetime import datetime
from functools import partial
from math import ceil
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import Config, DatasetStageArgs, LRSchedulerArgs, OptimizerArgs, ParallelismArgs
from nanotron.distributed import ProcessGroup
from nanotron.logging import LogItem, human_format, log_rank
from nanotron.models.base import NanotronModel
from nanotron.optim.base import BaseOptimizer, Optimizer
from nanotron.optim.gradient_accumulator import (
    FP32GradBucketManager,
    FP32GradientAccumulator,
    GradientAccumulator,
    get_fp32_accum_hook,
)
from nanotron.optim.named_optimizer import NamedOptimizer
from nanotron.optim.optimizer_from_gradient_accumulator import (
    OptimizerFromGradientAccumulator,
)
from nanotron.optim.zero import ZeroDistributedOptimizer
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode
from nanotron.random import (
    RandomStates,
    get_current_random_state,
    get_synced_random_state,
)
from nanotron.scaling.parametrization import LearningRateForSP, LearningRateForSpectralMup, ParametrizationMethod
from nanotron.serialize import DataStageMetadata
from nanotron.serialize.metadata import TrainingMetadata

logger = logging.get_logger(__name__)


def _vocab_size_with_padding(orig_vocab_size: int, pg_size: int, make_vocab_size_divisible_by: int):
    """Pad vocab size so it is divisible by pg_size * make_vocab_size_divisible_by."""

    multiple = make_vocab_size_divisible_by * pg_size
    after = int(ceil(orig_vocab_size / multiple) * multiple)
    if after != orig_vocab_size:
        log_rank(
            f"[Vocab Size Padding] Padded vocab (size: {orig_vocab_size}) with {after - orig_vocab_size} dummy tokens (new size: {after})",
            logger=logger,
            level=logging.WARNING,
            rank=0,
        )
    return after


def init_random_states(parallel_config: ParallelismArgs, tp_pg: ProcessGroup):
    # Get synchronized random states
    if parallel_config is None or parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        random_states = RandomStates(
            {"tp_synced": get_synced_random_state(random_state=get_current_random_state(), pg=tp_pg)}
        )
    else:
        # NOTE: We don't need to sync across TP when using sequence parallel (REDUCE_SCATTER)
        random_states = RandomStates({})
    return random_states


def lr_scheduler_builder(optimizer: Optimizer, lr_scheduler_args: LRSchedulerArgs, total_training_steps: int):
    if lr_scheduler_args.lr_decay_steps is None:
        lr_decay_steps = total_training_steps
        if lr_scheduler_args.lr_warmup_steps is not None:
            lr_decay_steps -= lr_scheduler_args.lr_warmup_steps
        if lr_scheduler_args.lr_decay_starting_step is not None:
            lr_decay_steps -= lr_scheduler_args.lr_decay_starting_step
    else:
        lr_decay_steps = lr_scheduler_args.lr_decay_steps

    if lr_scheduler_args.lr_decay_starting_step is None:
        if lr_scheduler_args.lr_warmup_steps is not None:
            lr_decay_starting_step = lr_scheduler_args.lr_warmup_steps
        else:
            lr_decay_starting_step = 0
    else:
        lr_decay_starting_step = lr_scheduler_args.lr_decay_starting_step

    def lr_lambda(current_step: int, initial_lr: float):
        """
        current_step: current training step
        initial_lr: the learning rate of a parameter group

        More info on initial_lr:
        And in standard parameterization, lr_lambda only takes a single learning rate.
        But in µTransfer, each parameter has a custom learning rate (custom_lr = lr_scheduler_args.learning_rate * scaling_factor),
        so each parameter group has a custom lr_lambda function.

        LR Scheduling function, it has from 2 up to 4 phases:
        - warmup,
        - optional: constant (if lr_decay_starting_step is set)
        - decay
        - optional: constant (if lr_decay_steps and/or lr_decay_starting_step are set)
        Warmup starts at lr=0 and ends at `lr=lr`
        Then it stays constant at lr if lr_decay_starting_step is set and larger than lr_warmup_steps
        Then it decays until `min_decay_lr` for lr_decay_steps if set, else: (total_training_steps - lr_warmup_steps or lr_decay_starting_step)
        Then it stays constant at min_decay_lr if lr_decay_starting_step is set and total_training_steps is larger)
        """
        # No warmup or decay
        if lr_scheduler_args.lr_warmup_steps == 0 and lr_decay_steps == 0:
            return initial_lr

        # Warmup phase
        elif lr_scheduler_args.lr_warmup_style is not None and current_step <= lr_scheduler_args.lr_warmup_steps:
            if lr_scheduler_args.lr_warmup_style == "linear":
                lmbda = initial_lr * current_step / max(lr_scheduler_args.lr_warmup_steps, 1)
            elif lr_scheduler_args.lr_warmup_style == "constant":
                lmbda = lr_scheduler_args.learning_rate
            else:
                raise ValueError(f"Unknown warmup style {lr_scheduler_args.lr_warmup_style}")

        # Optional constant phase at learning_rate
        elif current_step < lr_decay_starting_step:
            lmbda = initial_lr

        # Decay phase
        elif lr_scheduler_args.lr_decay_style is not None and current_step < lr_decay_starting_step + lr_decay_steps:
            if lr_scheduler_args.lr_decay_style == "cosine":
                lmbda = (
                    lr_scheduler_args.min_decay_lr
                    + (initial_lr - lr_scheduler_args.min_decay_lr)
                    * (1 + math.cos(math.pi * (current_step - lr_decay_starting_step) / lr_decay_steps))
                    / 2
                )
            elif lr_scheduler_args.lr_decay_style == "linear":
                lmbda = (
                    lr_scheduler_args.min_decay_lr
                    + (initial_lr - lr_scheduler_args.min_decay_lr)
                    * (lr_decay_steps - (current_step - lr_decay_starting_step))
                    / lr_decay_steps
                )
            elif lr_scheduler_args.lr_decay_style == "1-sqrt":
                lmbda = lr_scheduler_args.min_decay_lr + (initial_lr - lr_scheduler_args.min_decay_lr) * (
                    1 - math.sqrt((current_step - lr_decay_starting_step) / lr_decay_steps)
                )
            else:
                raise ValueError(f"Unknown decay style {lr_scheduler_args.lr_decay_style}")

        # Optional constant phase at min_decay_lr
        else:
            lmbda = lr_scheduler_args.min_decay_lr

        lmbda /= initial_lr  # Normalization for pytorch
        return lmbda

    def get_lr_lambda_for_param_group(lr: float):
        return partial(lr_lambda, initial_lr=lr)

    # NOTE: get learning rate scheduler for each param group
    lr_lambdas = []
    for param_group in optimizer.get_base_optimizer().param_groups:
        lr_lambdas.append(get_lr_lambda_for_param_group(lr=param_group["lr"]))

    assert len(lr_lambdas) == len(
        optimizer.get_base_optimizer().param_groups
    ), "Custom learning rate functions dont match the number of param groups"

    log_rank(
        f"[Optimizer Building] There are total {len(lr_lambdas)} custom learning rate function for parameter groups",
        logger=logger,
        level=logging.DEBUG,
    )

    lr_scheduler = LambdaLR(optimizer.get_base_optimizer(), lr_lambda=lr_lambdas)
    return lr_scheduler


def get_custom_weight_decay_for_named_parameters(
    named_parameters: Iterable[Tuple[str, torch.Tensor]],
    model: NanotronModel,
    module_id_to_prefix: Dict[int, str],
    weight_decay: float,
    exclude_named_params: List[str],
) -> List[Dict[str, Any]]:
    """
    Apply weight decay to all parameters except those whose names match regex patterns in exclude_named_params.

    Args:
        named_parameters: Iterable of (name, parameter) tuples
        model: The model instance
        module_id_to_prefix: Mapping from module id to name prefix
        weight_decay: The default weight decay value to apply
        exclude_named_params: List of regex patterns to exclude parameters from weight decay

    Returns:
        List of parameter groups with appropriate weight decay values
    """
    named_param_groups_with_custom_weight_decay = []
    counter_excluded_params = 0
    for name, param in named_parameters:
        # Handle tied parameters: we exclude all tied parameters if one of them is in the exclude list
        if param.is_tied:
            tied_name = param.get_tied_info().get_full_name_from_module_id_to_prefix(
                module_id_to_prefix=module_id_to_prefix
            )

        # Determine weight decay value
        should_exclude = False
        if exclude_named_params is not None:
            should_exclude = any(
                re.match(pattern, name) or re.match(pattern, tied_name) for pattern in exclude_named_params
            )
            if should_exclude:
                counter_excluded_params += 1
                log_rank(f"Excluding parameter {name} from weight decay", logger=logger, level=logging.DEBUG, rank=0)

        current_weight_decay = 0.0 if should_exclude else weight_decay

        # Create parameter group
        named_param_groups_with_custom_weight_decay.append(
            {"named_params": [(name, param)], "weight_decay": current_weight_decay}
        )

    log_rank(
        f"[Optimizer Building] {counter_excluded_params} params excluded from weight decay",
        logger=logger,
        level=logging.WARNING,
        rank=0,
    )
    return named_param_groups_with_custom_weight_decay


def get_custom_lr_for_named_parameters(
    parametrization_method: ParametrizationMethod,
    lr: float,
    named_parameters: Iterable[Tuple[str, torch.Tensor]],
    model: NanotronModel,
) -> List[Dict[str, Any]]:
    """
    Get custom learning rates for parameters based on the parametrization method.

    NOTE: in some paramtrization methods, we use a global learning rate for all parameters,
    in others we use a custom learning rate for each parameter (eg: spectral µTransfer).
    """

    assert parametrization_method in [ParametrizationMethod.SPECTRAL_MUP, ParametrizationMethod.STANDARD]

    lr_mapper_cls = (
        LearningRateForSpectralMup
        if parametrization_method == ParametrizationMethod.SPECTRAL_MUP
        else LearningRateForSP
    )

    log_rank(
        f"[Optimizer Building] Using {lr_mapper_cls.__name__} as learning rate",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    # NOTE: since in the case of pipeline parallelism, each rank only has a subset of the model
    # so we only get the parameters that are in the current rank
    learning_rate_mapper = lr_mapper_cls(names_to_modules=model.named_modules_in_pp_rank, lr=lr)

    named_param_groups_with_custom_lr = []
    for (
        name,
        param,
    ) in named_parameters:
        learning_rate = learning_rate_mapper.get_lr(name, param)
        assert isinstance(learning_rate, float), f"Expected a float, got {learning_rate} for parameter {name}"
        named_param_groups_with_custom_lr.append({"named_params": [(name, param)], "lr": learning_rate})

    log_rank(
        f"[Optimizer Building] Creating {len(named_param_groups_with_custom_lr)} param groups with custom learning rates",
        logger=logger,
        level=logging.DEBUG,
    )

    return named_param_groups_with_custom_lr


def merge_named_param_groups(
    named_param_groups_with_lr: List[Dict[str, Any]],
    named_param_groups_with_weight_decay: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    assert len(named_param_groups_with_lr) == len(
        named_param_groups_with_weight_decay
    ), "Named param groups don't match in length"

    named_param_groups = []
    for group_with_lr, group_with_weight_decay in zip(
        named_param_groups_with_lr, named_param_groups_with_weight_decay
    ):
        assert group_with_lr["named_params"] == group_with_weight_decay["named_params"]
        named_param_groups.append(
            {
                "named_params": group_with_lr["named_params"],
                "lr": group_with_lr["lr"],
                "weight_decay": group_with_weight_decay["weight_decay"],
            }
        )

    return named_param_groups

def init_optimizer_and_grad_accumulator(
    parametrization_method: ParametrizationMethod,
    model: Union[nn.Module, DistributedDataParallel],
    optimizer_args: OptimizerArgs,
    parallel_context: ParallelContext,
) -> Tuple[BaseOptimizer, GradientAccumulator]:
    log_rank(
        "Building Optimizer and Gradient Accumulator and Learning Rate Scheduler",
        logger=logger,
        level=logging.INFO,
        rank=0,
        is_separator=True,
    )  # noqa
    # Unwrap DDP
    unwrapped_model: NanotronModel = model.module if isinstance(model, DistributedDataParallel) else model

    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in unwrapped_model.named_modules()}
    # Fix the root_model
    module_id_to_prefix[id(unwrapped_model)] = ""

    named_parameters = list(unwrapped_model.get_named_params_with_correct_tied())

    named_param_groups_with_lr = get_custom_lr_for_named_parameters(
        parametrization_method=parametrization_method,
        named_parameters=named_parameters,
        model=unwrapped_model,
        lr=optimizer_args.learning_rate_scheduler.learning_rate,
    )
    named_param_groups_with_weight_decay = get_custom_weight_decay_for_named_parameters(
        named_parameters=named_parameters,
        model=unwrapped_model,
        module_id_to_prefix=module_id_to_prefix,
        weight_decay=optimizer_args.weight_decay,
        exclude_named_params=optimizer_args.weight_decay_exclude_named_params,
    )

    named_param_groups = merge_named_param_groups(named_param_groups_with_lr, named_param_groups_with_weight_decay)

    # Basic optimizer builder
    def basic_optimizer_builder(named_param_groups):
        optimizer = None

        if optimizer_args.optimizer_factory.name == "adamW":

            def optimizer(param_groups):
                return torch.optim.AdamW(
                    param_groups,
                    lr=optimizer_args.learning_rate_scheduler.learning_rate,
                    weight_decay=optimizer_args.weight_decay,
                    eps=optimizer_args.optimizer_factory.adam_eps,
                    betas=(optimizer_args.optimizer_factory.adam_beta1, optimizer_args.optimizer_factory.adam_beta2),
                    fused=optimizer_args.optimizer_factory.torch_adam_is_fused,
                )

        elif optimizer_args.optimizer_factory.name == "sgd":

            def optimizer(param_groups):
                return torch.optim.SGD(
                    param_groups,
                    lr=optimizer_args.learning_rate_scheduler.learning_rate,
                    weight_decay=optimizer_args.weight_decay,
                )

        else:
            raise ValueError(f"Optimizer {optimizer_args.optimizer_factory.name} is not supported")

        return NamedOptimizer(
            named_params_or_groups=named_param_groups,
            optimizer_builder=optimizer,
        )

    optimizer_builder = basic_optimizer_builder

    # Gradient accumulator builder
    grad_accumulator: Optional[GradientAccumulator] = None
    if optimizer_args.accumulate_grad_in_fp32:

        def grad_optimizer_builder(named_param_groups):
            result = OptimizerFromGradientAccumulator(
                gradient_accumulator_builder=lambda named_params: FP32GradientAccumulator(
                    named_parameters=named_params,
                    grad_buckets_named_params=named_parameters,
                ),
                named_params_or_groups=named_param_groups,
                optimizer_builder=basic_optimizer_builder,
            )

            nonlocal grad_accumulator
            grad_accumulator = result.gradient_accumulator

            return result

        optimizer_builder = grad_optimizer_builder

    if optimizer_args.zero_stage > 0:
        # Build optimizer
        optimizer = ZeroDistributedOptimizer(
            named_params_or_groups=named_param_groups,
            optimizer_builder=optimizer_builder,
            dp_pg=parallel_context.dp_pg,
        )

        # SANITY CHECK: assert that optimizer's named_params point to model's params (check only the first one)
        if (
            len(optimizer.zero_named_param_groups) > 0
            and len(optimizer.zero_named_param_groups[0]["named_params"]) > 0
        ):
            optim_model_param_name, optim_model_param = optimizer.zero_named_param_groups[0]["named_params"][0]
            if isinstance(model, DistributedDataParallel):
                optim_model_param_name = f"module.{optim_model_param_name}"
            param = model.get_parameter(optim_model_param_name)
            assert param.data_ptr() == optim_model_param.data_ptr()
    else:
        # Build optimizer
        optimizer = optimizer_builder(named_param_groups)

    if grad_accumulator is not None and optimizer_args.zero_stage > 0:
        # There's a way to only require to reduce_scatter the gradients instead of all_reducing
        # In order to do so I need to pass which segments of each parameter should be reduced on which dp rank.
        assert isinstance(optimizer, ZeroDistributedOptimizer)
        param_name_to_dp_rank_offsets = optimizer.param_name_to_dp_rank_offsets

        assert isinstance(grad_accumulator, FP32GradientAccumulator)
        grad_accumulator.assign_param_offsets(
            dp_rank=dist.get_rank(parallel_context.dp_pg),
            param_name_to_offsets=param_name_to_dp_rank_offsets,
        )

    # Register DDP hook to make fp32 grad accumulation work
    if isinstance(model, DistributedDataParallel) and grad_accumulator is not None:
        assert isinstance(grad_accumulator, FP32GradientAccumulator)
        model.register_comm_hook(
            state=FP32GradBucketManager(
                dp_cp_pg=parallel_context.dp_cp_pg,
                accumulator=grad_accumulator,
                param_id_to_name={
                    id(param): param.get_tied_info().get_full_name_from_module_id_to_prefix(
                        module_id_to_prefix=module_id_to_prefix
                    )
                    if param.is_tied # a tied param exists only once physically in memory
                    else name
                    for name, param in unwrapped_model.named_parameters()
                },
            ),
            hook=get_fp32_accum_hook(
                reduce_scatter=optimizer.inherit_from(ZeroDistributedOptimizer), reduce_op=dist.ReduceOp.AVG
            ),
        )

    return optimizer, grad_accumulator


def test_equal_dict(first: Dict, second: Dict, sub_paths: Optional[List[str]] = None) -> None:
    """Raise if doesn't match."""

    if sub_paths is None:
        sub_paths = []

    first_keys = set(first.keys())
    second_keys = set(second.keys())
    assert first_keys == second_keys, f"Keys don't match.\nFirst: {first_keys}\nSecond: {second_keys}"
    for key in first_keys:
        first_elt = first[key]
        second_elt = second[key]

        if isinstance(first_elt, dict):
            assert isinstance(second_elt, dict), f"{first_elt} doesn't match {second_elt}"
            test_equal_dict(first_elt, second_elt, sub_paths=sub_paths + [str(key)])
        elif isinstance(first_elt, torch.Tensor):
            assert isinstance(second_elt, torch.Tensor), f"{first_elt} doesn't match {second_elt}"
            torch.testing.assert_close(
                first_elt,
                second_elt,
                atol=0.0,
                rtol=0.0,
                msg=lambda msg: f"tensor at {'.'.join(sub_paths + [str(key)])} don't match.\nCur: {first_elt}\nRef: {second_elt}\n{msg}",
            )
        else:
            assert (
                first_elt == second_elt
            ), f"{first_elt} doesn't match {second_elt} at key {'.'.join(sub_paths + [str(key)])}"


def get_profiler(config: Config):
    if config.profiler is not None and dist.get_rank() == 0:
        if config.profiler.profiler_export_path is not None:
            on_trace_ready = tensorboard_trace_handler(
                config.profiler.profiler_export_path / datetime.now().strftime("%Y%m%d-%H%M%S-" + config.general.run)
            )
        else:
            on_trace_ready = None

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=config.profiler.wait,
                warmup=config.profiler.warmup,
                active=config.profiler.active,
                repeat=config.profiler.repeat,
                skip_first=config.profiler.skip_first,
            ),
            on_trace_ready=on_trace_ready,
            record_shapes=config.profiler.record_shapes,
            profile_memory=config.profiler.profile_memory,
            with_stack=config.profiler.with_stack,
        )
    else:
        prof = contextlib.nullcontext()
    return prof


def get_all_comps(n: int) -> List[List[List[int]]]:
    """Return a 3D numpy array with a series of pairs to test latency/bandwidth between:
        This basically make a square matrix from the triangle of pair-to-pair comparisons


    [[[0 1]
    [2 3]]

    [[0 2]
    [1 3]]

    [[0 3]
    [1 2]]]
    """
    # n: power of two
    if not ((n & (n - 1) == 0) and n != 0):
        # every power of 2 has exactly 1 bit set to 1 (the bit in that number's log base-2 index).
        # So when subtracting 1 from it, that bit flips to 0 and all preceding bits flip to 1.
        # That makes these 2 numbers the inverse of each other so when AND-ing them, we will get 0 as the result
        raise ValueError("n must be a power of two")

    def op(lst, d=4, r=1):
        lst = lst.reshape(-1, d)
        lst[1::2] = np.roll(lst[1::2], r, axis=1)
        return lst.T.reshape(-1)

    x = np.array(list(range(n)))
    comps = []
    d = 1
    while d < n:
        for r in range(d):
            comps.append(op(x, d=d, r=r).copy())
        d *= 2
    ret = np.stack(comps)
    return ret.reshape(ret.shape[0], -1, 2).tolist()


def test_all_pair_to_pair(
    parallel_context: ParallelContext, throughput_size: int, throughput_iters: int, only_node_to_node: bool = True
):
    """Test all pair-to-pair GPUs throughput

    Args:
        parallel_context: ParallelContext
        throughput_size: size of the tensor to send
        throughput_iters: number of warm-up iterations before testing the throughput
        only_node_to_node: if True, only test node-to-node throughput
    """
    comparisons = get_all_comps(parallel_context.world_pg.size())
    wr = dist.get_rank(parallel_context.world_pg)
    log_rank(
        f"[TEST] Testing throughput between {comparisons}",
        logger=logger,
        level=logging.WARNING,
        group=parallel_context.world_pg,
        rank=0,
    )
    for j, comp in enumerate(comparisons):
        dist.barrier(group=parallel_context.world_pg)
        for i, (a, b) in enumerate(comp):
            dist.barrier(group=parallel_context.world_pg)
            if wr not in [a, b]:
                continue
            if only_node_to_node and (a % 8 != 0 or b % 8 != 0):
                # We only check node-to-node throughput
                continue
            test_tensor = torch.zeros((int(throughput_size),), dtype=torch.uint8, device=torch.device("cuda"))
            for k in range(throughput_iters):
                pre = time.perf_counter()
                torch.cuda.synchronize()
                if wr == a:
                    dist.send(test_tensor, b, group=parallel_context.world_pg, tag=i + k)
                elif wr == b:
                    dist.recv(test_tensor, a, group=parallel_context.world_pg, tag=i + k)
                torch.cuda.synchronize()
                duration = time.perf_counter() - pre
            del test_tensor
            gc.collect()
            torch.cuda.empty_cache()
            tput = (float(throughput_size) / duration) * 8  # *8 for gigabits/second
            log_rank(
                f"[TEST] {j, i, wr} Results throughput from {a} to {b}: {tput/1e9:.4f} Gbps",
                logger=logger,
                level=logging.WARNING,
                group=parallel_context.world_pg,
                rank=None,
            )
    log_rank(
        "[TEST] All comparisons done",
        logger=logger,
        level=logging.WARNING,
        group=parallel_context.world_pg,
        rank=0,
    )


def create_table_log(
    config: Config,
    parallel_context: ParallelContext,
    model_tflops,
    hardware_tflops,
    tokens_per_sec,
    bandwidth,
    num_params,
    slurm_job_id,
):
    return [
        LogItem("job_id", slurm_job_id, "s"),
        LogItem("name", config.general.run, "s"),
        LogItem("nodes", math.ceil(parallel_context.world_pg.size() / torch.cuda.device_count()), "d"),
        LogItem("seq_len", config.tokens.sequence_length, "d"),
        LogItem("mbs", config.tokens.micro_batch_size, "d"),
        LogItem("batch_accum", config.tokens.batch_accumulation_per_replica, "d"),
        LogItem("gbs", config.global_batch_size, "d"),
        LogItem("mTFLOPs", model_tflops, ".2f"),
        LogItem("hTFLOPs", hardware_tflops, ".2f"),
        LogItem("tok/s/gpu", tokens_per_sec / parallel_context.world_pg.size(), ".2f"),
        LogItem("Mem Alloc (GB)", torch.cuda.max_memory_allocated() / 1024**3, ".2f"),
        LogItem("Mem Res (GB)", torch.cuda.max_memory_reserved() / 1024**3, ".2f"),
        # Important config columns
        LogItem("dp", config.parallelism.dp, "d"),
        LogItem("pp", config.parallelism.pp, "d"),
        LogItem("tp", config.parallelism.tp, "d"),
        LogItem("cp", config.parallelism.context_parallel_size, "d"),
        LogItem("ep", config.parallelism.expert_parallel_size, "d"),
        LogItem("pp_engine", str(config.parallelism.pp_engine), "s"),
        LogItem("tp_mode", config.parallelism.tp_mode, "s"),
        LogItem("tp_async_comm", str(config.parallelism.tp_linear_async_communication), "s"),
        LogItem("recompute_layer", str(config.parallelism.recompute_layer), "s"),
        LogItem("hidden_size", config.model.model_config.hidden_size, "d"),
        LogItem("hidden_act", config.model.model_config.hidden_act, "s"),
        LogItem("num_layers", config.model.model_config.num_hidden_layers, "d"),
        LogItem("num_heads", config.model.model_config.num_attention_heads, "d"),
        LogItem("num_kv_heads", config.model.model_config.num_key_value_heads, "d"),
        LogItem("max_pos", config.model.model_config.max_position_embeddings, "d"),
        LogItem("vocab_size", config.model.model_config.vocab_size, "d"),
        LogItem("tie_word_embeddings", str(config.model.model_config.tie_word_embeddings), "s"),
        LogItem("dtype", str(config.model.dtype), "s"),
        LogItem("zero_stage", config.optimizer.zero_stage, "d"),
        LogItem("ddp_bucket_cap_mb", config.model.ddp_bucket_cap_mb, "d"),
        LogItem("accumulate_grad_in_fp32", str(config.optimizer.accumulate_grad_in_fp32), "s"),
        # Params
        LogItem("Total Params", num_params["total"], "human_format"),
        LogItem("Local Params", num_params["local"], "human_format"),
    ]


def get_formatted_value(item):
    if item.log_format == "human_format":
        return human_format(item.scalar_value)
    return f"{item.scalar_value:{item.log_format}}"


def create_table_output(table_log, column_widths):
    header_row = "| " + " | ".join([item.tag.ljust(width) for item, width in zip(table_log, column_widths)]) + " |"
    separator_row = "| " + " | ".join(["-" * width for width in column_widths]) + " |"
    data_row = (
        "| "
        + " | ".join([get_formatted_value(item).ljust(width) for item, width in zip(table_log, column_widths)])
        + " |"
    )
    return f"{header_row}\n{separator_row}\n{data_row}"


def write_to_csv(csv_filename, table_log, model_tflops, slurm_job_id):
    """Write benchmark results to a CSV file with file locking using fcntl."""
    import fcntl

    try:
        # Check if csv_filename is valid
        if not csv_filename:
            logger.warning("No benchmark CSV path specified - skipping CSV output")
            return

        # Create output directory if needed
        csv_dir = os.path.dirname(csv_filename)
        if csv_dir:  # Only try to create directory if path has a directory component
            os.makedirs(csv_dir, exist_ok=True)

        # Format row data
        header = [item.tag for item in table_log]
        row = [get_formatted_value(item) for item in table_log]

        # Use fcntl for file locking
        max_attempts = 10
        attempt = 0
        log_rank(
            f"Attempting to write benchmark results to CSV file: {csv_filename}",
            logger=logger,
            level=logging.INFO,
        )
        while attempt < max_attempts:
            try:
                with open(csv_filename, mode="a+", newline="") as f:
                    # Get exclusive lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    log_rank(
                        f"Acquired lock for CSV file: {csv_filename}",
                        logger=logger,
                        level=logging.INFO,
                    )
                    try:
                        # Check if file is empty/new
                        f.seek(0)
                        first_char = f.read(1)
                        if not first_char:  # Empty file
                            writer = csv.writer(f)
                            writer.writerow(header)

                        # Go to end of file for append
                        f.seek(0, os.SEEK_END)
                        writer = csv.writer(f)
                        writer.writerow(row)
                        f.flush()
                        os.fsync(f.fileno())
                        break
                    finally:
                        # Release lock
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        log_rank(
                            f"Successfully wrote to CSV file: {csv_filename}. Releasing lock...",
                            logger=logger,
                            level=logging.INFO,
                        )
            except BlockingIOError:
                # Another process has the lock, wait and retry
                log_rank(
                    f"Another process has the lock for CSV file: {csv_filename}, waiting and retrying attempt {attempt + 1} of {max_attempts}...",
                    logger=logger,
                    level=logging.INFO,
                )
                attempt += 1
                time.sleep(0.1)  # Wait 100ms before retrying

        if attempt == max_attempts:
            logger.error(f"Failed to acquire lock for {csv_filename} after {max_attempts} attempts")

    except Exception as e:
        logger.error(f"Unexpected error writing to {csv_filename}: {str(e)}")
        return


def log_throughput(
    config: Config,
    parallel_context: ParallelContext,
    model_tflops=0,
    hardware_tflops=0,
    tokens_per_sec=0,
    bandwidth=0,
    num_params={"total": 0, "local": 0},
):
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "N/A")

    table_log = create_table_log(
        config, parallel_context, model_tflops, hardware_tflops, tokens_per_sec, bandwidth, num_params, slurm_job_id
    )
    column_widths = [max(len(item.tag), len(get_formatted_value(item))) for item in table_log]
    table_output = create_table_output(table_log, column_widths)

    log_rank(
        table_output,
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    if dist.get_rank(parallel_context.world_pg) == 0:
        write_to_csv(config.general.benchmark_csv_path, table_log, model_tflops, slurm_job_id)
    dist.barrier(group=parallel_context.world_pg)


def compute_remain_train_steps_of_a_data_stage_from_ckp(
    stage: DatasetStageArgs, config: Config, metadata: TrainingMetadata
) -> int:
    def is_last_stage():
        sorted_stages = sorted(config.data_stages, key=lambda x: x.start_training_step)
        return sorted_stages[-1].start_training_step == stage.start_training_step

    def is_resume_from_training():
        return metadata.last_train_step > 0

    if is_last_stage() is True:
        total_train_steps = config.tokens.train_steps
    else:
        next_stage = next((s for s in config.data_stages if s.start_training_step > stage.start_training_step), None)
        total_train_steps = next_stage.start_training_step

    if metadata.last_train_step > stage.start_training_step:
        # NOTE: if the last_train_step is larger than the start_training_step of the current stage,
        # it means that the training has already passed this stage
        # so there is no remaining steps
        return 0
    else:
        last_train_steps = metadata.last_train_step if is_resume_from_training() else stage.start_training_step
        return total_train_steps - last_train_steps + 1


def get_consumed_train_samples_of_a_data_stage_from_ckp(
    stage: DatasetStageArgs, metadata: TrainingMetadata
) -> Optional[int]:
    start_training_step = stage.start_training_step

    # find the stage in the metadata using the start_training_step
    actual_stage = next(
        (s for s in metadata.data_stages if s.start_training_step == start_training_step),
        None,
    )
    if actual_stage is None:
        # If stage not found in metadata, ensure we haven't passed this stage's start step
        assert (
            metadata.last_train_step < stage.start_training_step
        ), f"Stage {stage.name} starting at step {stage.start_training_step} not found in checkpoint metadata, but last_train_step ({metadata.last_train_step}) >= start_training_step"
        consumed_train_samples = 0

        # Add new stage to metadata
        if metadata.data_stages is None:
            metadata.data_stages = []

        actual_stage = DataStageMetadata(
            name=stage.name,
            start_training_step=stage.start_training_step,
            consumed_train_samples=consumed_train_samples,
        )
        metadata.data_stages.append(actual_stage)

    return actual_stage.consumed_train_samples, actual_stage.consumed_tokens_per_dataset_folder
