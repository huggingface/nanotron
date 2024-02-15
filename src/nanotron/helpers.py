import contextlib
import csv
import gc
import math
import os
import time
from datetime import datetime
from math import ceil
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import (
    Config,
    LRSchedulerArgs,
    OptimizerArgs,
    ParallelismArgs,
)
from nanotron.distributed import ProcessGroup
from nanotron.logging import LogItem, log_rank
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
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelLinearMode,
)
from nanotron.random import (
    RandomStates,
    get_current_random_state,
    get_synced_random_state,
)

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
        # We don't need to sync across TP when using sequence parallel (REDUCE_SCATTER)
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

    def lr_lambda(current_step: int):
        """LR Scheduling function, it has from 2 up to 4 phases:
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
            return lr_scheduler_args.learning_rate

        # Warmup phase
        elif lr_scheduler_args.lr_warmup_style is not None and current_step <= lr_scheduler_args.lr_warmup_steps:
            if lr_scheduler_args.lr_warmup_style == "linear":
                lmbda = lr_scheduler_args.learning_rate * current_step / max(lr_scheduler_args.lr_warmup_steps, 1)
            elif lr_scheduler_args.lr_warmup_style == "constant":
                lmbda = lr_scheduler_args.learning_rate
            else:
                raise ValueError(f"Unknown warmup style {lr_scheduler_args.lr_warmup_style}")

        # Optional constant phase at learning_rate
        elif current_step < lr_decay_starting_step:
            lmbda = lr_scheduler_args.learning_rate

        # Decay phase
        elif lr_scheduler_args.lr_decay_style is not None and current_step < lr_decay_starting_step + lr_decay_steps:
            if lr_scheduler_args.lr_decay_style == "cosine":
                lmbda = (
                    lr_scheduler_args.min_decay_lr
                    + (lr_scheduler_args.learning_rate - lr_scheduler_args.min_decay_lr)
                    * (1 + math.cos(math.pi * (current_step - lr_decay_starting_step) / lr_decay_steps))
                    / 2
                )
            elif lr_scheduler_args.lr_decay_style == "linear":
                lmbda = (
                    lr_scheduler_args.min_decay_lr
                    + (lr_scheduler_args.learning_rate - lr_scheduler_args.min_decay_lr)
                    * (lr_decay_steps - (current_step - lr_decay_starting_step))
                    / lr_decay_steps
                )
            else:
                raise ValueError(f"Unknown decay style {lr_scheduler_args.lr_decay_style}")

        # Optional constant phase at min_decay_lr
        else:
            lmbda = lr_scheduler_args.min_decay_lr

        lmbda /= lr_scheduler_args.learning_rate  # Normalization for pytorch
        return lmbda

    lr_scheduler = LambdaLR(optimizer.get_base_optimizer(), lr_lambda=lr_lambda)
    return lr_scheduler


def init_optimizer_and_grad_accumulator(
    model: nn.Module, optimizer_args: OptimizerArgs, parallel_context: ParallelContext
) -> Tuple[BaseOptimizer, GradientAccumulator]:
    # Unwrap DDP
    unwrapped_model: NanotronModel = model.module if isinstance(model, DistributedDataParallel) else model

    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in unwrapped_model.named_modules()}
    # Fix the root_model
    module_id_to_prefix[id(unwrapped_model)] = ""

    # named parameters
    named_parameters = list(unwrapped_model.get_named_params_with_correct_tied())

    # Basic optimizer builder
    def basic_optimizer_builder(named_param_groups):
        return NamedOptimizer(
            named_params_or_groups=named_param_groups,
            optimizer_builder=lambda param_groups: AdamW(  # pylint: disable=E0601
                param_groups,
                lr=optimizer_args.learning_rate_scheduler.learning_rate,
                weight_decay=optimizer_args.weight_decay,
                eps=optimizer_args.adam_eps,
                betas=(optimizer_args.adam_beta1, optimizer_args.adam_beta2),
                fused=optimizer_args.torch_adam_is_fused,
            ),
        )

    optimizer_builder = basic_optimizer_builder

    # Gradient accumulator builder
    grad_accumulator: Optional[GradientAccumulator] = None
    if optimizer_args.accumulate_grad_in_fp32:
        # TODO @thomasw21: Make an optimizer builder system, instead of doing everything in functional manner
        def grad_optimizer_builder(named_param_groups):
            result = OptimizerFromGradientAccumulator(
                gradient_accumulator_builder=lambda named_params: FP32GradientAccumulator(
                    named_parameters=named_params,
                    grad_buckets_named_params=named_parameters,
                ),
                named_params_or_groups=named_param_groups,
                optimizer_builder=basic_optimizer_builder,
            )

            # TODO @thomasw21: get better API to get the grad_accumulator
            nonlocal grad_accumulator
            grad_accumulator = result.gradient_accumulator

            return result

        optimizer_builder = grad_optimizer_builder

    if optimizer_args.zero_stage > 0:
        # Build optimizer
        optimizer = ZeroDistributedOptimizer(
            named_params_or_groups=named_parameters,
            # TODO @thomasw21: We need a better API for gradient accumulation/zero etc ...
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
        optimizer = optimizer_builder(named_parameters)

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
                dp_pg=parallel_context.dp_pg,
                accumulator=grad_accumulator,
                param_id_to_name={
                    id(param): param.get_tied_info().get_full_name_from_module_id_to_prefix(
                        module_id_to_prefix=module_id_to_prefix
                    )
                    if param.is_tied
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
    """Raise if doesn't match"""
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
    if config.profiler is not None:
        if config.profiler.profiler_export_path is not None:
            on_trace_ready = tensorboard_trace_handler(
                config.profiler.profiler_export_path / datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        else:
            on_trace_ready = None
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=3),
            on_trace_ready=on_trace_ready,
            # record_shapes=True,
            # profile_memory=True,
            with_stack=True,
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
    slurm_job_id,
):
    return [
        LogItem("job_id", slurm_job_id, "s"),
        LogItem("name", config.general.run, "s"),
        LogItem("nodes", math.ceil(parallel_context.world_pg.size() / 8), "d"),
        LogItem("seq_len", config.tokens.sequence_length, "d"),
        LogItem("mbs", config.tokens.micro_batch_size, "d"),
        LogItem("batch_accum", config.tokens.batch_accumulation_per_replica, "d"),
        LogItem("gbs", config.global_batch_size, "d"),
        LogItem("mTFLOPs", model_tflops, ".2f"),
        LogItem("hTFLOPs", hardware_tflops, ".2f"),
        LogItem("tok/s/gpu", tokens_per_sec / parallel_context.world_pg.size(), ".2f"),
        LogItem("Bandwidth (GB/s)", bandwidth, ".2f"),
        LogItem("Mem Alloc (GB)", torch.cuda.max_memory_allocated() / 1024**3, ".2f"),
        LogItem("Mem Res (GB)", torch.cuda.max_memory_reserved() / 1024**3, ".2f"),
    ]


def create_table_output(table_log, column_widths):
    header_row = "| " + " | ".join([item.tag.ljust(width) for item, width in zip(table_log, column_widths)]) + " |"
    separator_row = "| " + " | ".join(["-" * width for width in column_widths]) + " |"
    data_row = (
        "| "
        + " | ".join(
            [f"{item.scalar_value:{item.log_format}}".ljust(width) for item, width in zip(table_log, column_widths)]
        )
        + " |"
    )
    return f"{header_row}\n{separator_row}\n{data_row}"


def write_to_csv(csv_filename, table_log, model_tflops, slurm_job_id):
    if not os.path.exists(csv_filename):
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        with open(csv_filename, mode="w") as fo:
            writer = csv.writer(fo)
            writer.writerow([item.tag for item in table_log])
            writer.writerow([f"{item.scalar_value:{item.log_format}}" for item in table_log])
    # elif model_tflops > 0:
    #     # replace line with same job_id
    #     with open(csv_filename, mode="r") as fi:
    #         lines = fi.readlines()
    #     with open(csv_filename, mode="w") as fo:
    #         writer = csv.writer(fo)
    #         for line in lines:
    #             if line.startswith(slurm_job_id):
    #                 writer.writerow([f"{item.scalar_value:{item.log_format}}" for item in table_log])
    #             else:
    #                 fo.write(line)
    else:
        with open(csv_filename, mode="a") as fo:
            writer = csv.writer(fo)
            writer.writerow([f"{item.scalar_value:{item.log_format}}" for item in table_log])


def log_throughput(
    config: Config,
    parallel_context: ParallelContext,
    model_tflops=0,
    hardware_tflops=0,
    tokens_per_sec=0,
    bandwidth=0,
):
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "N/A")

    table_log = create_table_log(
        config, parallel_context, model_tflops, hardware_tflops, tokens_per_sec, bandwidth, slurm_job_id
    )
    column_widths = [max(len(item.tag), len(f"{item.scalar_value:{item.log_format}}")) for item in table_log]
    table_output = create_table_output(table_log, column_widths)

    log_rank(
        table_output,
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    if dist.get_rank(parallel_context.world_pg) == 0:
        write_to_csv(config.general.benchmark_csv_path, table_log, model_tflops, slurm_job_id)
