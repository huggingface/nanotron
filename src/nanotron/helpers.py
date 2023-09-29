import argparse
import contextlib
import logging as lg
import math
import sys
from math import ceil
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

from nanotron.config import (
    Config,
    LRSchedulerArgs,
    OptimizerArgs,
    ParallelismArgs,
)
from nanotron.core import distributed as dist
from nanotron.core import logging
from nanotron.core.dataclass import RandomStates
from nanotron.core.distributed import ProcessGroup
from nanotron.core.gradient_accumulator import (
    FP32GradBucketManager,
    FP32GradientAccumulator,
    GradientAccumulator,
    get_fp32_accum_hook,
)
from nanotron.core.logging import log_rank
from nanotron.core.optimizer.base import BaseOptimizer, Optimizer
from nanotron.core.optimizer.named_optimizer import NamedOptimizer
from nanotron.core.optimizer.optimizer_from_gradient_accumulator import (
    OptimizerFromGradientAccumulator,
)
from nanotron.core.optimizer.zero import ZeroDistributedOptimizer
from nanotron.core.parallelism.tensor_parallelism.nn import (
    TensorParallelLinearMode,
)
from nanotron.core.process_groups_initializer import DistributedProcessGroups
from nanotron.core.random import (
    get_current_random_state,
    get_synced_random_state,
)
from nanotron.models import NanotronModel

logger = logging.get_logger(__name__)

try:

    tb_logger_available = True
except ImportError:
    tb_logger_available = False

try:

    hub_logger_available = True
except ImportError:
    hub_logger_available = False


# TODO @nouamane: `AdamW` doesn't support `set_to_none=True`
# try:
#     from apex.optimizers import FusedAdam as AdamW
#
#     logger.warning("Discovered apex.optimizers.FusedAdam - will use it instead of torch.optim.AdamW")
# except Exception:
#     from torch.optim import AdamW


def get_args():
    parser = argparse.ArgumentParser()
    # CONFIG for YAML
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML config file")
    return parser.parse_args()


def set_logger_verbosity(logging_level: str, dpg: DistributedProcessGroups):
    formatter = lg.Formatter(
        fmt=f"%(asctime)s [%(levelname)s|DP={dist.get_rank(dpg.dp_pg)}|PP={dist.get_rank(dpg.pp_pg)}|TP={dist.get_rank(dpg.tp_pg)}]: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    # TODO @thomasw21: `logging.log_levels` returns valid lg log levels
    log_level = logging.log_levels[logging_level]

    # main root logger
    root_logger = logging.get_logger()
    root_logger.setLevel(log_level)
    handler = logging.NewLineStreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Nanotron
    logging.set_verbosity(log_level)
    logging.set_formatter(formatter=formatter)


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


def lr_scheduler_builder(optimizer: Optimizer, learning_rate: float, lr_scheduler_args: LRSchedulerArgs):
    def lr_lambda(current_step: int):
        """LR Scheduling function, it has 3 phases: warmup, decay, then constant. Warmup starts at lr=0 and ends at `lr=lr`, then it decays until `min_decay_lr` and then stays constant."""
        # No warmup or decay
        if lr_scheduler_args.lr_warmup_steps == 0 and lr_scheduler_args.lr_decay_steps == 0:
            return learning_rate

        # Warmup phase
        elif lr_scheduler_args.lr_warmup_style is not None and current_step <= lr_scheduler_args.lr_warmup_steps:
            if lr_scheduler_args.lr_warmup_style == "linear":
                lmbda = learning_rate * current_step / max(lr_scheduler_args.lr_warmup_steps, 1)
            elif lr_scheduler_args.lr_warmup_style == "constant":
                lmbda = learning_rate
            else:
                raise ValueError(f"Unknown warmup style {lr_scheduler_args.lr_warmup_style}")

        # Decay phase
        elif (
            lr_scheduler_args.lr_decay_style is not None
            and current_step < lr_scheduler_args.lr_decay_steps + lr_scheduler_args.lr_warmup_steps
        ):
            if lr_scheduler_args.lr_decay_style == "cosine":
                lmbda = (
                    lr_scheduler_args.min_decay_lr
                    + (learning_rate - lr_scheduler_args.min_decay_lr)
                    * (
                        1
                        + math.cos(
                            math.pi
                            * (current_step - lr_scheduler_args.lr_warmup_steps)
                            / lr_scheduler_args.lr_decay_steps
                        )
                    )
                    / 2
                )
            elif lr_scheduler_args.lr_decay_style == "linear":
                lmbda = (
                    lr_scheduler_args.min_decay_lr
                    + (learning_rate - lr_scheduler_args.min_decay_lr)
                    * (lr_scheduler_args.lr_decay_steps - (current_step - lr_scheduler_args.lr_warmup_steps))
                    / lr_scheduler_args.lr_decay_steps
                )
            else:
                raise ValueError(f"Unknown decay style {lr_scheduler_args.lr_decay_style}")

        # Constant phase
        else:
            lmbda = lr_scheduler_args.min_decay_lr

        lmbda /= learning_rate
        return lmbda

    lr_scheduler = LambdaLR(optimizer.get_base_optimizer(), lr_lambda=lr_lambda)
    return lr_scheduler


def init_optimizer_and_grad_accumulator(
    model: nn.Module, optimizer_args: OptimizerArgs, dpg: DistributedProcessGroups
) -> Tuple[BaseOptimizer, GradientAccumulator]:
    unwrapped_model: NanotronModel = model.module if isinstance(model, DistributedDataParallel) else model
    named_parameters = unwrapped_model.get_named_params_with_tied()

    # Basic optimizer builder
    def basic_optimizer_builder(named_param_groups):
        return NamedOptimizer(
            named_params_or_groups=named_param_groups,
            optimizer_builder=lambda param_groups: AdamW(
                param_groups,
                lr=optimizer_args.learning_rate,
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
            dp_pg=dpg.dp_pg,
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
            dp_rank=dist.get_rank(dpg.dp_pg),
            param_name_to_offsets=param_name_to_dp_rank_offsets,
        )

    # Register DDP hook to make fp32 grad accumulation work
    if isinstance(model, DistributedDataParallel) and grad_accumulator is not None:
        assert isinstance(grad_accumulator, FP32GradientAccumulator)
        model.register_comm_hook(
            state=FP32GradBucketManager(
                dp_pg=dpg.dp_pg,
                accumulator=grad_accumulator,
                param_id_to_name={
                    id(param): name
                    for name, param in named_parameters
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
    if config.profile is not None:
        if config.profile.profiler_export_path is not None:
            on_trace_ready = tensorboard_trace_handler(config.profile.profiler_export_path)
        else:
            on_trace_ready = None
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1, skip_first=1),
            on_trace_ready=on_trace_ready,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
    else:
        prof = contextlib.nullcontext()
    return prof
