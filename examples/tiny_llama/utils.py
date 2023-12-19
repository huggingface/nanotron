import logging as lg
import argparse
import sys
import datetime
import contextlib
import torch.distributed as dist
import numpy as np
from typing import Callable, List, Optional
from transformers import LlamaConfig
import torch
import torch.nn as nn
from math import ceil
from pprint import pformat
import math
from torch.nn.parallel import DistributedDataParallel
from typing import Callable, Dict, List, Optional, Tuple, Union, Generator
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from datasets.download.streaming_download_manager import xPath


from modeling_llama import LlamaDecoderLayer, RotaryEmbedding, RMSNorm, TensorParallelEmbedding

from nanotron.core.serialize import (
    save,
    save_random_states,
)
from nanotron.core.serialize.serialize import fs_open
from nanotron.core.serialize.path import check_path_is_local
from nanotron.core.parallelism.pipeline_parallelism.tensor_pointer import TensorPointer
from nanotron.core.optimizer.base import BaseOptimizer, Optimizer
from nanotron.core.parallelism.pipeline_parallelism.block import PipelineBlock
from nanotron.core.parallelism.parameters import NanotronParameter, sanity_check
from nanotron.core.dataclass import RandomStates
from nanotron.core.distributed import ProcessGroup
from nanotron.core.process_groups_initializer import DistributedProcessGroups
from nanotron.core.parallelism.tensor_parallelism.nn import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelLinearMode,
)
from nanotron.core.gradient_accumulator import (
    FP32GradBucketManager,
    FP32GradientAccumulator,
    GradientAccumulator,
    get_fp32_accum_hook,
)
from nanotron.core import logging
from nanotron.core.logging import log_rank
from nanotron.logger import LoggerWriter
from config import (
    Config,
    OptimizerArgs,
    ParallelismArgs,
    TensorboardLoggerConfig,
    HubLoggerConfig,
    LRSchedulerArgs
)
from nanotron.core.optimizer.named_optimizer import NamedOptimizer


from nanotron.core.utils import init_on_device_and_dtype

from nanotron.core.parallelism.tied_parameters import (
    get_tied_id_to_param,
    create_pg_for_tied_weights,
    tie_parameters,
)

from nanotron.core.random import (
    get_current_random_state,
    get_synced_random_state,
)
from nanotron.core.optimizer.optimizer_from_gradient_accumulator import (
    OptimizerFromGradientAccumulator,
)

from nanotron.core.tensor_init import init_method_normal, scaled_init_method_normal
from nanotron.core.optimizer.zero import ZeroDistributedOptimizer

try:
    from nanotron.logger import HubSummaryWriter
    hub_logger_available = True
except ImportError:
    hub_logger_available = False

try:
    from nanotron.logger import BatchSummaryWriter
    tb_logger_available = True
except ImportError:
    tb_logger_available = False

def get_args():
    parser = argparse.ArgumentParser()
    # CONFIG for YAML
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML config file")
    return parser.parse_args()

def set_logger_verbosity_format(logging_level: str, dpg: DistributedProcessGroups):
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

    # nanotron
    logging.set_verbosity(log_level)
    logging.set_formatter(formatter=formatter)

def setup_log_writers(config: Config):
    if config.logging.tensorboard_logger is None:
        tb_context = contextlib.nullcontext()
    else:
        current_time = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        logdir = str(config.logging.tensorboard_logger.tensorboard_dir / f"{config.general.name}_{current_time}")

        if isinstance(config.logging.tensorboard_logger, HubLoggerConfig):
            assert (
                hub_logger_available
            ), 'Hub Tensorboard Logger is not available. Please install nanotron with `pip install -e ".[hf-logger]"` or modify your config file'
            tb_context = HubSummaryWriter(
                logdir=logdir,
                repo_id=config.logging.tensorboard_logger.repo_id,
                path_in_repo=f"tensorboard/{config.general.name}_{current_time}",
            )
        if isinstance(config.logging.tensorboard_logger, TensorboardLoggerConfig):
            assert (
                tb_logger_available
            ), 'Tensorboard Logger is not available. Please install nanotron with `pip install -e ".[tb-logger]"` or modify your config file'
            tb_context = BatchSummaryWriter(logdir=logdir)
    loggerwriter = LoggerWriter(global_step=config.tokens.train_steps)
    return tb_context, loggerwriter

def _vocab_size_with_padding(orig_vocab_size: int, pg_size: int, make_vocab_size_divisible_by: int, logger: lg.Logger):
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

def init_model(
    model_builder: Callable[[], nn.Module],
    model_config: LlamaConfig,
    dtype: torch.dtype,
    dpg: DistributedProcessGroups,
    logger: lg.Logger,
    parallel_config: Optional[ParallelismArgs],
    make_ddp: bool,
    target_pp_ranks: Optional[List[int]] = None,
) -> nn.Module:
    """
    Initializes the model for training.

    Args:
        model_builder (Callable[[], nn.Module]): The model builder function.
        model_config (LlamaConfig): The configuration for the model.
        dtype (torch.dtype): The data type to use for the model.
        dpg (DistributedProcessGroups): The distributed process groups to use for the model.
        parallel_config (Optional[ParallelismArgs]): The parallelism configuration to use for the model.
        make_ddp (bool): Whether to make the model a DistributedDataParallel model.
        target_pp_ranks (Optional[List[int]], optional): The target pipeline parallel ranks to use for the model. Defaults to None.

    Returns:
        nn.Module: The initialized model.
    """
    # If no target pp ranks are specified, we assume that we want to use all pp ranks
    if target_pp_ranks is None:
        pp_size = dpg.pp_pg.size()
        target_pp_ranks = list(range(pp_size))
    else:
        pp_size = len(target_pp_ranks)

    # Build model
    model = model_builder()

    # Set rank for each pipeline block
    pipeline_blocks = [module for name, module in model.named_modules() if isinstance(module, PipelineBlock)]
    # "cuda" is already defaulted for each process to it's own cuda device
    with init_on_device_and_dtype(device=torch.device("cuda"), dtype=dtype):
        # TODO: https://github.com/huggingface/nanotron/issues/65

        # Balance compute across PP blocks
        d_ff = model_config.intermediate_size
        d_qkv = model_config.hidden_size // model_config.num_attention_heads
        block_compute_costs = {
            # CausalSelfAttention (qkv proj + attn out) + MLP
            LlamaDecoderLayer: 4 * model_config.num_attention_heads * d_qkv * model_config.hidden_size
            + 3 * d_ff * model_config.hidden_size,
            # This is the last lm_head
            TensorParallelColumnLinear: model_config.vocab_size * model_config.hidden_size,
        }
        block_cumulative_costs = np.cumsum(
            [
                block_compute_costs[module.module_builder] if module.module_builder in block_compute_costs else 0
                for module in pipeline_blocks
            ]
        )

        thresholds = [block_cumulative_costs[-1] * ((rank + 1) / pp_size) for rank in range(pp_size)]
        assert thresholds[-1] >= block_cumulative_costs[-1]
        target_pp_rank_idx = 0
        for block, cumulative_cost in zip(pipeline_blocks, block_cumulative_costs):
            assert target_pp_rank_idx < pp_size
            block.build_and_set_rank(target_pp_ranks[target_pp_rank_idx])

            if cumulative_cost > thresholds[target_pp_rank_idx]:
                target_pp_rank_idx += 1

    # Initialize rotary embeddings
    for module in model.modules():
        if not isinstance(module, RotaryEmbedding):
            continue
        module.init_rotary_embeddings()

    # Sync all parameters that have the same name and that are not sharded across TP
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            name = f"{module_name}.{param_name}"

            if isinstance(param, NanotronParameter) and param.is_sharded:
                continue

            shared_weights = [
                (
                    name,
                    # This adds all the tp_ranks in one go
                    tuple(sorted(dpg.world_rank_matrix[dist.get_rank(dpg.pp_pg), dist.get_rank(dpg.dp_pg), :])),
                )
            ]

            # TODO @thomasw21: Somehow declaring tied weights at local level doesn't work correctly.
            if parallel_config is None or parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
                # We add `reduce_op=None` in order to signal that the weight are synced by design without needing to reduce
                # when TP=2 we have LN that is duplicated across TP, so by design it's tied
                reduce_op = None
            else:
                reduce_op = dist.ReduceOp.SUM

            tie_parameters(root_module=model, ties=shared_weights, dpg=dpg, reduce_op=reduce_op)

    create_pg_for_tied_weights(root_module=model, dpg=dpg)

    # count number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    size_params = sum(p.numel() * p.element_size() for p in model.parameters())

    # TODO @nouamanetazi: better memory logs
    log_rank(
        f"Number of parameters: {num_params} ({size_params / 1024**2:.2f}MB). Expecting peak 4*param_size={4*size_params / 1024**2:.2f}MB with grads and Adam optim states (w/o memory optims)",
        logger=logger,
        level=logging.INFO,
        group=dpg.dp_pg,
        rank=0,
    )
    log_rank(
        f"[After model building] Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MB. Peak reserved memory: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MB",
        logger=logger,
        level=logging.INFO,
        group=dpg.dp_pg,
        rank=0,
    )

    # Model make it DDP
    if make_ddp is True:
        # TODO @thomasw21: DDP doesn't support broadcasting complex buffers (and we don't really need that broadcasting anyway)
        model = DistributedDataParallel(model, process_group=dpg.dp_pg, broadcast_buffers=False)

    # Sanity check the model
    sanity_check(root_module=model)

    return model

def init_model_randomly(model: nn.Module, config: Config, model_config: LlamaConfig, dpg: DistributedProcessGroups):
    # Used for embedding/position/qkv weight in attention/first layer weight of mlp/ /lm_head/
    init_method_ = init_method_normal(config.model.init_method.std)
    # Used for o weight in attention/second layer weight of mlp/
    scaled_init_method_ = scaled_init_method_normal(config.model.init_method.std, model_config.num_hidden_layers)
    # Layernorm weight all 0 or 1 depending on `apply_layernorm_1p`

    initialized_parameters = set()
    # Handle tensor parallelism
    with torch.no_grad():
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        # Fix the root_model
        module_id_to_prefix[id(model)] = ""

        for module_name, module in model.named_modules():
            if isinstance(module, TensorParallelColumnLinear):
                # Somehow Megatron-LM does something super complicated, https://github.com/NVIDIA/Megatron-LM/blob/2360d732a399dd818d40cbe32828f65b260dee11/megatron/core/tensor_parallel/layers.py#L96
                # What it does:
                #  - instantiate a buffer of the `full size` in fp32
                #  - run init method on it
                #  - shard result to get only a specific shard
                # Instead I'm lazy and just going to run init_method, since they are scalar independent
                assert {"weight"} == {name for name, _ in module.named_parameters()} or {"weight"} == {
                    name for name, _ in module.named_parameters()
                }
                for param_name, param in module.named_parameters():
                    assert isinstance(param, NanotronParameter)
                    if param.is_tied:
                        tied_info = param.get_tied_info()
                        full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                            module_id_to_prefix=module_id_to_prefix
                        )
                    else:
                        full_param_name = f"{module_name}.{param_name}"

                    if full_param_name in initialized_parameters:
                        # Already initialized
                        continue

                    if "weight" == param_name:
                        init_method_(param)
                    elif "bias" == param_name:
                        param.zero_()
                    else:
                        raise ValueError(f"Who the fuck is {param_name}?")

                    assert full_param_name not in initialized_parameters
                    initialized_parameters.add(full_param_name)
            elif isinstance(module, TensorParallelRowLinear):
                # Somehow Megatron-LM does something super complicated, https://github.com/NVIDIA/Megatron-LM/blob/2360d732a399dd818d40cbe32828f65b260dee11/megatron/core/tensor_parallel/layers.py#L96
                # What it does:
                #  - instantiate a buffer of the `full size` in fp32
                #  - run init method on it
                #  - shard result to get only a specific shard
                # Instead I'm lazy and just going to run init_method, since they are scalar independent
                assert {"weight"} == {name for name, _ in module.named_parameters()} or {"weight"} == {
                    name for name, _ in module.named_parameters()
                }
                for param_name, param in module.named_parameters():
                    assert isinstance(param, NanotronParameter)
                    if param.is_tied:
                        tied_info = param.get_tied_info()
                        full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                            module_id_to_prefix=module_id_to_prefix
                        )
                    else:
                        full_param_name = f"{module_name}.{param_name}"

                    if full_param_name in initialized_parameters:
                        # Already initialized
                        continue

                    if "weight" == param_name:
                        scaled_init_method_(param)
                    elif "bias" == param_name:
                        param.zero_()
                    else:
                        raise ValueError(f"Who the fuck is {param_name}?")

                    assert full_param_name not in initialized_parameters
                    initialized_parameters.add(full_param_name)
            elif isinstance(module, RMSNorm):
                assert {"weight"} == {name for name, _ in module.named_parameters()}
                for param_name, param in module.named_parameters():
                    assert isinstance(param, NanotronParameter)
                    if param.is_tied:
                        tied_info = param.get_tied_info()
                        full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                            module_id_to_prefix=module_id_to_prefix
                        )
                    else:
                        full_param_name = f"{module_name}.{param_name}"

                    if full_param_name in initialized_parameters:
                        # Already initialized
                        continue

                    if "weight" == param_name:
                        # TODO @thomasw21: Sometimes we actually want 0
                        param.fill_(1)
                    elif "bias" == param_name:
                        param.zero_()
                    else:
                        raise ValueError(f"Who the fuck is {param_name}?")

                    assert full_param_name not in initialized_parameters
                    initialized_parameters.add(full_param_name)
            elif isinstance(module, TensorParallelEmbedding):
                # TODO @thomasw21: Handle tied embeddings
                # Somehow Megatron-LM does something super complicated, https://github.com/NVIDIA/Megatron-LM/blob/2360d732a399dd818d40cbe32828f65b260dee11/megatron/core/tensor_parallel/layers.py#L96
                # What it does:
                #  - instantiate a buffer of the `full size` in fp32
                #  - run init method on it
                #  - shard result to get only a specific shard
                # Instead I'm lazy and just going to run init_method, since they are scalar independent
                assert {"weight"} == {name for name, _ in module.named_parameters()}

                assert isinstance(module.weight, NanotronParameter)
                if module.weight.is_tied:
                    tied_info = module.weight.get_tied_info()
                    full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                        module_id_to_prefix=module_id_to_prefix
                    )
                else:
                    full_param_name = f"{module_name}.weight"

                if full_param_name in initialized_parameters:
                    # Already initialized
                    continue

                init_method_(module.weight)
                assert full_param_name not in initialized_parameters
                initialized_parameters.add(full_param_name)

    assert initialized_parameters == {
        param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
        if param.is_tied
        else name
        for name, param in model.named_parameters()
    }, f"Somehow the initialized set of parameters don't match:\n - Expected: { {name for name, _ in model.named_parameters()} }\n - Got: {initialized_parameters}"

    # Synchronize parameters so that the model is consistent
    for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
        # sync across dp
        dist.all_reduce(param, op=dist.ReduceOp.AVG, group=dpg.dp_pg)

    for (_, group_ranks), param in sorted(
        get_tied_id_to_param(
            parameters=model.parameters(),
            root_module=model.module if isinstance(model, DistributedDataParallel) else model,
        ).items(),
        key=lambda x: x[0],
    ):
        group = dpg.world_ranks_to_pg[group_ranks]
        dist.all_reduce(param, op=dist.ReduceOp.AVG, group=group)

def init_optimizer_and_grad_accumulator(
    model: nn.Module, optimizer_args: OptimizerArgs, dpg: DistributedProcessGroups
) -> Tuple[BaseOptimizer, GradientAccumulator]:
    # Normalize DDP
    normalized_model = model.module if isinstance(model, DistributedDataParallel) else model

    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in normalized_model.named_modules()}
    # Fix the root_model
    root_model_id = id(normalized_model)
    module_id_to_prefix[root_model_id] = ""

    # named parameters
    named_parameters = [
        (
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            if param.is_tied
            else name,
            param,
        )
        for name, param in normalized_model.named_parameters()
    ]

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
                    id(param): param.get_tied_info().get_full_name_from_module_id_to_prefix(
                        module_id_to_prefix=module_id_to_prefix
                    )
                    if param.is_tied
                    else name
                    for name, param in normalized_model.named_parameters()
                },
            ),
            hook=get_fp32_accum_hook(
                reduce_scatter=optimizer.inherit_from(ZeroDistributedOptimizer), reduce_op=dist.ReduceOp.AVG
            ),
        )

    return optimizer, grad_accumulator


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

def dummy_infinite_data_generator(
    micro_batch_size: int,
    sequence_length: int,
    input_pp_rank: int,
    output_pp_rank: int,
    vocab_size: int,
    seed: int,
    dpg: DistributedProcessGroups,
):
    def dummy_infinite_data_generator() -> Generator[Dict[str, Union[torch.Tensor, TensorPointer]], None, None]:
        # Random generator
        generator = torch.Generator(device="cuda")
        # Make sure that TP are synced always
        generator.manual_seed(seed * (1 + dist.get_rank(dpg.dp_pg)) * (1 + dist.get_rank(dpg.pp_pg)))

        while True:
            yield {
                "input_ids": torch.randint(
                    0,
                    vocab_size,
                    (micro_batch_size, sequence_length),
                    dtype=torch.long,
                    device="cuda",
                    generator=generator,
                )
                if dist.get_rank(dpg.pp_pg) == input_pp_rank
                else TensorPointer(group_rank=input_pp_rank),
                "input_mask": torch.ones(
                    micro_batch_size,
                    sequence_length,
                    dtype=torch.bool,
                    device="cuda",
                )
                if dist.get_rank(dpg.pp_pg) == input_pp_rank
                else TensorPointer(group_rank=input_pp_rank),
                "label_ids": torch.randint(
                    0,
                    vocab_size,
                    (micro_batch_size, sequence_length),
                    dtype=torch.long,
                    device="cuda",
                    generator=generator,
                )
                if dist.get_rank(dpg.pp_pg) == output_pp_rank
                else TensorPointer(group_rank=output_pp_rank),
                "label_mask": torch.ones(
                    micro_batch_size,
                    sequence_length,
                    dtype=torch.bool,
                    device="cuda",
                )
                if dist.get_rank(dpg.pp_pg) == output_pp_rank
                else TensorPointer(group_rank=output_pp_rank),
            }

    return dummy_infinite_data_generator

def save_checkpoint(
    model,
    optimizer,
    lr_scheduler,
    random_states: RandomStates,
    model_config: LlamaConfig,
    config: Config,
    iteration_step: int,
    consumed_train_samples: int,
    checkpoints_path: xPath,
    dpg: DistributedProcessGroups,
    logger: lg.Logger,
) -> xPath:
    checkpoint_path = checkpoints_path / f"{iteration_step}"
    if check_path_is_local(checkpoint_path):
        if dist.get_rank(dpg.world_pg) == 0:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        dist.barrier(dpg.world_pg)
    log_rank(f"Saving checkpoint at {checkpoint_path}", logger=logger, level=logging.INFO, rank=0)
    checkpoint_metadata = {
        "last_train_step": iteration_step,
        # TODO: @nouamanetazi: Add more metadata to the checkpoint to be able to resume dataloader states properly
        "consumed_train_samples": consumed_train_samples,
    }
    save(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        dpg=dpg,
        root_folder=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        should_save_config=False,
    )
    save_random_states(random_states=random_states, dpg=dpg, root_folder=checkpoint_path)
    with fs_open(checkpoints_path / "latest.txt", mode="w") as fo:
        fo.write(f"{iteration_step}")
    with fs_open(checkpoint_path / "config.txt", mode="w") as fo:
        # TODO @nouamane: save as yaml
        fo.write(pformat(config))
    model_config.to_json_file(checkpoint_path / "model_config.json")
    return checkpoint_path

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

