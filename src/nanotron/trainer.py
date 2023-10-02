import contextlib
import dataclasses
import datetime
import json
import os
import sys
import time
from pprint import pformat
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets.download.streaming_download_manager import xPath
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig

from nanotron.clip_grads import clip_grad_norm
from nanotron.config import (
    Config,
    ExistingCheckpointInit,
    HubLoggerConfig,
    ParallelismArgs,
    RandomInit,
    TensorboardLoggerConfig,
    get_args_from_path,
)
from nanotron.core import distributed as dist
from nanotron.core import logging
from nanotron.core.logging import log_rank
from nanotron.core.optimizer.zero import ZeroDistributedOptimizer
from nanotron.core.parallelism.data_parallelism.utils import sync_gradients_across_dp
from nanotron.core.parallelism.parameters import NanotronParameter, sanity_check
from nanotron.core.parallelism.pipeline_parallelism.block import PipelineBlock
from nanotron.core.parallelism.pipeline_parallelism.engine import (
    PipelineEngine,
)
from nanotron.core.parallelism.pipeline_parallelism.tensor_pointer import TensorPointer
from nanotron.core.parallelism.pipeline_parallelism.utils import get_pp_rank_of
from nanotron.core.parallelism.tensor_parallelism.nn import (
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)
from nanotron.core.parallelism.tied_parameters import (
    create_pg_for_tied_weights,
    get_tied_id_to_param,
    sync_tied_weights_gradients,
    tie_parameters,
)
from nanotron.core.process_groups_initializer import DistributedProcessGroups, get_process_groups
from nanotron.core.random import (
    set_random_seed,
)
from nanotron.core.serialize import (
    load_lr_scheduler,
    load_meta,
    load_optimizer,
    load_weights,
    save,
    save_random_states,
)
from nanotron.core.serialize.path import check_path_is_local, parse_ckpt_path
from nanotron.core.serialize.serialize import fs_open
from nanotron.core.tensor_init import init_method_normal, scaled_init_method_normal
from nanotron.core.utils import (
    assert_tensor_synced_across_pg,
    init_on_device_and_dtype,
)
from nanotron.dataloaders.dataloader import sanity_check_dataloader
from nanotron.helpers import (
    _vocab_size_with_padding,
    get_profiler,
    init_optimizer_and_grad_accumulator,
    init_random_states,
    lr_scheduler_builder,
    set_logger_verbosity,
)
from nanotron.logger import LoggerWriter, LogItem
from nanotron.models import NanotronModel

if int(os.environ.get("USE_FAST", 0)) == 1:
    # We import the fast versions
    from nanotron.models.fast.falcon import FalconForTraining
    from nanotron.models.fast.gpt2 import GPTForTraining
    from nanotron.models.fast.llama import LlamaForTraining, RotaryEmbedding
else:
    from nanotron.models.falcon import FalconForTraining
    from nanotron.models.gpt2 import GPTForTraining
    from nanotron.models.llama import LlamaForTraining, RotaryEmbedding


logger = logging.get_logger(__name__)

try:
    from nanotron.logger import BatchSummaryWriter

    tb_logger_available = True
except ImportError:
    tb_logger_available = False

try:
    from nanotron.logger import HubSummaryWriter

    hub_logger_available = True
except ImportError:
    hub_logger_available = False

CONFIG_TO_MODEL_CLASS = {
    "LlamaConfig": LlamaForTraining,
    "GPTBigCodeConfig": GPTForTraining,
    "FalconConfig": FalconForTraining,
    "RWConfig": FalconForTraining,
}


# TODO @nouamane: add abstract class
class DistributedTrainer:
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Initialise all process groups
        self.dpg = get_process_groups(
            data_parallel_size=self.config.parallelism.dp,
            pipeline_parallel_size=self.config.parallelism.pp,
            tensor_parallel_size=self.config.parallelism.tp,
        )

        # Do a first NCCL sync to warmup and try to avoid Timeout after model/data loading
        test_tensor = torch.tensor([dist.get_rank(self.dpg.world_pg)], device=torch.device("cuda"))
        test_tensor_list = [torch.zeros_like(test_tensor) for _ in range(self.dpg.world_pg.size())]
        dist.all_gather(test_tensor_list, test_tensor, group=self.dpg.world_pg, async_op=False)
        dist.barrier()
        log_rank(
            f"Test NCCL sync for ranks {[t.item() for t in test_tensor_list]}",
            logger=logger,
            level=logging.INFO,
            group=self.dpg.dp_pg,
            rank=0,
        )

        # Set random states
        set_random_seed(self.config.model.seed)

        # Set log levels
        if dist.get_rank(self.dpg.world_pg) == 0:
            if self.config.logging.log_level is not None:
                set_logger_verbosity(self.config.logging.log_level, dpg=self.dpg)
        else:
            if self.config.logging.log_level_replica is not None:
                set_logger_verbosity(self.config.logging.log_level_replica, dpg=self.dpg)

        # Parsing checkpoint path
        checkpoint_path = parse_ckpt_path(config=self.config)

        # Init model and build on pp ranks
        self.random_states = init_random_states(parallel_config=self.config.parallelism, tp_pg=self.dpg.tp_pg)
        self.init_model(checkpoint_path=checkpoint_path)  # Defines self.model and self.model_config

        # Init optimizer
        self.optimizer, self.grad_accumulator = init_optimizer_and_grad_accumulator(
            model=self.model, optimizer_args=self.config.optimizer, dpg=self.dpg
        )
        if checkpoint_path is not None:
            load_optimizer(optimizer=self.optimizer, dpg=self.dpg, root_folder=checkpoint_path)

        # Init learning rate scheduler
        self.lr_scheduler = lr_scheduler_builder(
            optimizer=self.optimizer,
            learning_rate=self.config.optimizer.learning_rate,
            lr_scheduler_args=self.config.learning_rate_scheduler,
        )
        if checkpoint_path is not None:
            load_lr_scheduler(
                lr_scheduler=self.lr_scheduler,
                dpg=self.dpg,
                root_folder=checkpoint_path,
                is_zero=self.optimizer.inherit_from(ZeroDistributedOptimizer),
            )

        # Define iteration start state
        self.start_iteration_step: int
        self.consumed_train_samples: int
        if checkpoint_path is not None:
            checkpoint_metadata = load_meta(dpg=self.dpg, root_folder=checkpoint_path)
            log_rank(str(checkpoint_metadata), logger=logger, level=logging.INFO, rank=0)
            self.start_iteration_step = checkpoint_metadata.metas["last_train_step"]
            self.consumed_train_samples = checkpoint_metadata.metas["consumed_train_samples"]
            assert (
                self.config.tokens.train_steps > self.start_iteration_step
            ), f"Loaded checkpoint has already trained {self.start_iteration_step} batches, you need to specify a higher `config.tokens.train_steps`"
        else:
            self.start_iteration_step = 0
            self.consumed_train_samples = 0

        # Setup log writers on output rank
        self.logger_ranks = self.dpg.world_rank_matrix[self.model.output_pp_rank, 0, 0].flatten()
        self.tb_context, self.loggerwriter = self.setup_log_writers(
            config=self.config, logger_ranks=self.logger_ranks, dpg=self.dpg
        )

        # Log where each module is instantiated
        self.model.log_modules(level=logging.DEBUG, group=self.dpg.world_pg, rank=0)

        dist.barrier()
        log_rank(
            f"Global rank: { dist.get_rank(self.dpg.world_pg)}/{self.dpg.world_pg.size()} | PP: {dist.get_rank(self.dpg.pp_pg)}/{self.dpg.pp_pg.size()} | DP: {dist.get_rank(self.dpg.dp_pg)}/{self.dpg.dp_pg.size()} | TP: {dist.get_rank(self.dpg.tp_pg)}/{self.dpg.tp_pg.size()}",
            logger=logger,
            level=logging.INFO,
        )
        dist.barrier()

        # Dummy hyper parameter
        self.micro_batch_size = self.config.tokens.micro_batch_size
        self.n_micro_batches_per_batch = self.config.tokens.batch_accumulation_per_replica
        self.global_batch_size = self.micro_batch_size * self.n_micro_batches_per_batch * self.dpg.dp_pg.size()
        self.sequence_length = self.config.tokens.sequence_length

    @classmethod
    def from_config_file(cls, config_file: str):
        config = get_args_from_path(config_file)
        return cls(config=config)

    def train(self, dataloader: Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]) -> None:
        dataloader = sanity_check_dataloader(dataloader=dataloader, dpg=self.dpg, config=self.config)
        self.pipeline_engine: PipelineEngine = self.config.parallelism.pp_engine

        self.pipeline_engine.nb_microbatches = self.n_micro_batches_per_batch

        log_rank(
            f"[Before the start of training] datetime: {datetime.datetime.now()}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        # Kill switch
        self.check_kill_switch(save_ckpt=False)

        # TODO @nouamanetazi: refactor this
        # Useful mapping
        self.normalized_model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        self.module_id_to_prefix = {
            id(module): f"{module_name}." for module_name, module in self.normalized_model.named_modules()
        }
        # Fix the root_model
        self.module_id_to_prefix[id(self.normalized_model)] = ""

        prof = get_profiler(config=self.config)
        with self.tb_context as tb_writer:
            with prof:
                for self.iteration_step in range(self.start_iteration_step + 1, self.config.tokens.train_steps + 1):
                    if isinstance(prof, torch.profiler.profile):
                        prof.step()
                    self.iteration_start_time = time.time()

                    # Training step
                    outputs = self.training_step(dataloader=dataloader)

                    # Training Logs
                    self.consumed_train_samples += self.global_batch_size

                    if self.iteration_step % self.config.logging.iteration_step_info_interval == 0:
                        self.train_step_logs(tb_writer=tb_writer, outputs=outputs)

                    # Kill switch
                    self.check_kill_switch(save_ckpt=True)

                    # Checkpoint
                    if self.iteration_step % self.config.checkpoints.checkpoint_interval == 0:
                        self.save_checkpoint()

                    # Push to Hub
                    if (
                        isinstance(self.config.logging.tensorboard_logger, HubLoggerConfig)
                        and isinstance(tb_writer, HubSummaryWriter)
                        and (self.iteration_step - 1) % self.config.logging.tensorboard_logger.push_to_hub_interval
                        == 0
                    ):
                        # tb_writer only exists on a single rank
                        log_rank(
                            f"Push Tensorboard logging to Hub at iteration {self.iteration_step} to https://huggingface.co/{self.config.logging.tensorboard_logger.repo_id}/tensorboard",
                            logger=logger,
                            level=logging.INFO,
                        )
                        # it is a future that queues to avoid concurrent push
                        tb_writer.scheduler.trigger()

    def training_step(self, dataloader: Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]) -> Iterable[Dict]:
        self.before_tbi_sanity_checks()

        if self.iteration_step < 5:
            log_rank(
                f"[Before train batch iter] Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MB. Peak reserved memory: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MB",
                logger=logger,
                level=logging.INFO,
                group=self.dpg.world_pg,
                rank=0,
            )

        outputs = self.pipeline_engine.train_batch_iter(
            model=self.model,
            pg=self.dpg.pp_pg,
            batch=(next(dataloader) for _ in range(self.n_micro_batches_per_batch)),
            grad_accumulator=self.grad_accumulator,
        )

        if self.iteration_step < 5:
            log_rank(
                f"[After train batch iter] Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MB. Peak reserved memory: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MB",
                logger=logger,
                level=logging.INFO,
                group=self.dpg.world_pg,
                rank=0,
            )

        self.after_tbi_sanity_checks()

        # Sync tied weights
        # TODO @nouamane: Put this in hooks so we can overlap communication with gradient computation on the last backward pass.
        sync_tied_weights_gradients(
            module=self.normalized_model,
            dpg=self.dpg,
            grad_accumulator=self.grad_accumulator,
        )
        if not isinstance(self.model, DistributedDataParallel):
            # Manually sync across DP if it's not handled by DDP
            sync_gradients_across_dp(
                module=self.model,
                dp_pg=self.dpg.dp_pg,
                reduce_op=dist.ReduceOp.AVG,
                # TODO @thomasw21: This is too memory hungry, instead we run all_reduce
                reduce_scatter=False,  # optimizer.inherit_from(ZeroDistributedOptimizer),
                grad_accumulator=self.grad_accumulator,
            )

        # Clip gradients
        if self.config.optimizer.clip_grad is not None:
            # Normalize DDP
            named_parameters = [
                (
                    param.get_tied_info().get_full_name_from_module_id_to_prefix(
                        module_id_to_prefix=self.module_id_to_prefix
                    )
                    if param.is_tied
                    else name,
                    param,
                )
                for name, param in self.normalized_model.named_parameters()
                if param.requires_grad
            ]
            # TODO @nouamane: we need to split `world_rank_matrix` along PP axis, to separate ref from active model
            self.grad_norm_unclipped = clip_grad_norm(
                mp_pg=self.dpg.world_ranks_to_pg[
                    tuple(sorted(self.dpg.world_rank_matrix[:, dist.get_rank(self.dpg.dp_pg), :].reshape(-1)))
                ],
                named_parameters=named_parameters,
                grad_accumulator=self.grad_accumulator,
                max_norm=self.config.optimizer.clip_grad,
            )

        self.before_optim_step_sanity_checks()
        # Apply gradient
        self.optimizer.step()
        # PT 2.0: will change default to None as it gains performance.
        # https://github.com/pytorch/pytorch/issues/92656
        self.optimizer.zero_grad(set_to_none=True)

        # Update the learning rate
        self.lr_scheduler.step()

        self.after_optim_step_sanity_checks()

        return outputs

    def train_step_logs(self, tb_writer, outputs: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]]) -> None:
        # TODO @nouamanetazi: Megatron-LM seems to be using a barrier to report their interval time. Check if this is necessary. https://github.com/NouamaneTazi/Megatron-LM/blob/e241a96c3085b18e36c6cee1d68a8155de77b5a6/megatron/training.py#L607
        dist.barrier()
        torch.cuda.synchronize()
        elapsed_time_per_iteration_ms = (time.time() - self.iteration_start_time) * 1000
        tokens_per_sec = (
            self.global_batch_size * self.sequence_length / (elapsed_time_per_iteration_ms / 1000)
        )  # tokens_per_sec is calculated using sequence_length
        model_tflops, hardware_tflops = self.model.get_flops_per_sec(
            iteration_time_in_sec=elapsed_time_per_iteration_ms / 1000,
            sequence_length=self.sequence_length,
            global_batch_size=self.global_batch_size,
        )

        if dist.get_rank(self.dpg.world_pg) in self.logger_ranks:
            assert self.loggerwriter is not None, "loggerwriter should be defined on logger ranks"
            # This is an average on only one data rank.
            loss_avg = torch.stack(
                [output["loss"] for output in outputs]
            ).sum()  # already divided by n_micro_batches_per_batch
            # sync loss across DP
            dist.all_reduce(loss_avg, group=self.dpg.dp_pg, async_op=False, op=dist.ReduceOp.AVG)

            lr = self.lr_scheduler.get_last_lr()[0]

            log_entries = [
                LogItem("consumed_samples", self.consumed_train_samples, "12d"),
                LogItem("elapsed_time_per_iteration_ms", elapsed_time_per_iteration_ms, ".1f"),
                LogItem("tokens_per_sec", tokens_per_sec, "1.6E"),
                LogItem("tokens_per_sec_per_gpu", tokens_per_sec / self.dpg.world_pg.size(), "1.6E"),
                LogItem("global_batch_size", self.global_batch_size, "5d"),
                LogItem("lm_loss", loss_avg.item(), "1.6E"),
                LogItem("lr", lr, ".3E"),
                LogItem("model_tflops_per_gpu", model_tflops, ".2f"),
                LogItem("hardware_tflops_per_gpu", hardware_tflops, ".2f"),
            ]

            if self.config.optimizer.clip_grad is not None:
                log_entries.append(LogItem("grad_norm", self.grad_norm_unclipped.item(), ".3f"))

            if tb_writer is not None:
                tb_writer.add_scalars_from_list(log_entries, self.iteration_step)

                # Log config to tensorboard
                def flatten_dict(nested, sep="/"):
                    """Flatten dictionary and concatenate nested keys with separator."""

                    def rec(nest, prefix, into):
                        for k, v in nest.items():
                            if sep in k:
                                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
                            if isinstance(v, dict):
                                rec(v, prefix + k + sep, into)
                            else:
                                into[prefix + k] = v

                    flat = {}
                    rec(nested, "", flat)
                    return flat

                def config_to_md(config_dict):
                    config_markdown = "| Key | Value |\n| --- | --- |\n"
                    for key, value in config_dict.items():
                        config_markdown += f"| {key} | {value} |\n"
                    return config_markdown

                if self.iteration_step == 1 and hasattr(tb_writer, "add_text"):
                    config_markdown = config_to_md(flatten_dict(dataclasses.asdict(self.config)))
                    tb_writer.add_text("config", config_markdown, 1)
                    model_config_dict = json.loads(self.model_config.to_json_string())
                    tb_writer.add_text("model_config", config_to_md(flatten_dict(model_config_dict)), 1)

            self.loggerwriter.add_scalars_from_list(log_entries, self.iteration_step)

        elif isinstance(outputs[0]["loss"], torch.Tensor):
            # This is an average on only one data rank.
            loss_avg = torch.stack(
                [output["loss"] for output in outputs]
            ).sum()  # already divided by n_micro_batches_per_batch
            # sync loss across DP
            dist.all_reduce(loss_avg, group=self.dpg.dp_pg, async_op=False, op=dist.ReduceOp.AVG)

    @staticmethod
    def build_model(
        model_config: AutoConfig,
        model_builder: Callable[[], NanotronModel],
        dpg: DistributedProcessGroups,
        dtype: torch.dtype,
        target_pp_ranks: Optional[List[int]] = None,
        device: Optional[torch.device] = torch.device("cuda"),
    ) -> NanotronModel:
        # TODO: classes dont take same args
        model: NanotronModel = model_builder()

        # If no target pp ranks are specified, we assume that we want to use all pp ranks
        if target_pp_ranks is None:
            pp_size = dpg.pp_pg.size()
            target_pp_ranks = list(range(pp_size))
        else:
            pp_size = len(target_pp_ranks)

        # Set rank for each pipeline block
        pipeline_blocks = [module for name, module in model.named_modules() if isinstance(module, PipelineBlock)]
        # "cuda" is already defaulted for each process to it's own cuda device
        with init_on_device_and_dtype(device=device, dtype=dtype):
            # TODO: https://github.com/huggingface/nanotron/issues/65

            # Balance compute across PP blocks
            block_compute_costs = model.get_block_compute_costs()
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

            model.input_pp_rank = target_pp_ranks[0]
            model.output_pp_rank = target_pp_ranks[target_pp_rank_idx]

        return model

    def init_model(self, checkpoint_path: Optional[xPath]) -> None:
        trust_remote_code = (
            self.config.model.remote_code.trust_remote_code if hasattr(self.config.model, "remote_code") else None
        )
        model_config = AutoConfig.from_pretrained(self.config.model.hf_model_name, trust_remote_code=trust_remote_code)

        # TODO(thomwolf): hardcoded value for debugging
        # model_config.num_hidden_layers = 2

        model_config.vocab_size = _vocab_size_with_padding(
            model_config.vocab_size,
            pg_size=self.dpg.tp_pg.size(),
            make_vocab_size_divisible_by=self.config.model.make_vocab_size_divisible_by,
        )

        # TODO: add max_position_embeddings
        if hasattr(model_config, "max_position_embeddings"):
            assert (
                model_config.max_position_embeddings >= self.config.tokens.sequence_length
            ), f"max_position_embeddings ({model_config.max_position_embeddings}) must be >= sequence_length ({self.config.tokens.sequence_length})"

        log_rank(pformat(self.config), logger=logger, level=logging.INFO, rank=0)
        log_rank(str(model_config), logger=logger, level=logging.INFO, rank=0)

        model_config_cls = model_config.__class__.__name__
        assert (
            model_config_cls in CONFIG_TO_MODEL_CLASS
        ), f"Unsupported model config {model_config_cls}. Only {CONFIG_TO_MODEL_CLASS.keys()} are supported"

        self.model = self._init_model(
            model_config=model_config,
            model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config_cls](
                config=model_config,
                dpg=self.dpg,
                parallel_config=self.config.parallelism,
                random_states=self.random_states,
            ),
        )
        self.model_config = model_config

        # Load or initialize model weights
        if checkpoint_path is not None:
            # Load from checkpoint
            log_rank(
                f"Resuming training from checkpoint: {checkpoint_path}",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
            load_weights(model=self.model, dpg=self.dpg, root_folder=checkpoint_path)
        else:
            # We initialize the model.
            if isinstance(self.config.model.init_method, ExistingCheckpointInit):
                # Initialize model from an existing model checkpoint
                load_weights(model=self.model, dpg=self.dpg, root_folder=self.config.model.init_method.path)
            elif isinstance(self.config.model.init_method, RandomInit):
                # Initialize model randomly
                self.model.init_model_randomly(
                    init_method=init_method_normal(self.config.model.init_method.std),
                    scaled_init_method=scaled_init_method_normal(
                        self.config.model.init_method.std, self.model_config.num_hidden_layers
                    ),
                )
            else:
                raise ValueError(f"Unsupported {self.config.model.init_method}")

    def _init_model(
        self,
        model_config: AutoConfig,
        model_builder: Callable[[], NanotronModel],
        target_pp_ranks: Optional[List[int]] = None,
    ) -> Tuple[NanotronModel]:
        config = self.config
        dpg = self.dpg

        parallel_config = config.parallelism
        make_ddp = not (config.optimizer.accumulate_grad_in_fp32 and config.optimizer.zero_stage > 0)

        # Build model and set pp ranks
        model = self.build_model(
            model_config=model_config,
            dpg=dpg,
            dtype=config.model.dtype,
            target_pp_ranks=target_pp_ranks,
            model_builder=model_builder,
        )

        # Initialize rotary embeddings
        for module in model.modules():
            if not isinstance(module, RotaryEmbedding):
                continue
            module.init_rotary_embeddings()

        # Mark some parameters as tied
        mark_tied_parameters(model=model, dpg=dpg, parallel_config=parallel_config)

        # count number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        size_params = sum(p.numel() * p.element_size() for p in model.parameters())

        # TODO @nouamanetazi: better memory logs
        log_rank(
            f"Number of parameters: {num_params} ({size_params / 1024**2:.2f}MB)",
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

        # Sanity check the model, all parameters must be NanotronParameter (either tied or sharded)
        sanity_check(root_module=model)

        return model

    @staticmethod
    def setup_log_writers(config: Config, logger_ranks: Iterable[int], dpg: DistributedProcessGroups):
        """Setup all log writers on the appropriate ranks

        Args:
            config (Config): The config object
            logger_ranks (Iterable[int]): The ranks that should log
            dpg (DistributedProcessGroups): The distributed process groups
        """
        if dist.get_rank(dpg.world_pg) in logger_ranks:
            if config.logging.tensorboard_logger is None:
                tb_context = contextlib.nullcontext()
            else:
                current_time = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                logdir = str(
                    config.logging.tensorboard_logger.tensorboard_dir / f"{config.general.name}_{current_time}"
                )

                if isinstance(config.logging.tensorboard_logger, HubLoggerConfig):
                    assert (
                        hub_logger_available
                    ), 'Hub Tensorboard Logger is not available. Please install nanotron with `pip install -e ".[hf-logger]"` or modify your config file'
                    tb_context = HubSummaryWriter(
                        logdir=logdir,
                        repo_id=config.logging.tensorboard_logger.repo_id,
                        repo_private=not config.logging.tensorboard_logger.repo_public,
                        path_in_repo=f"tensorboard/{config.general.name}_{current_time}_{dpg.dp_pg.size()}_{dpg.pp_pg.size()}_{dpg.tp_pg.size()}",
                    )
                elif isinstance(config.logging.tensorboard_logger, TensorboardLoggerConfig):
                    assert (
                        tb_logger_available
                    ), 'Tensorboard Logger is not available. Please install nanotron with `pip install -e ".[tb-logger]"` or modify your config file'
                    tb_context = BatchSummaryWriter(logdir=logdir)
                else:
                    raise ValueError(
                        f"Unsupported value for `config.logging.tensorboard_logger`, got {config.logging.tensorboard_logger}"
                    )

            loggerwriter = LoggerWriter(global_step=config.tokens.train_steps)
        else:
            tb_context = contextlib.nullcontext()
            loggerwriter = None

        return tb_context, loggerwriter

    def check_kill_switch(self, save_ckpt: bool):
        if self.config.general.kill_switch_path.exists():
            log_rank(
                f"Detected kill switch at {self.config.general.kill_switch_path}. Exiting",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

            # Save checkpoint
            if save_ckpt:
                self.save_checkpoint()

            # TODO @thomasw21: Do I need to return a barrier in order to be sure everyone saved before exiting.
            sys.exit(0)

    def save_checkpoint(self) -> xPath:
        checkpoints_path = self.config.checkpoints.checkpoints_path
        checkpoint_path = checkpoints_path / f"{self.iteration_step}"
        if check_path_is_local(checkpoint_path):
            if dist.get_rank(self.dpg.world_pg) == 0:
                checkpoint_path.mkdir(parents=True, exist_ok=True)
            dist.barrier(self.dpg.world_pg)
        log_rank(f"Saving checkpoint at {checkpoint_path}", logger=logger, level=logging.INFO, rank=0)
        checkpoint_metadata = {
            "last_train_step": self.iteration_step,
            # TODO: @nouamanetazi: Add more metadata to the checkpoint to be able to resume dataloader states properly
            "consumed_train_samples": self.consumed_train_samples,
        }
        save(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            dpg=self.dpg,
            root_folder=checkpoint_path,
            checkpoint_metadata=checkpoint_metadata,
        )
        save_random_states(random_states=self.random_states, dpg=self.dpg, root_folder=checkpoint_path)
        with fs_open(checkpoints_path / "latest.txt", mode="w") as fo:
            fo.write(f"{self.iteration_step}")
        with fs_open(checkpoint_path / "config.txt", mode="w") as fo:
            # TODO @nouamane: save as yaml
            fo.write(pformat(self.config))
        self.model_config.to_json_file(checkpoint_path / "model_config.json")
        return checkpoint_path

    def before_tbi_sanity_checks(self) -> None:
        if not self.config.general.ignore_sanity_checks:
            # SANITY CHECK: Check that the model params are synchronized across dp
            for name, param in sorted(self.model.named_parameters(), key=lambda x: x[0]):
                assert_tensor_synced_across_pg(
                    tensor=param, pg=self.dpg.dp_pg, msg=lambda err: f"{name} are not synchronized across DP {err}"
                )

            # SANITY CHECK: Tied weights are synchronized
            tied_params_list = sorted(
                get_tied_id_to_param(
                    parameters=self.normalized_model.parameters(),
                    root_module=self.normalized_model,
                ).items(),
                key=lambda x: x[0],
            )
            for (name, group_ranks), param in tied_params_list:
                group = self.dpg.world_ranks_to_pg[group_ranks]
                assert_tensor_synced_across_pg(
                    tensor=param,
                    pg=group,
                    msg=lambda err: f"[Before train] Tied weights {name} are not synchronized. {err}",
                )

            # SANITY CHECK: Check that the grad accumulator buffers are ready for DDP
            if self.grad_accumulator is not None:
                for _, elt in self.grad_accumulator.fp32_grad_buffers.items():
                    fp32_grad_buffer = elt["fp32_grad"]
                    torch.testing.assert_close(
                        fp32_grad_buffer,
                        torch.zeros_like(fp32_grad_buffer),
                        atol=0,
                        rtol=0,
                        msg="Grad accumulator buffers must be zeroed in first accumulation step.",
                    )

    def after_tbi_sanity_checks(self) -> None:
        if not self.config.general.ignore_sanity_checks:
            # SANITY CHECK: Check that gradient flow on the entire model
            # SANITY CHECK: Check that all parameters that required gradients, have actually a gradient
            # SANITY CHECK: Check for nan/inf
            for name, param in self.normalized_model.named_parameters():
                if not param.requires_grad:
                    continue

                if param.is_tied:
                    tied_info = param.get_tied_info()
                    name = tied_info.get_full_name_from_module_id_to_prefix(
                        module_id_to_prefix=self.module_id_to_prefix
                    )

                if self.grad_accumulator is not None:
                    grad = self.grad_accumulator.get_grad_buffer(name=name)
                else:
                    grad = param.grad

                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    raise ValueError("Gradient is nan or inf")
                if grad is None:
                    log_rank(
                        f"Process rank { dist.get_rank(self.dpg.world_pg)}/{self.dpg.world_pg.size()}: {name} is missing gradient",
                        logger=logger,
                        level=logging.ERROR,
                    )

    def before_optim_step_sanity_checks(self) -> None:
        if not self.config.general.ignore_sanity_checks:
            # SANITY CHECK: Test tied weights gradients are synchronized
            for (name, group_ranks), param in sorted(
                get_tied_id_to_param(
                    parameters=self.normalized_model.parameters(), root_module=self.normalized_model
                ).items(),
                key=lambda x: x[0],
            ):
                if not param.requires_grad:
                    continue

                if self.grad_accumulator is not None:
                    grad = self.grad_accumulator.get_grad_buffer(name=name)
                else:
                    grad = param.grad

                assert grad is not None, f"Grad is None for {name}"
                group = self.dpg.world_ranks_to_pg[group_ranks]
                assert_tensor_synced_across_pg(
                    tensor=grad,
                    pg=group,
                    msg=lambda err: f"[Before optimizer step] Tied weights grads for {name} are not synchronized. {err}",
                )

            # SANITY CHECK: Test gradients are synchronized across DP
            for name, param in sorted(self.normalized_model.named_parameters(), key=lambda x: x[0]):
                if not param.requires_grad:
                    continue

                if param.is_tied:
                    tied_info = param.get_tied_info()
                    name = tied_info.get_full_name_from_module_id_to_prefix(
                        module_id_to_prefix=self.module_id_to_prefix
                    )

                if self.grad_accumulator is not None:
                    grad = self.grad_accumulator.get_grad_buffer(name=name)
                else:
                    grad = param.grad

                assert grad is not None, f"Grad is None for {name}"
                assert_tensor_synced_across_pg(
                    tensor=grad,
                    pg=self.dpg.dp_pg,
                    msg=lambda err: f"[Before optimizer step] weights grads for {name} are not synchronized across DP. {err}",
                )

            # SANITY CHECK: Check that the model params are synchronized across dp
            for name, param in sorted(self.model.named_parameters(), key=lambda x: x[0]):
                assert_tensor_synced_across_pg(
                    tensor=param, pg=self.dpg.dp_pg, msg=lambda err: f"{name} are not synchronized across DP {err}"
                )

            # SANITY CHECK: Tied weights are synchronized
            tied_params_list = sorted(
                get_tied_id_to_param(
                    parameters=self.normalized_model.parameters(), root_module=self.normalized_model
                ).items(),
                key=lambda x: x[0],
            )

            for (name, group_ranks), param in tied_params_list:
                group = self.dpg.world_ranks_to_pg[group_ranks]
                assert_tensor_synced_across_pg(
                    tensor=param,
                    pg=group,
                    msg=lambda err: f"[Before optimizer step] Tied weights {name} are not synchronized. {err}",
                )

    def after_optim_step_sanity_checks(self) -> None:
        if not self.config.general.ignore_sanity_checks:
            # SANITY CHECK: Check that gradients is cleared
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                if param.grad is not None:
                    log_rank(
                        f"Process rank { dist.get_rank(self.dpg.world_pg)}/{self.dpg.world_pg.size()}: {name} still has gradient despite having ran the optimizer",
                        logger=logger,
                        level=logging.ERROR,
                    )


def mark_tied_parameters(
    model: NanotronModel, dpg: DistributedProcessGroups, parallel_config: Optional[ParallelismArgs] = None
):
    if isinstance(model, GPTForTraining):
        # Tie embeddings
        shared_embeddings = [
            (
                target,
                (
                    dpg.world_rank_matrix[
                        get_pp_rank_of(target, module=model), dist.get_rank(dpg.dp_pg), dist.get_rank(dpg.tp_pg)
                    ],
                ),
            )
            for target in [
                "model.token_position_embeddings.pp_block.token_embedding.weight",
                "model.lm_head.pp_block.weight",
            ]
        ]
        tie_parameters(root_module=model, ties=shared_embeddings, dpg=dpg, reduce_op=dist.ReduceOp.SUM)

    # TODO @nouamane: refactor tying parameters
    if isinstance(model, FalconForTraining):
        # Tie embeddings
        shared_embeddings = [
            (
                target,
                (
                    dpg.world_rank_matrix[
                        get_pp_rank_of(target, module=model), dist.get_rank(dpg.dp_pg), dist.get_rank(dpg.tp_pg)
                    ],
                ),
            )
            for target in [
                "transformer.word_embeddings.pp_block.token_embedding.weight",
                "transformer.lm_head.pp_block.weight",
            ]
        ]
        tie_parameters(root_module=model, ties=shared_embeddings, dpg=dpg, reduce_op=dist.ReduceOp.SUM)

    # Sync all parameters that have the same name and that are not sharded
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            name = f"{module_name}.{param_name}"

            if isinstance(model, GPTForTraining) and ".qkv.kv." in name:
                assert param.is_tied, f"Expected {name} to already be synced"
                # kv is deliberately skipped as it's tied in model init (_mark_kv_parameters_in_module_as_tied)
                continue

            if isinstance(param, NanotronParameter) and param.is_sharded:
                continue

            if isinstance(module, TensorParallelRowLinear) and "bias" == param_name:
                # bias for TensorParallelRowLinear only exists on TP=0 so we don't need to tie it
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
