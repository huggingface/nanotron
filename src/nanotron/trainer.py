import datetime
import gc
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from nanotron import logging
from nanotron.config import (
    Config,
    ExistingCheckpointInit,
    ParallelismArgs,
    RandomInit,
    get_config_from_file,
)
from nanotron.core import distributed as dist
from nanotron.core.clip_grads import clip_grad_norm
from nanotron.core.parallel.data_parallelism.utils import sync_gradients_across_dp
from nanotron.core.parallel.parameters import NanotronParameter, sanity_check
from nanotron.core.parallel.pipeline_parallelism.block import PipelineBlock
from nanotron.core.parallel.pipeline_parallelism.engine import (
    PipelineEngine,
)
from nanotron.core.parallel.pipeline_parallelism.tensor_pointer import TensorPointer
from nanotron.core.parallel.pipeline_parallelism.utils import get_pp_rank_of
from nanotron.core.parallel.tensor_parallelism.nn import (
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)
from nanotron.core.parallel.tied_parameters import (
    create_pg_for_tied_weights,
    get_tied_id_to_param,
    sync_tied_weights_gradients,
    tie_parameters,
)
from nanotron.core.process_groups import DistributedProcessGroups
from nanotron.core.random import (
    set_random_seed,
)
from nanotron.core.tensor_init import init_method_normal, scaled_init_method_normal
from nanotron.core.utils import (
    assert_tensor_synced_across_pg,
    check_env,
    init_on_device_and_dtype,
)
from nanotron.dataloader import sanity_check_dataloader
from nanotron.distributed import ParallelContext, ParallelMode
from nanotron.helpers import (
    _vocab_size_with_padding,
    get_profiler,
    init_optimizer_and_grad_accumulator,
    init_random_states,
    log_throughput,
    lr_scheduler_builder,
    set_logger_verbosity_format,
)
from nanotron.logging import LoggerWriter, LogItem, human_format, log_rank
from nanotron.models import NanotronModel
from nanotron.serialize import (
    load_lr_scheduler,
    load_meta,
    load_optimizer,
    load_weights,
    parse_ckpt_path,
    save,
    save_random_states,
)

if int(os.environ.get("USE_FAST", 0)) == 1:
    # We import the fast versions
    from nanotron.models.fast.falcon import FalconForTraining
    from nanotron.models.fast.gpt2 import GPTForTraining
    from nanotron.models.fast.llama import LlamaForTraining, RotaryEmbedding

    # from nanotron.models.fast.starcoder2 import Starcoder2ForTraining
else:
    from nanotron.models.falcon import FalconForTraining
    from nanotron.models.gpt2 import GPTForTraining
    from nanotron.models.llama import LlamaForTraining, RotaryEmbedding

    # from nanotron.models.fast.starcoder2 import Starcoder2ForTraining

logger = logging.get_logger(__name__)

# Reduce the logging noise from torch.distributed when creating new process groups
dist_logger = logging.get_logger(dist.dist.__name__)
dist_logger.setLevel(logging.WARNING)

CONFIG_TO_MODEL_CLASS = {
    "LlamaConfig": LlamaForTraining,
    "GPTBigCodeConfig": GPTForTraining,
    "FalconConfig": FalconForTraining,
    "RWConfig": FalconForTraining,
    # "Starcoder2Config": Starcoder2ForTraining,
}

MIN_GPU_MEM_THRESHOLD = 8e10  # 80GB
NUM_THROUGHPUT_ITERS = 5
THROUGHPUT_TENSOR_SIZE = 8e9  # 8GB


# TODO @nouamane: add abstract class
class DistributedTrainer:
    def __init__(self, config_or_config_file: Union[Config, str]):
        super().__init__()
        check_env()
        self.config = get_config_from_file(config_or_config_file)
        self.model_config = self.config.model.model_config

        ########################################
        ## We start with setting up loggers and process groups
        ########################################

        # Initialise all process groups
        # self.dpg = get_process_groups(
        #     data_parallel_size=self.config.parallelism.dp,
        #     pipeline_parallel_size=self.config.parallelism.pp,
        #     tensor_parallel_size=self.config.parallelism.tp,
        # )
        self.parallel_context = ParallelContext.from_torch(
            tensor_parallel_size=self.config.parallelism.tp,
            pipeline_parallel_size=self.config.parallelism.pp,
            data_parallel_size=self.config.parallelism.dp,
        )

        # Set log levels
        if self.parallel_context.get_global_rank() == 0:
            if self.config.logging.log_level is not None:
                set_logger_verbosity_format(self.config.logging.log_level, self.parallel_context)
        else:
            if self.config.logging.log_level_replica is not None:
                set_logger_verbosity_format(self.config.logging.log_level_replica, self.parallel_context)

        ########################################
        ## Do a couple of NCCL and CUDA tests to catch faulty nodes
        ########################################

        # Do a first NCCL sync to warmup and try to avoid Timeout after model/data loading
        log_rank(
            f"[TEST] Running NCCL sync for ranks {list(range(self.parallel_context.get_world_size(ParallelMode.GLOBAL)))}",
            logger=logger,
            level=logging.WARNING,
            group=self.parallel_context.get_group(ParallelMode.DATA),
            rank=0,
        )
        test_tensor = torch.tensor([self.parallel_context.get_global_rank()], device=torch.device("cuda"))
        test_tensor_list = [
            torch.zeros_like(test_tensor) for _ in range(self.parallel_context.get_world_size(ParallelMode.GLOBAL))
        ]
        dist.all_gather(
            test_tensor_list, test_tensor, group=self.parallel_context.get_group(ParallelMode.GLOBAL), async_op=False
        )
        dist.barrier()
        log_rank(
            f"[TEST] NCCL sync for ranks {[t.item() for t in test_tensor_list]}",
            logger=logger,
            level=logging.WARNING,
            group=self.parallel_context.get_group(ParallelMode.DATA),
            rank=0,
        )

        # Test to allocate a large tensor to test memory
        gc.collect()
        torch.cuda.empty_cache()
        free_mem, total_mem = torch.cuda.mem_get_info()
        log_rank(
            f"[TEST] free memory free_mem: {human_format(free_mem)}, total_mem: {human_format(total_mem)}",
            logger=logger,
            level=logging.WARNING,
            group=self.parallel_context.get_group(ParallelMode.GLOBAL),
            rank=None,
        )
        if free_mem < MIN_GPU_MEM_THRESHOLD:
            raise RuntimeError(f"Not enough memory to train the model on node {os.environ.get('SLURMD_NODENAME')}")
        # Try to allocate all the memory
        test_tensor_size = int(free_mem * 0.9)
        test_tensor = torch.zeros((test_tensor_size,), dtype=torch.uint8, device=torch.device("cuda"))
        log_rank(
            f"[TEST] Allocated a tensor of size {human_format(test_tensor_size)} (90% of free memory)",
            logger=logger,
            level=logging.WARNING,
            group=self.parallel_context.get_group(ParallelMode.GLOBAL),
            rank=None,
        )
        del test_tensor
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Log benchmark info
        if os.environ.get("NANOTRON_BENCHMARK", "0") == "1":
            log_throughput(self.config, self.parallel_context)

        ########################################
        ## Setting up our model, optimizers, schedulers, etc.
        ########################################

        # Set random states
        set_random_seed(self.config.general.seed)

        # Init model and build on pp ranks
        # TODO(xrsrke): add parallel_context to init_model_states
        self.random_states = init_random_states(
            parallel_config=self.config.parallelism, tp_pg=self.parallel_context.get_group(ParallelMode.TENSOR)
        )
        self.model, checkpoint_path = self.init_model()  # Defines self.model
        self.normalized_model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

        # Init optimizer
        self.optimizer, self.grad_accumulator = init_optimizer_and_grad_accumulator(
            model=self.model, optimizer_args=self.config.optimizer, parallel_context=self.parallel_context
        )
        if checkpoint_path is not None:
            load_optimizer(
                optimizer=self.optimizer, parallel_context=self.parallel_context, root_folder=checkpoint_path
            )

        # Init learning rate scheduler
        self.lr_scheduler = lr_scheduler_builder(
            optimizer=self.optimizer,
            lr_scheduler_args=self.config.optimizer.learning_rate_scheduler,
            total_training_steps=self.config.tokens.train_steps,
        )
        if checkpoint_path is not None:
            load_lr_scheduler(
                lr_scheduler=self.lr_scheduler,
                root_folder=checkpoint_path,
            )

        # Define iteration start state
        self.start_iteration_step: int
        self.consumed_train_samples: int
        if checkpoint_path is not None:
            checkpoint_metadata = load_meta(checkpoint_path, self.parallel_context)
            log_rank(str(checkpoint_metadata), logger=logger, level=logging.INFO, rank=0)
            self.start_iteration_step = checkpoint_metadata.metas["last_train_step"]
            self.consumed_train_samples = checkpoint_metadata.metas["consumed_train_samples"]
            assert (
                self.config.tokens.train_steps > self.start_iteration_step
            ), f"Loaded checkpoint has already trained {self.start_iteration_step} batches, you need to specify a higher `config.tokens.train_steps`"
        else:
            self.start_iteration_step = 0
            self.consumed_train_samples = 0

        # Setup tensorboard write and log writers on output rank
        self.logger_ranks = self.parallel_context.world_rank_matrix[
            self.normalized_model.output_pp_rank, 0, 0
        ].flatten()
        self.loggerwriter = self.setup_log_writers()

        # Log where each module is instantiated
        self.normalized_model.log_modules(
            level=logging.DEBUG, group=self.parallel_context.get_group(ParallelMode.GLOBAL), rank=0
        )

        # Log config and model config
        # self.log_object(self.config, "config")
        # if hasattr(self.model_config, "to_json_string"):
        #     model_config_dict = json.loads(self.model_config.to_json_string())
        # else:
        #     model_config_dict = asdict(self.model_config)
        # self.log_object(model_config_dict, "model_config")

        # Log environment variables
        # self.log_object(os.environ, "environment_variables")
        # if os.environ.get("SLURM_JOB_ID", None) is not None:
        #     keys = [
        #         "JobId",
        #         "Name",
        #         "Command",
        #         "STDOUT",
        #         "STDERR",
        #         "NumNodes",
        #         "NodeList",
        #         "GroupID",
        #         "OverSubscribe",
        #         "Partition",
        #         "cpus-per-task",
        #         "UserName",
        #         "SubmitTime",
        #     ]
        #     format_str = ",".join(f"{k}:1000" for k in keys)
        #     output = subprocess.check_output(
        #         [f'squeue --Format="{format_str}" -j {os.environ.get("SLURM_JOB_ID", None)} --noheader'],
        #         universal_newlines=True,
        #         stderr=subprocess.STDOUT,
        #         shell=True,
        #     )
        #     slurm_dict = {k: output[i * 1000 : (i + 1) * 1000].strip() for i, k in enumerate(keys)}
        #     slurm_job_name = slurm_dict["Name"]
        #     slurm_job_id = slurm_dict["JobId"]
        #     for key, value in os.environ.items():
        #         if key.startswith("SLURM") or key.startswith("SRUN"):
        #             slurm_dict[key] = value
        #     slurm_dict = {
        #         k: o.replace("%x", slurm_job_name).replace("%j", slurm_job_id).replace("%n", "0").replace("%t", "0")
        #         for k, o in slurm_dict.items()
        #     }
        #     for key, value in os.environ.items():
        #         if key.startswith("SLURM") or key.startswith("SRUN"):
        #             slurm_dict[key] = value
        #     self.log_object(slurm_dict, "slurm")

        # Do a first NCCL sync to warmup and try to avoid Timeout after model/data loading
        test_tensor = torch.tensor([self.parallel_context.get_global_rank()], device=torch.device("cuda"))
        test_tensor_list = [
            torch.zeros_like(test_tensor) for _ in range(self.parallel_context.get_world_size(ParallelMode.GLOBAL))
        ]
        dist.all_gather(
            test_tensor_list, test_tensor, group=self.parallel_context.get_group(ParallelMode.GLOBAL), async_op=False
        )
        dist.barrier()
        log_rank(
            f"[SECOND TEST] NCCL sync for ranks {[t.item() for t in test_tensor_list]}",
            logger=logger,
            level=logging.WARNING,
            group=self.parallel_context.get_group(ParallelMode.DATA),
            rank=0,
        )

        global_rank = self.parallel_context.get_global_rank()
        world_size = self.parallel_context.get_world_size(ParallelMode.GLOBAL)
        local_pp_rank = self.parallel_context.get_local_rank(ParallelMode.PIPELINE)
        pp_world_size = self.parallel_context.get_world_size(ParallelMode.PIPELINE)
        self.parallel_context.get_local_rank(ParallelMode.DATA)

        log_rank(
            f"Global rank: {global_rank}/{world_size} | PP: {local_pp_rank}/{pp_world_size} | DP: {self.parallel_context.get_local_rank(ParallelMode.DATA)}/{self.parallel_context.get_world_size(ParallelMode.DATA)} | TP: {self.parallel_context.get_local_rank(ParallelMode.TENSOR)}/{self.parallel_context.get_world_size(ParallelMode.TENSOR)}",
            logger=logger,
            level=logging.INFO,
        )

        self.micro_batch_size = self.config.tokens.micro_batch_size
        self.n_micro_batches_per_batch = self.config.tokens.batch_accumulation_per_replica
        self.global_batch_size = (
            self.micro_batch_size
            * self.n_micro_batches_per_batch
            * self.parallel_context.get_world_size(ParallelMode.DATA)
        )
        self.sequence_length = self.config.tokens.sequence_length
        self.iteration_step = self.start_iteration_step
        self.limit_val_batches = self.config.tokens.limit_val_batches

        # # S3 Mover and save initial state
        # if self.config.checkpoints.s3 is not None:
        #     # Only local rank 0 should upload
        #     dummy = bool(int(os.environ.get("LOCAL_RANK", None)) != 0)
        #     self.s3_mover = S3Mover(
        #         local_path=self.config.checkpoints.checkpoints_path,
        #         s3_path=self.config.checkpoints.s3.upload_s3_path,
        #         # duplicate_checkpoint_path=self.config.checkpoints.resume_checkpoint_path,
        #         remove_after_upload=self.config.checkpoints.s3.remove_after_upload,
        #         s5cmd_numworkers=self.config.checkpoints.s3.s5cmd_numworkers,
        #         s5cmd_concurrency=self.config.checkpoints.s3.s5cmd_concurrency,
        #         s5cmd_path=self.config.checkpoints.s3.s5cmd_path,
        #         dummy=dummy,
        #     )
        # else:
        #     self.s3_mover = None
        # if self.config.checkpoints.lighteval is not None and self.parallel_context.get_global_rank() == 0:
        #     # We only start evaluation runs on the first node
        #     if self.s3_mover is None:
        #         raise ValueError("lighteval requires s3 upload of checkpoints to be enabled")
        #     self.lighteval_runner = LightEvalRunner(config=self.config, dpg=self.dpg)
        #     self.s3_mover.post_upload_callback = self.lighteval_runner.eval_single_checkpoint

        if self.config.checkpoints.save_initial_state and checkpoint_path is None:
            self.save_checkpoint()

    # def log_object(self, dataclass_object: Any, name: str):
    #     if not dataclass_object or isinstance(self.tb_context, contextlib.nullcontext):
    #         return

    #     self.tb_context.add_text(name, obj_to_markdown(dataclass_object), global_step=1)

    #     # Dataclass objects are usually configs so we push then already now
    #     self.tb_context.flush()

    #     if isinstance(self.tb_context, HubSummaryWriter):
    #         self.tb_context.scheduler.trigger()

    @classmethod
    def from_config_file(cls, config_file: str):
        config = get_config_from_file(config_file)
        return cls(config_or_config_file=config)

    def train(
        self,
        dataloader_or_dls: Union[Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]], Tuple[Iterator, ...]],
        # data_config_log: Optional[TrainDataLog] = None,
    ) -> None:
        if isinstance(dataloader_or_dls, tuple):
            dataloader_or_dls[1] if len(dataloader_or_dls) > 1 else None
            dataloader_or_dls[2] if len(dataloader_or_dls) > 2 else None
            dataloader = dataloader_or_dls[0]
        else:
            dataloader = dataloader_or_dls
        dataloader = sanity_check_dataloader(
            dataloader=dataloader, parallel_context=self.parallel_context, config=self.config
        )

        # Log data config
        # self.log_object(data_config_log, name="data_config")

        self.pipeline_engine: PipelineEngine = self.config.parallelism.pp_engine

        self.pipeline_engine.nb_microbatches = self.n_micro_batches_per_batch

        log_rank(
            f"[Start training] datetime: {datetime.datetime.now()} | mbs: {self.micro_batch_size} | grad_accum: {self.n_micro_batches_per_batch} | global_batch_size: {self.global_batch_size} | sequence_length: {self.sequence_length} | train_steps: {self.config.tokens.train_steps} | start_iteration_step: {self.start_iteration_step} | consumed_train_samples: {self.consumed_train_samples}",  # noqa
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
        torch.cuda.empty_cache()
        with prof:
            for self.iteration_step in range(self.start_iteration_step + 1, self.config.tokens.train_steps + 1):
                if isinstance(prof, torch.profiler.profile):
                    prof.step()
                self.iteration_start_time = time.time()

                # Training step
                outputs, loss_avg = self.training_step(dataloader=dataloader)

                # Training Logs
                self.consumed_train_samples += self.global_batch_size

                if (self.iteration_step - 1) % self.config.logging.iteration_step_info_interval == 0:
                    self.train_step_logs(outputs=outputs, loss_avg=loss_avg)

                # Kill switch
                self.check_kill_switch(save_ckpt=True)

                # Checkpoint
                if self.iteration_step % self.config.checkpoints.checkpoint_interval == 0:
                    self.save_checkpoint()

                # Update our background upload/removal of checkpoints
                # if self.s3_mover is not None:
                #     self.s3_mover.update()

                # Validation #TODO: fix validation
                # if (
                #     valid_dataloader is not None
                #     and self.iteration_step % self.config.tokens.val_check_interval == 0
                # ):
                #     self.validation_step(dataloader=valid_dataloader)

                # Push to Hub
                # if (
                #     isinstance(self.tb_context, HubSummaryWriter)
                #     and (self.iteration_step - 1) % self.config.logging.tensorboard_logger.push_to_hub_interval
                #     == 0
                # ):
                #     # tb_writer only exists on a single rank
                #     log_rank(
                #         f"Push Tensorboard logging to Hub at iteration {self.iteration_step} to https://huggingface.co/{self.config.logging.tensorboard_logger.repo_id}/tensorboard",
                #         logger=logger,
                #         level=logging.INFO,
                #     )
                #     # it is a future that queues to avoid concurrent push
                #     self.tb_context.scheduler.trigger()

        # if self.s3_mover is not None:
        #     self.s3_mover.distributed_wait_for_completion(group=self.parallel_context.get_group(ParallelMode.GLOBAL))

    def training_step(
        self, dataloader: Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]
    ) -> Tuple[Iterable[Dict], Optional[torch.Tensor]]:
        self.before_tbi_sanity_checks()

        if self.iteration_step < 5:
            log_rank(
                f"[Before train batch iter] Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MiB."
                f" Peak allocated {torch.cuda.max_memory_allocated() / 1024**2:.2f}MiB."
                f" Peak reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MiB",
                logger=logger,
                level=logging.INFO,
                group=self.parallel_context.get_group(ParallelMode.GLOBAL),
                rank=0,
            )
            torch.cuda.reset_peak_memory_stats()

        outputs = self.pipeline_engine.train_batch_iter(
            model=self.model,
            parallel_context=self.parallel_context,
            batch=(next(dataloader) for _ in range(self.n_micro_batches_per_batch)),
            nb_microbatches=self.n_micro_batches_per_batch,
            grad_accumulator=self.grad_accumulator,
        )

        if self.iteration_step < 5:
            log_rank(
                f"[After train batch iter] Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MiB."
                f" Peak allocated {torch.cuda.max_memory_allocated() / 1024**2:.2f}MiB."
                f" Peak reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MiB",
                logger=logger,
                level=logging.INFO,
                group=self.parallel_context.get_group(ParallelMode.GLOBAL),
                rank=0,
            )
            torch.cuda.reset_peak_memory_stats()

        self.after_tbi_sanity_checks()

        if isinstance(self.model, DistributedDataParallel) and self.grad_accumulator is not None:
            # Wait for fp32 grads allreduce to finish to make sure grads are synced across DP
            assert (
                self.grad_accumulator.fp32_grads_allreduce_handle is not None
            ), "No fp32_grads_allreduce_handle maybe you're using only a single training process"
            self.grad_accumulator.fp32_grads_allreduce_handle.wait()

        # Sync tied weights
        if not isinstance(self.model, DistributedDataParallel):
            # Manually sync across DP if it's not handled by DDP
            sync_gradients_across_dp(
                module=self.model,
                dp_pg=self.parallel_context.get_group(ParallelMode.DATA),
                reduce_op=dist.ReduceOp.AVG,
                # TODO @thomasw21: This is too memory hungry, instead we run all_reduce
                reduce_scatter=False,  # optimizer.inherit_from(ZeroDistributedOptimizer),
                grad_accumulator=self.grad_accumulator,
            )

        # TODO @nouamane: Put this in hooks so we can overlap communication with gradient computation on the last backward pass.
        sync_tied_weights_gradients(
            module=self.normalized_model,
            grad_accumulator=self.grad_accumulator,
            parallel_context=self.parallel_context,
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
                mp_pg=self.parallel_context.world_ranks_to_pg[
                    tuple(
                        sorted(
                            self.parallel_context.world_rank_matrix[
                                :, self.parallel_context.get_local_rank(ParallelMode.DATA), :
                            ].reshape(-1)
                        )
                    )
                ],
                named_parameters=named_parameters,
                grad_accumulator=self.grad_accumulator,
                max_norm=self.config.optimizer.clip_grad,
            )

        self.before_optim_step_sanity_checks()

        # Compute DP average loss and overlap with optimizer step
        if isinstance(outputs[0]["loss"], torch.Tensor):
            # This is an average on only one data rank.
            loss_avg = torch.stack(
                [output["loss"] for output in outputs]
            ).sum()  # already divided by n_micro_batches_per_batch
            # sync loss across DP
            handle = dist.all_reduce(
                loss_avg, group=self.parallel_context.get_group(ParallelMode.DATA), async_op=True, op=dist.ReduceOp.AVG
            )
        else:
            loss_avg = None
            handle = None

        # Apply gradient
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Update the learning rate
        self.lr_scheduler.step()

        self.after_optim_step_sanity_checks()

        if handle is not None:
            handle.wait()
        return outputs, loss_avg

    def validation_step(self, dataloader: Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]) -> Iterable[Dict]:
        outputs = self.pipeline_engine.validate_batch_iter(
            model=self.model,
            batch=(next(dataloader) for _ in range(self.limit_val_batches)),
            nb_microbatches=self.limit_val_batches,
        )
        return outputs

    def train_step_logs(
        self,
        outputs: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        loss_avg: Optional[torch.Tensor],
    ) -> None:
        # TODO @nouamanetazi: Megatron-LM seems to be using a barrier to report their interval time. Check if this is necessary. https://github.com/NouamaneTazi/Megatron-LM/blob/e241a96c3085b18e36c6cee1d68a8155de77b5a6/megatron/training.py#L607
        dist.barrier()
        torch.cuda.synchronize()
        elapsed_time_per_iteration_ms = (time.time() - self.iteration_start_time) * 1000
        tokens_per_sec = (
            self.global_batch_size * self.sequence_length / (elapsed_time_per_iteration_ms / 1000)
        )  # tokens_per_sec is calculated using sequence_length
        model_tflops, hardware_tflops = self.normalized_model.get_flops_per_sec(
            iteration_time_in_sec=elapsed_time_per_iteration_ms / 1000,
            sequence_length=self.sequence_length,
            global_batch_size=self.global_batch_size,
        )

        if self.parallel_context.get_global_rank() in self.logger_ranks:
            assert self.loggerwriter is not None, "loggerwriter should be defined on logger ranks"

            lr = self.lr_scheduler.get_last_lr()[0]

            log_entries = [
                # LogItem("consumed_samples", self.consumed_train_samples, "human_format"),  # , "12d"),
                LogItem(
                    "consumed_tokens", self.consumed_train_samples * self.config.tokens.sequence_length, "human_format"
                ),  # , "12d"),
                LogItem("elapsed_time_per_iteration_ms", elapsed_time_per_iteration_ms, "human_format"),  # , ".1f"),
                LogItem("tokens_per_sec", tokens_per_sec, "human_format"),  # , "1.6E"),
                LogItem(
                    "tokens_per_sec_per_gpu",
                    tokens_per_sec / self.parallel_context.get_world_size(ParallelMode.GLOBAL),
                    "human_format",
                ),  # , "1.6E"),
                LogItem("global_batch_size", self.global_batch_size, "human_format"),  # , "5d"),
                LogItem("lm_loss", loss_avg.item(), "human_format"),  # , "1.6E"),
                LogItem("lr", lr, "human_format"),  # , ".3E"),
                LogItem("model_tflops_per_gpu", model_tflops, "human_format"),  # , ".2f"),
                LogItem("hardware_tflops_per_gpu", hardware_tflops, "human_format"),  # , ".2f"),
            ]

            if self.config.optimizer.clip_grad is not None:
                log_entries.append(LogItem("grad_norm", self.grad_norm_unclipped.item(), "human_format"))  # , ".3f"))

            # if not self.s3_mover.dummy:
            #     log_entries.append(
            #         LogItem("s3_mover_busy", self.s3_mover.get_state_as_int(), "human_format")
            #     )  # , ".3f"))

            # Log not too often the memory
            if self.iteration_step < 5 or (self.iteration_step - 1) % self.config.checkpoints.checkpoint_interval == 0:
                total, used, free = shutil.disk_usage("/")
                log_entries.extend(
                    [
                        LogItem(
                            "cuda_memory_allocated", torch.cuda.memory_allocated(), "human_format"
                        ),  #  / 1024**2, ".2f"),
                        LogItem(
                            "cuda_max_memory_reserved", torch.cuda.max_memory_reserved(), "human_format"
                        ),  #  / 1024**2, ".2f"),
                        LogItem("hd_total_memory_tb", total, "human_format"),  #  / (2**40), ".2f"),
                        LogItem("hd_used_memory_tb", used, "human_format"),  #  / (2**40), ".2f"),
                        LogItem("hd_free_memory_tb", free, "human_format"),  #  / (2**40), ".2f"),
                    ]
                )

            # if not isinstance(tb_writer, contextlib.nullcontext):
            #     tb_writer.add_scalars_from_list(log_entries, self.iteration_step)

            self.loggerwriter.add_scalars_from_list(log_entries, self.iteration_step)

        # Nanotron Benchmark mode: we log the throughput and exit
        if os.environ.get("NANOTRON_BENCHMARK", "0") == "1" and self.iteration_step == 3:
            log_throughput(
                self.config,
                self.parallel_context,
                model_tflops,
                hardware_tflops,
                tokens_per_sec,
            )
            log_rank("Throughput logging complete", logger=logger, level=logging.INFO)
            if "SLURM_JOB_ID" in os.environ:
                os.system("scancel " + os.environ["SLURM_JOB_ID"])
            else:
                exit(0)

    @staticmethod
    def build_model(
        model_builder: Callable[[], NanotronModel],
        parallel_context: DistributedProcessGroups,
        dtype: torch.dtype,
        target_pp_ranks: Optional[List[int]] = None,
        device: Optional[torch.device] = torch.device("cuda"),
    ) -> NanotronModel:
        """Build the model and set the pp ranks for each pipeline block."""
        # TODO: classes dont take same args
        log_rank(
            "Building model..",
            logger=logger,
            level=logging.INFO,
            rank=0,
            group=parallel_context.get_group(ParallelMode.GLOBAL),
        )
        model: NanotronModel = model_builder()

        # If no target pp ranks are specified, we assume that we want to use all pp ranks
        if target_pp_ranks is None:
            pp_world_size = parallel_context.get_world_size(ParallelMode.PIPELINE)
            target_pp_ranks = list(range(pp_world_size))
        else:
            pp_world_size = len(target_pp_ranks)

        # Set rank for each pipeline block
        global_group = parallel_context.get_group(ParallelMode.GLOBAL)
        # TODO(xrsrke): do new api: log_rank(local_rank, parallel_mode)
        log_rank("Setting PP block ranks..", logger=logger, level=logging.INFO, rank=0, group=global_group)
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

            thresholds = [block_cumulative_costs[-1] * ((rank + 1) / pp_world_size) for rank in range(pp_world_size)]
            assert thresholds[-1] >= block_cumulative_costs[-1]
            target_pp_rank_idx = 0
            for block, cumulative_cost in zip(pipeline_blocks, block_cumulative_costs):
                assert target_pp_rank_idx < pp_world_size
                block.build_and_set_rank(target_pp_ranks[target_pp_rank_idx])

                if cumulative_cost > thresholds[target_pp_rank_idx]:
                    target_pp_rank_idx += 1

            model.input_pp_rank = target_pp_ranks[0]
            model.output_pp_rank = target_pp_ranks[target_pp_rank_idx]

        return model

    def init_model(self) -> Tuple[NanotronModel, Optional[str]]:
        """Initialize the model and load weights from checkpoint if needed."""
        # TODO: add max_position_embeddings
        self.model_config.vocab_size = _vocab_size_with_padding(
            self.model_config.vocab_size,
            pg_size=self.parallel_context.get_world_size(ParallelMode.TENSOR),
            make_vocab_size_divisible_by=self.config.model.make_vocab_size_divisible_by,
        )

        if (
            getattr(self.model_config, "max_position_embeddings", None) is not None
            and self.model_config.max_position_embeddings != self.config.tokens.sequence_length
        ):
            if isinstance(self.config.model.init_method, ExistingCheckpointInit):
                log_rank(
                    f"Finetuning a model with a sequence length {self.config.tokens.sequence_length} that is different from the checkpoint's max_position_embeddings {self.model_config.max_position_embeddings}.",  # noqa
                    logger=logger,
                    level=logging.WARNING,
                    rank=0,
                )
            else:
                log_rank(
                    f"Setting max_position_embeddings to {self.config.tokens.sequence_length}. Previous value was {self.model_config.max_position_embeddings}.",
                    logger=logger,
                    level=logging.INFO,
                    rank=0,
                )
                self.model_config.max_position_embeddings = self.config.tokens.sequence_length

        # log_rank(pformat(self.config), logger=logger, level=logging.INFO, rank=0)
        log_rank(pformat(self.config), logger=logger, level=logging.INFO, rank=0)
        log_rank(pformat(self.model_config), logger=logger, level=logging.INFO, rank=0)

        model_config_cls = self.model_config.__class__.__name__
        assert (
            model_config_cls in CONFIG_TO_MODEL_CLASS
        ), f"Unsupported model config {model_config_cls}. Only {CONFIG_TO_MODEL_CLASS.keys()} are supported"

        model = self._init_model(
            model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config_cls](
                config=self.model_config,
                parallel_context=self.parallel_context,
                parallel_config=self.config.parallelism,
                random_states=self.random_states,
            ),
        )
        normalized_model = model.module if isinstance(model, DistributedDataParallel) else model

        # Load or initialize model weights
        checkpoint_path = parse_ckpt_path(config=self.config)
        reloaded_from_checkpoint = False
        if checkpoint_path is not None:
            # Reload from a training checkpoint
            log_rank(f"Loading weights from {checkpoint_path}", logger=logger, level=logging.INFO, rank=0)
            load_weights(model=normalized_model, parallel_context=self.parallel_context, root_folder=checkpoint_path)
            reloaded_from_checkpoint = True
        if not reloaded_from_checkpoint:
            log_rank("No checkpoint path provided.", logger=logger, level=logging.INFO)
            if isinstance(self.config.model.init_method, ExistingCheckpointInit):
                # Initialize model from an pretrained model checkpoint
                load_weights(
                    model=normalized_model,
                    parallel_context=self.parallel_context,
                    root_folder=self.config.model.init_method.path,
                )
            elif isinstance(self.config.model.init_method, RandomInit):
                # Initialize model randomly
                normalized_model.init_model_randomly(
                    init_method=init_method_normal(self.config.model.init_method.std),
                    scaled_init_method=scaled_init_method_normal(
                        self.config.model.init_method.std, self.model_config.num_hidden_layers
                    ),
                )
                # Synchronize parameters so that the model is consistent
                # sync all params across dp
                for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
                    dist.all_reduce(
                        param, op=dist.ReduceOp.AVG, group=self.parallel_context.get_group(ParallelMode.DATA)
                    )

                # sync tied params across tied groups
                for (_, group_ranks), param in sorted(
                    get_tied_id_to_param(
                        parameters=model.parameters(),
                        root_module=normalized_model,
                    ).items(),
                    key=lambda x: x[0],
                ):
                    group = self.parallel_context.world_ranks_to_pg[group_ranks]
                    dist.all_reduce(param, op=dist.ReduceOp.AVG, group=group)
            else:
                raise ValueError(f"Unsupported {self.config.model.init_method}")

        return model, checkpoint_path

    def _init_model(
        self,
        model_builder: Callable[[], NanotronModel],
        target_pp_ranks: Optional[List[int]] = None,
    ) -> NanotronModel:
        config = self.config
        parallel_context = self.parallel_context

        parallel_config = config.parallelism
        make_ddp = not (config.optimizer.accumulate_grad_in_fp32 and config.optimizer.zero_stage > 0)

        # Build model and set pp ranks
        model = self.build_model(
            parallel_context=parallel_context,
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
        mark_tied_parameters(model, parallel_config, self.parallel_context)

        # count number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        size_params = sum(p.numel() * p.element_size() for p in model.parameters())
        total_params = torch.tensor(num_params, device="cuda")
        total_size = torch.tensor(size_params, device="cuda")

        global_group = parallel_context.get_group(ParallelMode.GLOBAL)
        dp_group = parallel_context.get_group(ParallelMode.DATA)
        tp_group = parallel_context.get_group(ParallelMode.TENSOR)
        pp_group = parallel_context.get_group(ParallelMode.PIPELINE)

        dist.all_reduce(total_params, group=tp_group, async_op=False, op=dist.ReduceOp.SUM)  # TP
        dist.all_reduce(total_params, group=pp_group, async_op=False, op=dist.ReduceOp.SUM)  # PP
        dist.all_reduce(total_size, group=tp_group, async_op=False, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_size, group=pp_group, async_op=False, op=dist.ReduceOp.SUM)

        # TODO @nouamanetazi: better memory logs
        log_rank(
            f"Total number of parameters: {human_format(total_params.item())} ({total_size.item() / 1024**2:.2f}MiB)",
            logger=logger,
            level=logging.INFO,
            group=global_group,
            rank=0,
        )
        log_rank(
            f"Local number of parameters: {human_format(num_params)} ({size_params / 1024**2:.2f}MiB)",
            logger=logger,
            level=logging.INFO,
            group=dp_group,
            rank=0,
        )
        log_rank(
            f"[After model building] Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MiB."
            f" Peak allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f}MiB"
            f" Peak reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MiB",
            logger=logger,
            level=logging.INFO,
            group=dp_group,
            rank=0,
        )

        # Model make it DDP
        if make_ddp is True:
            # TODO @thomasw21: DDP doesn't support broadcasting complex buffers (and we don't really need that broadcasting anyway)
            model = DistributedDataParallel(
                model, process_group=dp_group, broadcast_buffers=False, bucket_cap_mb=config.model.ddp_bucket_cap_mb
            )

        # Sanity check the model, all parameters must be NanotronParameter (either tied or sharded)
        sanity_check(root_module=model)

        return model

    def setup_log_writers(
        self,
    ) -> Optional[LoggerWriter]:
        """Setup all log writers on the appropriate ranks."""
        if self.parallel_context.get_global_rank() in self.logger_ranks:
            loggerwriter = LoggerWriter(global_step=self.config.tokens.train_steps)
        else:
            loggerwriter = None

        return loggerwriter

    def check_kill_switch(self, save_ckpt: bool):
        if self.config.general.kill_switch_path and self.config.general.kill_switch_path.exists():
            log_rank(
                f"Detected kill switch at {self.config.general.kill_switch_path}. Exiting",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

            # Save checkpoint
            if save_ckpt:
                self.save_checkpoint()
            dist.barrier()
            sys.exit(0)

    def save_checkpoint(self) -> Path:
        # if self.s3_mover is not None:
        #     self.s3_mover.distributed_wait_for_completion(self.parallel_context.get_group(ParallelMode.GLOBAL))
        #     if self.s3_mover.post_upload_callback_outputs is not None:
        #         slurm_job_id, slurm_log = self.s3_mover.post_upload_callback_outputs
        #         self.log_object({"job_id": slurm_job_id, "log": slurm_log}, "slurm_eval")

        checkpoints_path = self.config.checkpoints.checkpoints_path
        checkpoint_path = checkpoints_path / f"{self.iteration_step}"
        if self.config.checkpoints.checkpoints_path_is_shared_file_system:
            should_mkdir = self.parallel_context.get_global_rank() == 0
        else:
            should_mkdir = bool(int(os.environ.get("LOCAL_RANK", None)) == 0)
        if should_mkdir:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        dist.barrier(self.parallel_context.get_group(ParallelMode.GLOBAL))

        log_rank(f"Saving checkpoint at {checkpoint_path}", logger=logger, level=logging.WARNING, rank=0)
        checkpoint_metadata = {
            "last_train_step": self.iteration_step,
            # TODO: @nouamanetazi: Add more metadata to the checkpoint to be able to resume dataloader states properly
            "consumed_train_samples": self.consumed_train_samples,
        }

        # Update step/samples numbers before we save the config
        self.config.general.step = self.iteration_step
        self.config.general.consumed_train_samples = self.consumed_train_samples

        save(
            model=self.normalized_model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            should_save_model=bool(
                self.parallel_context.get_local_rank(ParallelMode.DATA) == 0
            ),  # We only save the weights on DP==0
            should_save_optimizer=True,
            should_save_lr_scheduler=bool(
                self.parallel_context.get_global_rank() == 0
            ),  # We only save the lr_scheduler on world_rank==0
            should_save_config=bool(
                self.parallel_context.get_global_rank() == 0
            ),  # We only save the config on world_rank==0
            parallel_context=self.parallel_context,
            root_folder=checkpoint_path,
            checkpoint_metadata=checkpoint_metadata,
            config=self.config,
        )
        save_random_states(self.random_states, checkpoint_path, self.parallel_context)
        with open(checkpoints_path / "latest.txt", mode="w") as fo:
            fo.write(f"{self.iteration_step}")

        if hasattr(self.model_config, "to_json_file"):
            self.model_config.to_json_file(checkpoint_path / "model_config.json")
        else:
            with open(checkpoint_path / "model_config.json", mode="w") as fo:
                fo.write(json.dumps(asdict(self.model_config)))

        # Upload to S3
        # if self.s3_mover is not None:
        #     self.s3_mover.start_uploading()

        return checkpoint_path

    def before_tbi_sanity_checks(self) -> None:
        if not self.config.general.ignore_sanity_checks:
            # SANITY CHECK: Check that the model params are synchronized across dp
            for name, param in sorted(self.model.named_parameters(), key=lambda x: x[0]):
                assert_tensor_synced_across_pg(
                    tensor=param,
                    pg=self.parallel_context.get_group(ParallelMode.DATA),
                    msg=lambda err: f"{name} are not synchronized across DP {err}",
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
                group = self.parallel_context.world_ranks_to_pg[group_ranks]
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

            # SANITY CHECK: run model specific sanity checks
            self.normalized_model.before_tbi_sanity_checks()

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
                        f"Process rank { self.parallel_context.get_global_rank()}/{self.parallel_context.get_world_size(ParallelMode.GLOBAL)}: {name} is missing gradient",
                        logger=logger,
                        level=logging.ERROR,
                    )

            # SANITY CHECK: run model specific sanity checks
            self.normalized_model.after_tbi_sanity_checks()

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
                group = self.parallel_context.world_ranks_to_pg[group_ranks]
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
                    pg=self.parallel_context.get_group(ParallelMode.DATA),
                    msg=lambda err: f"[Before optimizer step] weights grads for {name} are not synchronized across DP. {err}",
                )

            # SANITY CHECK: Check that the model params are synchronized across dp
            for name, param in sorted(self.model.named_parameters(), key=lambda x: x[0]):
                assert_tensor_synced_across_pg(
                    tensor=param,
                    pg=self.parallel_context.get_group(ParallelMode.DATA),
                    msg=lambda err: f"{name} are not synchronized across DP {err}",
                )

            # SANITY CHECK: Tied weights are synchronized
            tied_params_list = sorted(
                get_tied_id_to_param(
                    parameters=self.normalized_model.parameters(), root_module=self.normalized_model
                ).items(),
                key=lambda x: x[0],
            )

            for (name, group_ranks), param in tied_params_list:
                group = self.parallel_context.world_ranks_to_pg[group_ranks]
                assert_tensor_synced_across_pg(
                    tensor=param,
                    pg=group,
                    msg=lambda err: f"[Before optimizer step] Tied weights {name} are not synchronized. {err}",
                )

            # SANITY CHECK: run model specific sanity checks
            self.normalized_model.before_optim_step_sanity_checks()

    def after_optim_step_sanity_checks(self) -> None:
        if not self.config.general.ignore_sanity_checks:
            # SANITY CHECK: Check that gradients is cleared
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                if param.grad is not None:
                    log_rank(
                        f"Process rank { self.parallel_context.get_global_rank()}/{self.parallel_context.get_world_size(ParallelMode.GLOBAL)}: {name} still has gradient despite having ran the optimizer",
                        logger=logger,
                        level=logging.ERROR,
                    )

            # SANITY CHECK: run model specific sanity checks
            self.normalized_model.after_optim_step_sanity_checks()


def mark_tied_parameters(
    model: NanotronModel,
    parallel_config: Optional[ParallelismArgs] = None,
    parallel_context: Optional[ParallelContext] = None,
):
    assert parallel_context is not None
    # Tie embeddings
    embeddings_lm_head_tied_names = model.get_embeddings_lm_head_tied_names()

    dp_rank = parallel_context.get_local_rank(ParallelMode.DATA)
    tp_rank = parallel_context.get_local_rank(ParallelMode.TENSOR)
    pp_rank = parallel_context.get_local_rank(ParallelMode.PIPELINE)

    if len(embeddings_lm_head_tied_names) > 0:
        shared_embeddings = [
            (
                target,
                (parallel_context.world_rank_matrix[get_pp_rank_of(target, module=model), dp_rank, tp_rank],),
            )
            for target in embeddings_lm_head_tied_names
        ]
        tie_parameters(
            root_module=model, ties=shared_embeddings, reduce_op=dist.ReduceOp.SUM, parallel_context=parallel_context
        )

    # Sync all parameters that have the same name and that are not sharded
    assert not isinstance(model, DistributedDataParallel), "model shouldn't be DDP at this point"
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            name = f"{module_name}.{param_name}"

            if isinstance(model, GPTForTraining) and ".qkv.kv." in name:
                assert param.is_tied, f"Expected {name} to already be synced"
                # kv is deliberately skipped as it's tied in model init (_mark_kv_parameters_in_module_as_tied)
                continue

            # if isinstance(model, Starcoder2ForTraining) and ".qkv.kv." in name
            #     assert param.is_tied, f"Expected {name} to already be synced"
            #     # kv is deliberately skipped as it's tied in model init (_mark_kv_parameters_in_module_as_tied)
            #     continue

            if isinstance(param, NanotronParameter) and param.is_sharded:
                continue

            if isinstance(module, TensorParallelRowLinear) and "bias" == param_name:
                # bias for TensorParallelRowLinear only exists on TP=0 so we don't need to tie it
                continue

            shared_weights = [
                (
                    name,
                    # This adds all the tp_ranks in one go
                    tuple(sorted(parallel_context.world_rank_matrix[pp_rank, dp_rank, :])),
                )
            ]

            if parallel_config is None or parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
                # We add `reduce_op=None` in order to signal that the weight are synced by design without needing to reduce
                # when TP=2 we have LN that is duplicated across TP, so by design it's tied
                reduce_op = None
            else:
                reduce_op = dist.ReduceOp.SUM

            tie_parameters(
                root_module=model, ties=shared_weights, reduce_op=reduce_op, parallel_context=parallel_context
            )

    create_pg_for_tied_weights(model, parallel_context)
