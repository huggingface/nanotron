import torch.distributed as dist
import contextlib
from transformers import LlamaForCausalLM, LlamaTokenizerFast, LlamaConfig
from pprint import pformat
from torch.nn.parallel import DistributedDataParallel
from datasets.download.streaming_download_manager import xPath
import datetime
import time
import sys
import torch

from modeling_llama import LlamaForTraining
from config import (
    RandomInit,
    ExistingCheckpointInit,
    get_config_from_file,
    HubLoggerConfig,
)
from flops import get_flops_per_sec

from nanotron.core.process_groups_initializer import get_process_groups
from nanotron.core.serialize import (
    load,
    load_meta,
    load_lr_scheduler,
    load_weights,
    load_optimizer,
)
from nanotron.clip_grads import clip_grad_norm
from nanotron.core.utils import (
    assert_tensor_synced_across_pg,
)
from nanotron.core.parallelism.pipeline_parallelism.engine import (
    PipelineEngine,
)
from nanotron.core.parallelism.data_parallelism.utils import sync_gradients_across_dp
from nanotron.core.parallelism.pipeline_parallelism.block import PipelineBlock
from nanotron.core.serialize.serialize import fs_open
from nanotron.core.random import set_random_seed
from nanotron.core.logging import log_rank
from nanotron.core import logging
from nanotron.core.parallelism.tied_parameters import (
    get_tied_id_to_param,
    sync_tied_weights_gradients,
)
from nanotron.core.optimizer.zero import ZeroDistributedOptimizer
from nanotron.core.parallelism.pipeline_parallelism.tensor_pointer import TensorPointer
from nanotron.logger import LogItem

try:
    from nanotron.logger import HubSummaryWriter
    hub_logger_available = True
except ImportError:
    hub_logger_available = False


logger = logging.get_logger(__name__)

from utils import (
    _vocab_size_with_padding,
    init_optimizer_and_grad_accumulator,
    get_args,
    set_logger_verbosity_format,
    setup_log_writers,
    init_random_states,
    init_model,
    init_model_randomly,
    lr_scheduler_builder,
    dummy_infinite_data_generator,
    save_checkpoint,
    test_equal_dict,
)

def main():

    config_file = get_args().config_file
    config = get_config_from_file(config_file)

    dpg = get_process_groups(
        data_parallel_size=config.parallelism.dp,
        pipeline_parallel_size=config.parallelism.pp,
        tensor_parallel_size=config.parallelism.tp,
    )

    # Set random states
    set_random_seed(config.model.seed)

    # Set log levels
    if dist.get_rank(dpg.world_pg) == 0:
        if config.logging.log_level is not None:
            set_logger_verbosity_format(config.logging.log_level, dpg=dpg)
    else:
        if config.logging.log_level_replica is not None:
            set_logger_verbosity_format(config.logging.log_level_replica, dpg=dpg)

    # Setup all writers
    if (
        dist.get_rank(dpg.pp_pg) == dpg.pp_pg.size() - 1
        and dist.get_rank(dpg.tp_pg) == 0
        and dist.get_rank(dpg.dp_pg) == 0
    ):
        tb_context, loggerwriter = setup_log_writers(config=config)
    else:
        tb_context = contextlib.nullcontext()
        loggerwriter = None

    # Choosing Checkpoint path
    load_from_candidate = config.checkpoints.resume_checkpoint_path
    if load_from_candidate is None:
        latest_meta_path: xPath = config.checkpoints.checkpoints_path / "latest.txt"
        if latest_meta_path.exists():
            with fs_open(config.checkpoints.checkpoints_path / "latest.txt", mode="r") as fi:
                # TODO @thomasw21: make a better structure system so that we get typing correct
                load_from_candidate = int(fi.read())
    checkpoint_path = (
        config.checkpoints.checkpoints_path / str(load_from_candidate) if load_from_candidate is not None else None
    )
    if checkpoint_path is not None:
        log_rank(
            f"Loading checkpoint from {checkpoint_path}:",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

    # Init model
    model_name = config.model.model_name
    model_config: LlamaConfig = LlamaConfig.from_pretrained(model_name)
    model_config.vocab_size = _vocab_size_with_padding(
        model_config.vocab_size,
        pg_size=dpg.tp_pg.size(),
        make_vocab_size_divisible_by=config.model.make_vocab_size_divisible_by,
        logger=logger
    )
    assert (
        model_config.max_position_embeddings >= config.tokens.sequence_length
    ), f"max_position_embeddings ({model_config.max_position_embeddings}) must be >= sequence_length ({config.tokens.sequence_length})"

    log_rank(pformat(config), logger=logger, level=logging.INFO, rank=0)
    log_rank(str(model_config), logger=logger, level=logging.INFO, rank=0)

    optimizer_args = config.optimizer

    random_states = init_random_states(parallel_config=config.parallelism, tp_pg=dpg.tp_pg)

    model = init_model(
        model_builder=lambda: LlamaForTraining(config=model_config, dpg=dpg, parallel_config=config.parallelism),
        model_config=model_config,
        parallel_config=config.parallelism,
        dtype=config.model.dtype,
        dpg=dpg,
        # TODO @thomasw21: Figure out why using DDP with accumulate_in_fp_32 and ZeRO-1 performs poorly.
        make_ddp=not (optimizer_args.accumulate_grad_in_fp32 and optimizer_args.zero_stage > 0),
        logger=logger,
    )

    if checkpoint_path is not None:
        load_weights(model=model, dpg=dpg, root_folder=checkpoint_path)
    else:
        # We initialize the model.
        if isinstance(config.model.init_method, ExistingCheckpointInit):
            # Initialize model from an existing model checkpoint
            load_weights(model=model, dpg=dpg, root_folder=config.model.init_method.resume_checkpoint_path)
        elif isinstance(config.model.init_method, RandomInit):
            # Initialize model randomly
            init_model_randomly(model=model, config=config, model_config=model_config, dpg=dpg)
        else:
            raise ValueError(f"Unsupported {config.model.init_method}")


    #Init optimizer
    optimizer, grad_accumulator = init_optimizer_and_grad_accumulator(
        model=model, optimizer_args=optimizer_args, dpg=dpg
    )
    if checkpoint_path is not None:
        load_optimizer(optimizer=optimizer, dpg=dpg, root_folder=checkpoint_path)


    # Init learning rate scheduler
    lr_scheduler_args = config.learning_rate_scheduler
    lr_scheduler = lr_scheduler_builder(
        optimizer=optimizer, learning_rate=config.optimizer.learning_rate, lr_scheduler_args=lr_scheduler_args
    )

    if checkpoint_path is not None:
        load_lr_scheduler(
            lr_scheduler=lr_scheduler,
            root_folder=checkpoint_path,
        )
    
    # Define iteration start state
    start_iteration_step: int
    consumed_train_samples: int
    if checkpoint_path is not None:
        checkpoint_metadata = load_meta(dpg=dpg, root_folder=checkpoint_path)
        log_rank(str(checkpoint_metadata), logger=logger, level=logging.INFO, rank=0)
        start_iteration_step = checkpoint_metadata.metas["last_train_step"]
        consumed_train_samples = checkpoint_metadata.metas["consumed_train_samples"]
        assert (
            config.tokens.train_steps > start_iteration_step
        ), f"Loaded checkpoint has already trained {start_iteration_step} batches, you need to specify a higher `config.tokens.train_steps`"
    else:
        start_iteration_step = 0
        consumed_train_samples = 0

    # Log where each module is instantiated
    for name, module in model.named_modules():
        if not isinstance(module, PipelineBlock):
            continue
        log_rank(
            f"module_name: {name} | PP: {module.rank}/{dpg.pp_pg.size()}",
            logger=logger,
            level=logging.DEBUG,
            group=dpg.world_pg,
            rank=0,
        )

    dist.barrier()
    log_rank(
        f"Global rank: { dist.get_rank(dpg.world_pg)}/{dpg.world_pg.size()} | PP: {dist.get_rank(dpg.pp_pg)}/{dpg.pp_pg.size()} | DP: {dist.get_rank(dpg.dp_pg)}/{dpg.dp_pg.size()} | TP: {dist.get_rank(dpg.tp_pg)}/{dpg.tp_pg.size()}",
        logger=logger,
        level=logging.INFO,
    )
    dist.barrier()

    # Create dataloader
    log_rank("Using `datasets` library", logger=logger, level=logging.INFO, rank=0)

    # Dummy hyper parameter
    micro_batch_size = config.tokens.micro_batch_size
    n_micro_batches_per_batch = config.tokens.batch_accumulation_per_replica
    global_batch_size = micro_batch_size * n_micro_batches_per_batch * dpg.dp_pg.size()
    sequence_length = config.tokens.sequence_length

    # Create a dummy data loader
    if isinstance(model, DistributedDataParallel):
        input_pp_rank = model.module.model.token_position_embeddings.rank
        output_pp_rank = model.module.loss.rank
    else:
        input_pp_rank = model.model.token_position_embeddings.rank
        output_pp_rank = model.loss.rank

    if config.data.dataset is None:
        log_rank("Using dummy data generator", logger=logger, level=logging.INFO, rank=0)
        data_iterator = dummy_infinite_data_generator(
            micro_batch_size=micro_batch_size,
            sequence_length=sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            vocab_size=model_config.vocab_size,
            seed=config.data.seed,
            dpg=dpg,
        )()

    # Backward from time to time
    # TODO @thomasw21: Make a much better API
    pipeline_engine: PipelineEngine = config.parallelism.pp_engine

    log_rank(
        f"[Before the start of training] datetime: {datetime.datetime.now()}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    start_time = time.time()


    if config.profile is None:
        prof = contextlib.nullcontext()

    # Useful mapping
    normalized_model = model.module if isinstance(model, DistributedDataParallel) else model
    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in normalized_model.named_modules()}
    # Fix the root_model
    module_id_to_prefix[id(normalized_model)] = ""

    # Training logic

    with tb_context as tb_writer:
        with prof:
            for iteration_step in range(start_iteration_step, config.tokens.train_steps):
                if isinstance(prof, torch.profiler.profile):
                    prof.step()


                if not config.general.ignore_sanity_checks:
                    # SANITY CHECK: Check that the model params are synchronized across dp
                    for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
                        assert_tensor_synced_across_pg(
                            tensor=param, pg=dpg.dp_pg, msg=lambda err: f"{name} are not synchronized across DP {err}"
                        )

                    # SANITY CHECK: Tied weights are synchronized
                    tied_params_list = sorted(
                        get_tied_id_to_param(
                            parameters=normalized_model.parameters(),
                            root_module=normalized_model,
                        ).items(),
                        key=lambda x: x[0],
                    )

                    for (name, group_ranks), param in tied_params_list:
                        group = dpg.world_ranks_to_pg[group_ranks]
                        assert_tensor_synced_across_pg(
                            tensor=param,
                            pg=group,
                            msg=lambda err: f"[Before train] Tied weights {name} are not synchronized. {err}",
                        )

                # SANITY CHECK: Check that the grad accumulator buffers are ready for DDP
                if not config.general.ignore_sanity_checks:
                    if grad_accumulator is not None:
                        for _, elt in grad_accumulator.fp32_grad_buffers.items():
                            fp32_grad_buffer = elt["fp32_grad"]
                            torch.testing.assert_close(
                                fp32_grad_buffer,
                                torch.zeros_like(fp32_grad_buffer),
                                atol=0,
                                rtol=0,
                                msg="Grad accumulator buffers must be zeroed in first accumulation step.",
                            )

                if iteration_step <= 5:
                    log_rank(
                        f"[Before train batch iter] (iteration {iteration_step}) Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MB. Peak reserved memory: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MB",
                        logger=logger,
                        level=logging.INFO,
                        group=dpg.dp_pg,
                        rank=0,
                    )

                outputs = pipeline_engine.train_batch_iter(
                    model=model,
                    pg=dpg.pp_pg,
                    batch=(next(data_iterator) for _ in range(n_micro_batches_per_batch)),
                    nb_microbatches=n_micro_batches_per_batch,
                    grad_accumulator=grad_accumulator,
                )

                if iteration_step <= 5:
                    log_rank(
                        f"[After train batch iter] (iteration {iteration_step}) Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MB. Peak reserved memory: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MB",
                        logger=logger,
                        level=logging.INFO,
                        group=dpg.dp_pg,
                        rank=0,
                    )

                if not config.general.ignore_sanity_checks:
                    # SANITY CHECK: Check that gradient flow on the entire model
                    # SANITY CHECK: Check that all parameters that required gradients, have actually a gradient
                    # SANITY CHECK: Check for nan/inf
                    for name, param in normalized_model.named_parameters():
                        if not param.requires_grad:
                            continue

                        if param.is_tied:
                            tied_info = param.get_tied_info()
                            name = tied_info.get_full_name_from_module_id_to_prefix(
                                module_id_to_prefix=module_id_to_prefix
                            )

                        if grad_accumulator is not None:
                            grad = grad_accumulator.get_grad_buffer(name=name)
                        else:
                            grad = param.grad

                        if torch.isnan(grad).any() or torch.isinf(grad).any():
                            raise ValueError("Gradient is nan or inf")
                        if grad is None:
                            log_rank(
                                f"Process rank { dist.get_rank(dpg.world_pg)}/{dpg.world_pg.size()}: {name} is missing gradient",
                                logger=logger,
                                level=logging.ERROR,
                            )

                # Sync tied weights
                # TODO @thomasw21: Put this in hooks so we can overlap communication with gradient computation on the last backward pass.
                sync_tied_weights_gradients(
                    module=normalized_model,
                    dpg=dpg,
                    grad_accumulator=grad_accumulator,
                )
                if not isinstance(model, DistributedDataParallel):
                    # Manually sync across DP if it's not handled by DDP
                    sync_gradients_across_dp(
                        module=model,
                        dp_pg=dpg.dp_pg,
                        reduce_op=dist.ReduceOp.AVG,
                        # TODO @thomasw21: This is too memory hungry: you need to run assign a correctly sized tensor in the reduce_scatter_coalesced operator.
                        # Instead we run all_reduce that doesn't need extra memory.
                        reduce_scatter=False,
                        # reduce_scatter=optimizer.inherit_from(ZeroDistributedOptimizer),
                        grad_accumulator=grad_accumulator,
                    )

                if not config.general.ignore_sanity_checks:
                    # SANITY CHECK: Test tied weights gradients are synchronized
                    for (name, group_ranks), param in sorted(
                        get_tied_id_to_param(
                            parameters=normalized_model.parameters(),
                            root_module=normalized_model,
                        ).items(),
                        key=lambda x: x[0],
                    ):
                        if not param.requires_grad:
                            continue

                        if grad_accumulator is not None:
                            grad = grad_accumulator.get_grad_buffer(name=name)
                        else:
                            grad = param.grad

                        assert grad is not None, f"Grad is None for {name}"
                        group = dpg.world_ranks_to_pg[group_ranks]
                        assert_tensor_synced_across_pg(
                            tensor=grad,
                            pg=group,
                            msg=lambda err: f"[Before gradient clipping] Tied weights grads for {name} are not synchronized. {err}",
                        )

                    # SANITY CHECK: Test gradients are synchronized across DP
                    for name, param in sorted(normalized_model.named_parameters(), key=lambda x: x[0]):
                        if not param.requires_grad:
                            continue

                        if param.is_tied:
                            tied_info = param.get_tied_info()
                            name = tied_info.get_full_name_from_module_id_to_prefix(
                                module_id_to_prefix=module_id_to_prefix
                            )

                        if grad_accumulator is not None:
                            grad = grad_accumulator.get_grad_buffer(name=name)
                        else:
                            grad = param.grad

                        assert grad is not None, f"Grad is None for {name}"
                        assert_tensor_synced_across_pg(
                            tensor=grad,
                            pg=dpg.dp_pg,
                            msg=lambda err: f"[Before gradient clipping] weights grads for {name} are not synchronized across DP. {err}",
                        )

                # Clip gradients
                grad_norm_unclipped = None
                if config.optimizer.clip_grad is not None:
                    # Normalize DDP
                    named_parameters = [
                        (
                            param.get_tied_info().get_full_name_from_module_id_to_prefix(
                                module_id_to_prefix=module_id_to_prefix
                            )
                            if param.is_tied
                            else name,
                            param,
                        )
                        for name, param in normalized_model.named_parameters()
                    ]
                    grad_norm_unclipped = clip_grad_norm(
                        mp_pg=dpg.world_ranks_to_pg[
                            tuple(sorted(dpg.world_rank_matrix[:, dist.get_rank(dpg.dp_pg), :].reshape(-1)))
                        ],
                        named_parameters=named_parameters,
                        grad_accumulator=grad_accumulator,
                        max_norm=config.optimizer.clip_grad,
                    )

                if not config.general.ignore_sanity_checks:
                    # SANITY CHECK: Test tied weights gradients are synchronized
                    for (name, group_ranks), param in sorted(
                        get_tied_id_to_param(
                            parameters=normalized_model.parameters(), root_module=normalized_model
                        ).items(),
                        key=lambda x: x[0],
                    ):
                        if not param.requires_grad:
                            continue

                        if grad_accumulator is not None:
                            grad = grad_accumulator.get_grad_buffer(name=name)
                        else:
                            grad = param.grad

                        assert grad is not None, f"Grad is None for {name}"
                        group = dpg.world_ranks_to_pg[group_ranks]
                        assert_tensor_synced_across_pg(
                            tensor=grad,
                            pg=group,
                            msg=lambda err: f"[Before optimizer step] Tied weights grads for {name} are not synchronized. {err}",
                        )

                    # SANITY CHECK: Test gradients are synchronized across DP
                    for name, param in sorted(normalized_model.named_parameters(), key=lambda x: x[0]):
                        if not param.requires_grad:
                            continue

                        if param.is_tied:
                            tied_info = param.get_tied_info()
                            name = tied_info.get_full_name_from_module_id_to_prefix(
                                module_id_to_prefix=module_id_to_prefix
                            )

                        if grad_accumulator is not None:
                            grad = grad_accumulator.get_grad_buffer(name=name)
                        else:
                            grad = param.grad

                        assert grad is not None, f"Grad is None for {name}"
                        assert_tensor_synced_across_pg(
                            tensor=grad,
                            pg=dpg.dp_pg,
                            msg=lambda err: f"[Before optimizer step] weights grads for {name} are not synchronized across DP. {err}",
                        )

                if not config.general.ignore_sanity_checks:
                    # SANITY CHECK: Check that the model params are synchronized across dp
                    for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
                        assert_tensor_synced_across_pg(
                            tensor=param, pg=dpg.dp_pg, msg=lambda err: f"{name} are not synchronized across DP {err}"
                        )

                    # SANITY CHECK: Tied weights are synchronized
                    tied_params_list = sorted(
                        get_tied_id_to_param(
                            parameters=normalized_model.parameters(), root_module=normalized_model
                        ).items(),
                        key=lambda x: x[0],
                    )

                    for (name, group_ranks), param in tied_params_list:
                        group = dpg.world_ranks_to_pg[group_ranks]
                        assert_tensor_synced_across_pg(
                            tensor=param,
                            pg=group,
                            msg=lambda err: f"[Before optimizer step] Tied weights {name} are not synchronized. {err}",
                        )

                # Apply gradient
                optimizer.step()
                optimizer.zero_grad()

                # Update the learning rate
                lr_scheduler.step()

                if not config.general.ignore_sanity_checks:
                    # SANITY CHECK: Check that gradients is cleared
                    for name, param in model.named_parameters():
                        if not param.requires_grad:
                            continue

                        if param.grad is not None:
                            log_rank(
                                f"Process rank { dist.get_rank(dpg.world_pg)}/{dpg.world_pg.size()}: {name} still has gradient despite having ran the optimizer",
                                logger=logger,
                                level=logging.ERROR,
                            )

                # Training Logs
                iteration_step += 1
                consumed_train_samples += global_batch_size

                if iteration_step % config.logging.iteration_step_info_interval == 0:
                    # TODO @nouamanetazi: Megatron-LM seems to be using a barrier to report their interval time. Check if this is necessary. https://github.com/NouamaneTazi/Megatron-LM/blob/e241a96c3085b18e36c6cee1d68a8155de77b5a6/megatron/training.py#L607
                    dist.barrier()
                    torch.cuda.synchronize()
                    elapsed_time_per_iteration_ms = (
                        (time.time() - start_time) / config.logging.iteration_step_info_interval * 1000
                    )
                    tokens_per_sec = (
                        global_batch_size * sequence_length / (elapsed_time_per_iteration_ms / 1000)
                    )  # tokens_per_sec is calculated using sequence_length
                    try:
                        num_key_values_heads = model_config.num_key_value_heads
                    except AttributeError:
                        num_key_values_heads = model_config.num_attention_heads
                    model_tflops, hardware_tflops = get_flops_per_sec(
                        iteration_time_in_sec=elapsed_time_per_iteration_ms / 1000,
                        world_size=dpg.world_pg.size(),
                        num_layers=model_config.num_hidden_layers,
                        hidden_size=model_config.hidden_size,
                        num_heads=model_config.num_attention_heads,
                        num_key_value_heads=num_key_values_heads,
                        vocab_size=model_config.vocab_size,
                        seq_len=sequence_length,
                        batch_size=global_batch_size,
                        ffn_hidden_size=model_config.intermediate_size,
                        recompute_granularity=config.parallelism.recompute_granularity,
                    )
                    if (
                        dist.get_rank(dpg.pp_pg) == dpg.pp_pg.size() - 1
                        and dist.get_rank(dpg.tp_pg) == 0
                        and dist.get_rank(dpg.dp_pg) == 0
                    ):
                        # case where outputs is a list of dicts
                        if isinstance(outputs[0], dict):
                            # we only keep loss key
                            outputs = [output["loss"] for output in outputs]

                        assert all(not isinstance(output, TensorPointer) for output in outputs)
                        # This is an average on only one data rank.
                        loss_avg_per_mbs = torch.tensor(outputs).mean().item()
                        lr = lr_scheduler.get_last_lr()[0]

                        log_entries = [
                            LogItem("consumed_samples", iteration_step * global_batch_size, "12d"),
                            LogItem("elapsed_time_per_iteration_ms", elapsed_time_per_iteration_ms, ".1f"),
                            LogItem("tokens_per_sec", tokens_per_sec, "1.6E"),
                            LogItem("tokens_per_sec_per_gpu", tokens_per_sec / dpg.world_pg.size(), "1.6E"),
                            LogItem("global_batch_size", global_batch_size, "5d"),
                            LogItem("lm_loss", loss_avg_per_mbs, "1.6E"),
                            LogItem("lr", lr, ".3E"),
                            LogItem("model_tflops_per_gpu", model_tflops, ".2f"),
                            LogItem("hardware_tflops_per_gpu", hardware_tflops, ".2f"),
                        ]

                        if grad_norm_unclipped is not None:
                            log_entries.append(LogItem("grad_norm", grad_norm_unclipped, ".3f"))

                        if tb_writer is not None:
                            tb_writer.add_scalars_from_list(log_entries, iteration_step)
                        loggerwriter.add_scalars_from_list(log_entries, iteration_step)

                    start_time = time.time()

                # # Kill switch
                # if config.general.kill_switch_path.exists():
                #     log_rank(
                #         f"Detected kill switch at {config.general.kill_switch_path}. Exiting",
                #         logger=logger,
                #         level=logging.INFO,
                #         rank=0,
                #     )

                #     # Save checkpoint
                #     save_checkpoint(
                #         model=model,
                #         optimizer=optimizer,
                #         lr_scheduler=lr_scheduler,
                #         random_states=random_states,
                #         model_config=model_config,
                #         config=config,
                #         iteration_step=iteration_step,
                #         consumed_train_samples=consumed_train_samples,
                #         checkpoints_path=config.checkpoints.checkpoints_path,
                #         dpg=dpg,
                #         logger=logger,
                #     )

                #     # TODO @thomasw21: Do I need to return a barrier in order to be sure everyone saved before exiting.
                #     sys.exit(0)

                # Checkpoint
                if iteration_step % config.checkpoints.checkpoint_interval == 0:
                    save_checkpoint(
                        model=normalized_model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        random_states=random_states,
                        model_config=model_config,
                        config=config,
                        iteration_step=iteration_step,
                        consumed_train_samples=consumed_train_samples,
                        checkpoints_path=config.checkpoints.checkpoints_path,
                        dpg=dpg,
                        logger=logger,
                    )

                # Push to Hub
                if (
                    isinstance(config.logging.tensorboard_logger, HubLoggerConfig)
                    and isinstance(tb_writer, HubSummaryWriter)
                    and iteration_step % config.logging.tensorboard_logger.push_to_hub_interval == 0
                ):
                    # tb_writer only exists on a single rank
                    log_rank(
                        f"Push Tensorboard logging to Hub at iteration {iteration_step}",
                        logger=logger,
                        level=logging.INFO,
                    )
                    # it is a future that queues to avoid concurrent push
                    tb_writer.scheduler.trigger()

                if not config.general.ignore_sanity_checks:
                    # SANITY CHECK: Check that the model params are synchronized
                    for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
                        assert_tensor_synced_across_pg(
                            tensor=param, pg=dpg.dp_pg, msg=lambda err: f"{name} is not synced across DP. {err}"
                        )

                    # SANITY CHECK: Tied weights are synchronized
                    for (name, group_ranks), param in sorted(
                        get_tied_id_to_param(
                            parameters=normalized_model.parameters(), root_module=normalized_model
                        ).items(),
                        key=lambda x: x[0],
                    ):
                        group = dpg.world_ranks_to_pg[group_ranks]
                        assert_tensor_synced_across_pg(
                            tensor=param,
                            pg=group,
                            msg=lambda err: f"[After train] Tied weights {name} are not synchronized. {err}",
                        )


   # Test saving
    checkpoint_path = save_checkpoint(
        model=normalized_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        random_states=random_states,
        model_config=model_config,
        config=config,
        iteration_step=config.tokens.train_steps,
        consumed_train_samples=consumed_train_samples,
        checkpoints_path=config.checkpoints.checkpoints_path,
        dpg=dpg,
        logger=logger,
    )

    # Done training: we run a set of validation at the end
    log_rank(
        f"Finished training with iteration {config.tokens.train_steps}", logger=logger, level=logging.INFO, rank=0
    )

    # Wait til everything is saved to begin loading
    dist.barrier()

    # Test that saved checkpoint has save everything correctly.
    # NOTE: it's important to have same zero stage to be able to compare the optimizer state dict

    # Load model
    new_model = init_model(
        model_builder=lambda: LlamaForTraining(
            config=model_config,
            dpg=dpg,
            parallel_config=config.parallelism,
        ),
        model_config=model_config,
        parallel_config=config.parallelism,
        dtype=config.model.dtype,
        dpg=dpg,
        # TODO @thomasw21: Figure out why using DDP with accumulate_in_fp_32 and ZeRO-1 performs poorly.
        make_ddp=not (optimizer_args.accumulate_grad_in_fp32 and optimizer_args.zero_stage > 0),
        logger=logger
    )

    new_optimizer, new_grad_accumulator = init_optimizer_and_grad_accumulator(
        model=new_model, optimizer_args=optimizer_args, dpg=dpg
    )
    new_lr_scheduler = lr_scheduler_builder(
        optimizer=new_optimizer, learning_rate=optimizer_args.learning_rate, lr_scheduler_args=lr_scheduler_args
    )
    load(
        model=new_model.module if isinstance(new_model, DistributedDataParallel) else new_model,
        optimizer=new_optimizer,
        lr_scheduler=new_lr_scheduler,
        dpg=dpg,
        root_folder=checkpoint_path,
    )

    # SANITY CHECK: Check that the loaded model match
    test_equal_dict(new_model.state_dict(), model.state_dict())
    # SANITY CHECK: Check that the loaded optimizer match
    if optimizer.inherit_from(ZeroDistributedOptimizer):
        assert new_optimizer.inherit_from(ZeroDistributedOptimizer)
        # TODO @thomasw21: Check that the optimizer state corresponds to the non zero version
    else:
        assert not new_optimizer.inherit_from(ZeroDistributedOptimizer)
    test_equal_dict(new_optimizer.state_dict(), optimizer.state_dict())
    # SANITY CHECK: Check that the loaded optim scheduler match
    test_equal_dict(new_lr_scheduler.state_dict(), lr_scheduler.state_dict())

    # Check that it converges.

    # Check TFLOPS


if __name__ == "__main__":
    main()