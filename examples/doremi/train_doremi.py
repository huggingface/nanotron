"""

You can run using command:
```
USE_FAST=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 examples/doremi/train_doremi.py --config-file examples/doremi/tiny_llama.yaml
```
"""
import argparse
from typing import Optional

import torch
import torch.nn.functional as F
from dataloader import (
    clm_process,
    get_datasets,
    get_train_dataloader,
)
from huggingface_hub import __version__ as hf_hub_version
from nanotron import logging
from nanotron.config import (
    PretrainDatasetsArgs,
)
from nanotron.core import distributed as dist
from nanotron.core.utils import (
    main_rank_first,
)
from nanotron.dataloader import sanity_check_dataloader
from nanotron.logging import log_rank
from nanotron.models.fast.llama import LlamaForTraining, LlamaModel
from nanotron.trainer import DistributedTrainer
from torch.nn.parallel import DistributedDataParallel
from trainer import DoReMiTrainer
from transformers import AutoTokenizer
from transformers import __version__ as tf_version

logger = logging.get_logger(__name__)


# TODO(xrsrke): make this configurable
NUM_DOMAINS = 5

CONFIG_TO_MODEL_TRAINING_CLASS = {
    "LlamaConfig": LlamaForTraining,
}
CONFIG_TO_MODEL_INFERENCE_CLASS = {
    "LlamaConfig": LlamaModel,
}


def get_dataloader(trainer: DistributedTrainer, sanity_check_dataloader_interval: Optional[int] = None):
    # Prepare dataloader
    tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
    log_rank(
        f"Loading tokenizer from {tokenizer_path} and transformers/hf_hub versions {tf_version, hf_hub_version}",
        logger=logger,
        level=logging.INFO,
        rank=None,
    )

    if isinstance(trainer.model, DistributedDataParallel):
        input_pp_rank = trainer.model.module.input_pp_rank
        output_pp_rank = trainer.model.module.output_pp_rank
    else:
        input_pp_rank = trainer.model.input_pp_rank
        output_pp_rank = trainer.model.output_pp_rank

    # TODO(xrsrke): remove this
    if trainer.config.data.dataset is None:
        # dataloader = dummy_infinite_data_generator(
        #     micro_batch_size=trainer.micro_batch_size,
        #     sequence_length=trainer.sequence_length,
        #     input_pp_rank=input_pp_rank,
        #     output_pp_rank=output_pp_rank,
        #     vocab_size=trainer.model_config.vocab_size,
        #     seed=trainer.config.data.seed,
        #     dpg=trainer.dpg,
        # )()
        pass

    elif isinstance(trainer.config.data.dataset, PretrainDatasetsArgs):
        log_rank("Using `datasets` library", logger=logger, level=logging.INFO, rank=0)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        with main_rank_first(trainer.dpg.world_pg):
            # 1st device processes dataset and cache it, then other devices load from cache
            # TODO @nouamanetazi: this may timeout before 1st device finishes processing dataset. Can we have a ctxmanager to modify timeout?
            # TODO: generalise to include  for validation/test splits

            # TODO(xrsrke): support load a generic doremi dataset
            raw_dataset = get_datasets(
                hf_dataset_or_datasets=trainer.config.data.dataset.hf_dataset_or_datasets,
                splits=trainer.config.data.dataset.hf_dataset_splits,
            )["train"]
            tokenizer = AutoTokenizer.from_pretrained(trainer.config.tokenizer.tokenizer_name_or_path)
            tokenizer.pad_token = tokenizer.eos_token

            # NUM_DOMAINS = trainer.config.doremi.num_domains
            # raw_datasets = {f"domain_{i+1}": raw_dataset for i in range(NUM_DOMAINS)}
            raw_datasets = [raw_dataset for i in range(NUM_DOMAINS)]

            train_datasets = []
            # TODO(xrsrke): parallelize this
            for raw_dataset in raw_datasets:
                train_dataset = clm_process(
                    raw_dataset=raw_dataset,
                    tokenizer=tokenizer,
                    text_column_name=trainer.config.data.dataset.text_column_name,
                    dataset_processing_num_proc_per_process=trainer.config.data.dataset.dataset_processing_num_proc_per_process,
                    dataset_overwrite_cache=trainer.config.data.dataset.dataset_overwrite_cache,
                    sequence_length=trainer.sequence_length,
                )
                train_datasets.append(train_dataset)

            domain_weights = F.softmax(torch.ones(NUM_DOMAINS, requires_grad=False), dim=-1)

            dataloader = get_train_dataloader(
                domain_weights=domain_weights,
                ref_model=trainer.ref_model,
                train_datasets=train_datasets,
                sequence_length=trainer.sequence_length,
                dpg=trainer.dpg,
                input_pp_rank=input_pp_rank,
                output_pp_rank=output_pp_rank,
                micro_batch_size=trainer.micro_batch_size,
                consumed_train_samples=trainer.consumed_train_samples,
                dataloader_num_workers=trainer.config.data.num_loading_workers,
                seed_worker=trainer.config.data.seed,
                dataloader_drop_last=True,
            )

            for batch in dataloader:
                assert 1 == 1

            assert 1 == 1
            # TODO(xrsrke): investigate why this fail, and add it back
            # Check if we have enough samples for train_steps
            # assert (
            #     trainer.config.tokens.train_steps - trainer.start_iteration_step
            # ) * trainer.global_batch_size // trainer.dpg.dp_pg.size() < len(dataloader), (
            #     f"Dataset is too small for steps ({len(dataloader)} < {(trainer.config.tokens.train_steps - trainer.start_iteration_step) * trainer.global_batch_size // trainer.dpg.dp_pg.size()}), "
            #     f"Try train_steps<={len(dataloader) * trainer.dpg.dp_pg.size() // trainer.global_batch_size + trainer.start_iteration_step}"
            # )
    else:
        raise ValueError(f"Unhandled case of `self.config.data.dataset`. Got: {trainer.config.data.dataset}")

    # SANITY dataloder first samples
    if sanity_check_dataloader_interval is not None:
        NUM_BATCHES = 10
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, revision=trainer.config.model.tokenizer_revision)
        if dist.get_rank() == 0:
            check_step = -1

            with open("sanity_check.txt", "w") as f:
                f.write("")
            for i, batch in enumerate(dataloader):
                check_step += 1
                if i % sanity_check_dataloader_interval == 0:
                    with open("sanity_check.txt", "a") as f:
                        f.write("\n\n")
                        f.write("*" * 40)
                        f.write(f"Sanity check {check_step}")
                        f.write("*" * 40)
                    print(batch)

                    # joblib.dump(batch, f"sanity_check_{check_step}.pkl")
                    texts = tokenizer.batch_decode(
                        batch["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False
                    )

                    for j, text in enumerate(texts):
                        # if j != 1:
                        #     continue
                        print(f"\n\n>>Batch {i} || Sample {j}<<\n")
                        print(text[:400])
                        with open("sanity_check.txt", "a") as f:
                            f.write(f"\n\n>>Batch {i} || Sample {j}<<\n")
                            f.write(text)

                    if i // sanity_check_dataloader_interval == NUM_BATCHES - 1:
                        break
            assert False
        dist.barrier()
    return dataloader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    parser.add_argument("--job-id", type=str, help="Optional job name")
    parser.add_argument(
        "--sanity-check-dataloader-interval",
        type=int,
        default=None,
        help="Optional interval to print dataloader samples",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    trainer = DoReMiTrainer(config_file)
    dataloader = get_dataloader(trainer, args.sanity_check_dataloader_interval)
    dataloader = sanity_check_dataloader(dataloader=dataloader, dpg=trainer.dpg, config=trainer.config)

    # config = trainer.config
    # model_config = config.model.model_config
    # model_config_cls = model_config.__class__.__name__
    # parallel_config = config.parallelism

    # dpg = get_process_groups(
    #     data_parallel_size=parallel_config.dp,
    #     pipeline_parallel_size=parallel_config.pp,
    #     tensor_parallel_size=parallel_config.tp,
    # )

    # log_rank(
    #     f"[DoReMi] datetime: {datetime.datetime.now()} | Initializing reference model",  # noqa
    #     logger=logger,
    #     level=logging.INFO,
    #     rank=0,
    # )

    # # from nanotron.trainer import CONFIG_TO_MODEL_CLASS
    # ref_model = DistributedTrainer.build_model(
    #     model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config_cls](
    #         config=model_config,
    #         dpg=trainer.dpg,
    #         parallel_config=parallel_config,
    #         # NOTE: ref_model uses the same random states as the model being trained
    #         random_states=trainer.random_states,
    #     ),
    #     dtype=config.model.dtype,
    #     dpg=trainer.dpg,
    # )
    # ref_model.eval()

    trainer.train(dataloader)

    # trainer.pipeline_engine = trainer.config.parallelism.pp_engine
    # trainer.pipeline_engine.nb_microbatches = trainer.n_micro_batches_per_batch

    # assert 1 == 1

    # log_rank(
    #     f"[Start training] datetime: {datetime.datetime.now()} | mbs: {trainer.micro_batch_size} | grad_accum: {trainer.n_micro_batches_per_batch} | global_batch_size: {trainer.global_batch_size} | sequence_length: {trainer.sequence_length} | train_steps: {trainer.config.tokens.train_steps} | start_iteration_step: {trainer.start_iteration_step} | consumed_train_samples: {trainer.consumed_train_samples}",  # noqa
    #     logger=logger,
    #     level=logging.INFO,
    #     rank=0,
    # )
    # # Kill switch
    # trainer.check_kill_switch(save_ckpt=False)

    # trainer.normalized_model = (
    #     trainer.model.module if isinstance(trainer.model, DistributedDataParallel) else trainer.model
    # )
    # trainer.module_id_to_prefix = {
    #     id(module): f"{module_name}." for module_name, module in trainer.normalized_model.named_modules()
    # }
    # trainer.module_id_to_prefix[id(trainer.normalized_model)] = ""
    # torch.cuda.empty_cache()

    # from nanotron.core.clip_grads import clip_grad_norm
    # from nanotron.core.parallel.pipeline_parallelism.engine import OneForwardOneBackwardPipelineEngine
    # from nanotron.core.parallel.tied_parameters import sync_tied_weights_gradients

    # ref_pipeline_engine = OneForwardOneBackwardPipelineEngine()
    # for trainer.iteration_step in range(trainer.start_iteration_step + 1, trainer.config.tokens.train_steps + 1):
    #     trainer.iteration_start_time = time.time()

    #     # NOTE: in nanotron, a training step = an epoch
    #     # outputs, loss_avg = trainer.training_step(dataloader=dataloader)

    #     # if trainer.iteration_step < 5:
    #     #     log_rank(
    #     #         f"[Before train batch iter] Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MiB."
    #     #         f" Peak allocated {torch.cuda.max_memory_allocated() / 1024**2:.2f}MiB."
    #     #         f" Peak reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MiB",
    #     #         logger=logger,
    #     #         level=logging.INFO,
    #     #         group=trainer.dpg.world_pg,
    #     #         rank=0,
    #     #     )
    #     #     torch.cuda.reset_peak_memory_stats()

    #     # with torch.no_grad():
    #     #     pass

    #     ref_outputs = ref_pipeline_engine.train_batch_iter(
    #         model=ref_model,
    #         pg=trainer.dpg.pp_pg,
    #         batch=(next(dataloader) for _ in range(trainer.n_micro_batches_per_batch)),
    #         nb_microbatches=trainer.n_micro_batches_per_batch,
    #         grad_accumulator=trainer.grad_accumulator,
    #     )
    #     outputs = trainer.pipeline_engine.train_batch_iter(
    #         model=trainer.model,
    #         pg=trainer.dpg.pp_pg,
    #         batch=(next(dataloader) for _ in range(trainer.n_micro_batches_per_batch)),
    #         nb_microbatches=trainer.n_micro_batches_per_batch,
    #         grad_accumulator=trainer.grad_accumulator,
    #     )

    #     # if trainer.iteration_step < 5:
    #     #     log_rank(
    #     #         f"[After train batch iter] Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MiB."
    #     #         f" Peak allocated {torch.cuda.max_memory_allocated() / 1024**2:.2f}MiB."
    #     #         f" Peak reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MiB",
    #     #         logger=logger,
    #     #         level=logging.INFO,
    #     #         group=trainer.dpg.world_pg,
    #     #         rank=0,
    #     #     )
    #     #     torch.cuda.reset_peak_memory_stats()

    #     # trainer.after_tbi_sanity_checks()

    #     if isinstance(trainer.model, DistributedDataParallel) and trainer.grad_accumulator is not None:
    #         # Wait for fp32 grads allreduce to finish to make sure grads are synced across DP
    #         assert (
    #             trainer.grad_accumulator.fp32_grads_allreduce_handle is not None
    #         ), "No fp32_grads_allreduce_handle maybe you're using only a single training process"
    #         trainer.grad_accumulator.fp32_grads_allreduce_handle.wait()

    #     # Sync tied weights
    #     if not isinstance(trainer.model, DistributedDataParallel):
    #         # Manually sync across DP if it's not handled by DDP
    #         sync_gradients_across_dp(
    #             module=trainer.model,
    #             dp_pg=trainer.dpg.dp_pg,
    #             reduce_op=dist.ReduceOp.AVG,
    #             # TODO @thomasw21: This is too memory hungry, instead we run all_reduce
    #             reduce_scatter=False,  # optimizer.inherit_from(ZeroDistributedOptimizer),
    #             grad_accumulator=trainer.grad_accumulator,
    #         )

    #     # TODO @nouamane: Put this in hooks so we can overlap communication with gradient computation on the last backward pass.
    #     sync_tied_weights_gradients(
    #         module=trainer.normalized_model,
    #         dpg=trainer.dpg,
    #         grad_accumulator=trainer.grad_accumulator,
    #     )

    #     # Clip gradients
    #     if trainer.config.optimizer.clip_grad is not None:
    #         # Normalize DDP
    #         named_parameters = [
    #             (
    #                 param.get_tied_info().get_full_name_from_module_id_to_prefix(
    #                     module_id_to_prefix=trainer.module_id_to_prefix
    #                 )
    #                 if param.is_tied
    #                 else name,
    #                 param,
    #             )
    #             for name, param in trainer.normalized_model.named_parameters()
    #             if param.requires_grad
    #         ]
    #         # TODO @nouamane: we need to split `world_rank_matrix` along PP axis, to separate ref from active model
    #         trainer.grad_norm_unclipped = clip_grad_norm(
    #             mp_pg=trainer.dpg.world_ranks_to_pg[
    #                 tuple(sorted(trainer.dpg.world_rank_matrix[:, dist.get_rank(trainer.dpg.dp_pg), :].reshape(-1)))
    #             ],
    #             named_parameters=named_parameters,
    #             grad_accumulator=trainer.grad_accumulator,
    #             max_norm=trainer.config.optimizer.clip_grad,
    #         )

    #     trainer.before_optim_step_sanity_checks()

    #     # Compute DP average loss and overlap with optimizer step
    #     if isinstance(outputs[0]["loss"], torch.Tensor):
    #         assert 1 == 1
    #         # This is an average on only one data rank.
    #         loss_avg = torch.stack(
    #             [output["loss"] for output in outputs]
    #         ).sum()  # already divided by n_micro_batches_per_batch
    #         # sync loss across DP
    #         handle = dist.all_reduce(loss_avg, group=trainer.dpg.dp_pg, async_op=True, op=dist.ReduceOp.AVG)
    #     else:
    #         loss_avg = None
    #         handle = None

    #     # Apply gradient
    #     trainer.optimizer.step()
    #     trainer.optimizer.zero_grad()

    #     # Update the learning rate
    #     trainer.lr_scheduler.step()

    #     trainer.after_optim_step_sanity_checks()

    #     if handle is not None:
    #         handle.wait()

    #     # with torch.no_grad():
    #     #     state = PipelineTrainBatchState()  # TODO: do i need state?
    #     #     nb_microbatches = trainer.n_micro_batches_per_batch

    #     #     outputs = []

    #     #     with attach_pipeline_state_to_model(model=ref_model, pipeline_state=state):
    #     #         # All forward
    #     #         for micro_batch in batch:
    #     #             context = self._get_fwd_context(model=ref_model)
    #     #             output = self.forward(context=context, state=state, micro_batch=micro_batch, model=model)
    #     #             # TODO @thomasw21: Somehow this needs to be done somewhere else to support interleaving. Somewhere right after a "stage"
    #     #             for _ in range(len(state.microbatches_activations_to_send)):
    #     #                 send_activation = state.microbatches_activations_to_send.popleft()
    #     #                 # Execute
    #     #                 send_activation()

    #     #             # We make `output` a dict
    #     #             if not isinstance(output, dict):
    #     #                 output = {"loss": output}

    #     #             # Store the loss for each microbatch
    #     #             if not isinstance(output["loss"], TensorPointer):
    #     #                 output = {k: v.detach() for k, v in output.items()}
    #     #             outputs.append(output)

    #     # Training Logs
    #     trainer.consumed_train_samples += trainer.global_batch_size

    #     if (trainer.iteration_step - 1) % trainer.config.logging.iteration_step_info_interval == 0:
    #         trainer.train_step_logs(outputs=outputs, loss_avg=loss_avg)

    #     # # Kill switch
    #     # trainer.check_kill_switch(save_ckpt=True)

    #     # Checkpoint
    #     if trainer.iteration_step % trainer.config.checkpoints.checkpoint_interval == 0:
    #         trainer.save_checkpoint()
