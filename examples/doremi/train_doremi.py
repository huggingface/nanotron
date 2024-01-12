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
from nanotron.logging import log_rank
from nanotron.trainer import DistributedTrainer
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer
from transformers import __version__ as tf_version

logger = logging.get_logger(__name__)


# TODO(xrsrke): make this configurable
NUM_DOMAINS = 5


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

    # Load trainer and data
    trainer = DistributedTrainer(config_file)
    dataloader = get_dataloader(trainer, args.sanity_check_dataloader_interval)

    # Train
    # trainer.train(dataloader)

    if isinstance(dataloader, tuple):
        dataloader[1] if len(dataloader) > 1 else None
        dataloader[2] if len(dataloader) > 2 else None
        dataloader = dataloader[0]
    else:
        dataloader = dataloader

    from nanotron.dataloader import sanity_check_dataloader

    dataloader = sanity_check_dataloader(dataloader=dataloader, dpg=trainer.dpg, config=trainer.config)

    # Log data config
    # self.log_object(data_config_log, name="data_config")

    trainer.pipeline_engine = trainer.config.parallelism.pp_engine

    trainer.pipeline_engine.nb_microbatches = trainer.n_micro_batches_per_batch

    import datetime

    log_rank(
        f"[Start training] datetime: {datetime.datetime.now()} | mbs: {trainer.micro_batch_size} | grad_accum: {trainer.n_micro_batches_per_batch} | global_batch_size: {trainer.global_batch_size} | sequence_length: {trainer.sequence_length} | train_steps: {trainer.config.tokens.train_steps} | start_iteration_step: {trainer.start_iteration_step} | consumed_train_samples: {trainer.consumed_train_samples}",  # noqa
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    # Kill switch
    trainer.check_kill_switch(save_ckpt=False)

    # TODO @nouamanetazi: refactor this
    # Useful mapping
    trainer.normalized_model = (
        trainer.model.module if isinstance(trainer.model, DistributedDataParallel) else trainer.model
    )
    trainer.module_id_to_prefix = {
        id(module): f"{module_name}." for module_name, module in trainer.normalized_model.named_modules()
    }
    # Fix the root_model
    trainer.module_id_to_prefix[id(trainer.normalized_model)] = ""

    # iteration_step = 10

    import time

    from nanotron.helpers import get_profiler

    prof = get_profiler(config=trainer.config)
    torch.cuda.empty_cache()

    with prof:
        for trainer.iteration_step in range(trainer.start_iteration_step + 1, trainer.config.tokens.train_steps + 1):
            if isinstance(prof, torch.profiler.profile):
                prof.step()
            trainer.iteration_start_time = time.time()

            # Training step
            outputs, loss_avg = trainer.training_step(dataloader=dataloader)

            # Training Logs
            trainer.consumed_train_samples += trainer.global_batch_size

            if (trainer.iteration_step - 1) % trainer.config.logging.iteration_step_info_interval == 0:
                trainer.train_step_logs(outputs=outputs, loss_avg=loss_avg)

            # Kill switch
            trainer.check_kill_switch(save_ckpt=True)

            # Checkpoint
            if trainer.iteration_step % trainer.config.checkpoints.checkpoint_interval == 0:
                trainer.save_checkpoint()

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
    #     self.s3_mover.distributed_wait_for_completion(group=self.dpg.world_pg)
