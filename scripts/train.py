"""

You can run using command:
```
USE_FAST=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 scripts/train.py --config-file configs/config.yaml
```
"""
import argparse
from typing import Dict, Iterator, Union

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer

from nanotron.config import (
    Config,
    PretrainDatasetsArgs,
    PretrainNemoArgs,
    get_args_from_path,
)
from nanotron.core import logging
from nanotron.core.logging import log_rank
from nanotron.core.parallelism.pipeline_parallelism.tensor_pointer import TensorPointer
from nanotron.core.utils import (
    main_rank_first,
)
from nanotron.dataloaders.dataloader import (
    clm_process,
    dummy_infinite_data_generator,
    get_datasets,
    get_train_dataloader,
)
from nanotron.dataloaders.nemo import get_nemo_dataloader, get_nemo_datasets
from nanotron.trainer import DistributedTrainer

logger = logging.get_logger(__name__)


def get_dataloader(trainer) -> Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]:
    # Prepare dataloader
    tokenizer = AutoTokenizer.from_pretrained(trainer.config.model.hf_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if isinstance(trainer.model, DistributedDataParallel):
        input_pp_rank = trainer.model.module.input_pp_rank
        output_pp_rank = trainer.model.module.output_pp_rank
    else:
        input_pp_rank = trainer.model.input_pp_rank
        output_pp_rank = trainer.model.output_pp_rank

    if config.data.dataset is None:
        dataloader = dummy_infinite_data_generator(
            micro_batch_size=trainer.micro_batch_size,
            sequence_length=trainer.sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            vocab_size=trainer.model_config.vocab_size,
            seed=trainer.config.data.seed,
            dpg=trainer.dpg,
        )()
    elif isinstance(config.data.dataset, PretrainNemoArgs):
        log_rank("Using Nemo Dataloader", logger=logger, level=logging.INFO, rank=0)

        train_dataset, valid_dataset, test_datasets = get_nemo_datasets(
            config=config.data.dataset,
            global_batch_size=trainer.global_batch_size,
            sequence_length=config.tokens.sequence_length,
            train_steps=config.tokens.train_steps,
            limit_val_batches=config.tokens.limit_val_batches,
            val_check_interval=config.tokens.val_check_interval,
            test_iters=config.tokens.limit_test_batches,
            seed=config.data.seed,
            dpg=trainer.dpg,
        )
        dataloader = get_nemo_dataloader(
            dataset=train_dataset,
            sequence_length=trainer.sequence_length,
            micro_batch_size=trainer.micro_batch_size,
            global_batch_size=trainer.global_batch_size,
            num_workers=config.data.num_loading_workers,
            cfg=config.data.dataset,
            consumed_samples=trainer.consumed_train_samples,
            dpg=trainer.dpg,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            dataloader_drop_last=True,
        )
    elif isinstance(config.data.dataset, PretrainDatasetsArgs):
        log_rank("Using `datasets` library", logger=logger, level=logging.INFO, rank=0)

        with main_rank_first(trainer.dpg.world_pg):
            # 1st device processes dataset and cache it, then other devices load from cache
            # TODO @nouamanetazi: this may timeout before 1st device finishes processing dataset. Can we have a ctxmanager to modify timeout?
            # TODO: generalise to include  for validation/test splits
            raw_dataset = get_datasets(
                dataset_mixer=config.data.dataset.hf_dataset_mixer, splits=config.data.dataset.hf_dataset_splits
            )["train"]
            tokenizer = AutoTokenizer.from_pretrained(trainer.config.model.hf_model_name)

            train_dataset = clm_process(
                raw_dataset=raw_dataset,
                tokenizer=tokenizer,
                text_column_name=config.data.dataset.text_column_name,
                dataset_processing_num_proc_per_process=config.data.dataset.dataset_processing_num_proc_per_process,
                dataset_overwrite_cache=config.data.dataset.dataset_overwrite_cache,
                sequence_length=trainer.sequence_length,
            )
            dataloader = get_train_dataloader(
                train_dataset=train_dataset,
                sequence_length=trainer.sequence_length,
                dpg=trainer.dpg,
                input_pp_rank=input_pp_rank,
                output_pp_rank=output_pp_rank,
                micro_batch_size=trainer.micro_batch_size,
                consumed_train_samples=trainer.consumed_train_samples,
                dataloader_num_workers=config.data.num_loading_workers,
                seed_worker=config.data.seed,
                dataloader_drop_last=True,
            )
            # Check if we have enough samples for train_steps
            assert (
                config.tokens.train_steps - trainer.start_iteration_step
            ) * trainer.global_batch_size // trainer.dpg.dp_pg.size() < len(
                dataloader
            ), f"Dataset is too small for steps ({len(dataloader)} < {(config.tokens.train_steps - trainer.start_iteration_step) * trainer.global_batch_size // trainer.dpg.dp_pg.size()}), Try train_steps<={len(dataloader) * trainer.dpg.dp_pg.size() // trainer.global_batch_size + trainer.start_iteration_step}"

    else:  # TODO: other datasets
        raise ValueError(f"Unhandled case of `self.config.data.dataset`. Got: {trainer.config.data.dataset}")

    return dataloader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML config file")
    return parser.parse_args()


if __name__ == "__main__":
    config_file = get_args().config_file
    config: Config = get_args_from_path(config_file)
    trainer = DistributedTrainer(config=config)

    dataloader = get_dataloader(trainer)
    trainer.train(dataloader=dataloader)
