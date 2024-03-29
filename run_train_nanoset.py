"""
Nanotron training script.

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml
```
"""
import argparse
from typing import Dict, cast

from nanotron import logging
from nanotron.config import (
    DataArgs,
    DatasetStageArgs,
)
from nanotron.data.dataloader_builder import build_nanoset_dataloader
from nanotron.data.dataset_builder import NanosetBuilder
from nanotron.data.nanoset import NanosetConfig
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from torch.utils.data import DataLoader

logger = logging.get_logger(__name__)


def get_dataloader_from_data_stage(trainer: DistributedTrainer, data: DataArgs) -> DataLoader:
    """Returns train, valid and test dataloaders"""

    # First, we need to know which ranks to feed the dataloader to
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    # Create Nanoset config
    nanoset_config = NanosetConfig(
        random_seed=data.seed,
        sequence_length=trainer.sequence_length,
        data_path=data.dataset.data_path,
        split=data.dataset.split,
        train_split_samples=trainer.config.tokens.train_steps * trainer.global_batch_size,
        path_to_cache=data.dataset.path_to_cache,
    )

    # Build Nanoset datasets
    train_dataset, valid_dataset, test_dataset = NanosetBuilder(nanoset_config).build()

    # Prepare train, valid and test dataloaders
    train_dataloader = build_nanoset_dataloader(
        train_dataset,
        trainer.sequence_length,
        parallel_context=trainer.parallel_context,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        micro_batch_size=trainer.micro_batch_size,
        consumed_train_samples=trainer.consumed_train_samples,
        dataloader_num_workers=data.num_loading_workers,
        dataloader_drop_last=True,
    )

    # Valid dataloader
    _ = build_nanoset_dataloader(
        valid_dataset,
        trainer.sequence_length,
        parallel_context=trainer.parallel_context,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        micro_batch_size=trainer.micro_batch_size,
        dataloader_num_workers=data.num_loading_workers,
        dataloader_drop_last=True,
    )

    # Test dataloader
    _ = build_nanoset_dataloader(
        test_dataset,
        trainer.sequence_length,
        parallel_context=trainer.parallel_context,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        micro_batch_size=trainer.micro_batch_size,
        dataloader_num_workers=data.num_loading_workers,
        dataloader_drop_last=True,
    )

    return train_dataloader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


def get_dataloader(trainer: DistributedTrainer) -> Dict[str, DataLoader]:
    sorted_stages = sorted(trainer.config.data_stages, key=lambda stage: stage.start_training_step)
    dataloaders = {}
    for idx, stage in enumerate(sorted_stages):
        # NOTE: we only create the dataloader for the first stage,
        # then we lazy initialize the dataloader for the other stages
        stage = cast(DatasetStageArgs, stage)
        dataloader = (
            get_dataloader_from_data_stage(trainer, stage.data)
            if idx == 0
            else lambda stage=stage: get_dataloader_from_data_stage(trainer, stage.data)
        )
        dataloaders[stage.name] = dataloader
    return dataloaders


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load trainer and data
    trainer = DistributedTrainer(config_file)

    train_dataloader = get_dataloader(trainer)

    # Train
    trainer.train(train_dataloader)
