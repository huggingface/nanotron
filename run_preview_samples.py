"""
Tool to preview first samples used for training

https://github.com/huggingface/nanotron/issues/184

Usage demonstrated in examples/run_preview_samples.sh. 
"""

import argparse

from typing import Type, Union, cast

import numpy as np
from nanotron import logging
from nanotron.config import DataArgs, DatasetStageArgs, NanosetDatasetsArgs, PretrainDatasetsArgs
from nanotron.data.dataloader_builder import build_nanoset_dataloader
from nanotron.dataloader import (
    clm_process,
    get_datasets,
    get_train_dataloader,
)

from nanotron.parallel import ParallelContext
from nanotron.config import Config
from nanotron.logging import log_rank
from nanotron.logging import warn_once
from nanotron.trainer import DistributedTrainer
import nanotron.trainer 
from torch.utils.data import DataLoader

try:
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoTokenizer
    from transformers import __version__ as tf_version
except ImportError:
    hf_hub_version = None
    tf_version = None

logger = logging.get_logger(__name__)


def _get_dataloader_and_tokenizer_from_data(
    data: DataArgs,
    config: Config,
    parallel_context: ParallelContext
):
    sequence_length = config.tokens.sequence_length
    micro_batch_size = config.tokens.micro_batch_size
    consumed_train_samples = 0
    input_pp_rank, output_pp_rank = 0, int(parallel_context.pp_pg.size() - 1)

    # HuggingFace datasets
    if isinstance(data.dataset, PretrainDatasetsArgs):
        log_rank("Using `datasets` library", logger=logger, level=logging.INFO, rank=0)
        tokenizer_path = config.tokenizer.tokenizer_name_or_path
        log_rank(
            f"Loading tokenizer from {tokenizer_path} and transformers/hf_hub versions {tf_version, hf_hub_version}",
            logger=logger,
            level=logging.INFO,
            rank=0)

        raw_dataset = get_datasets(
            hf_dataset_or_datasets=data.dataset.hf_dataset_or_datasets,
            hf_dataset_config_name=data.dataset.hf_dataset_config_name,
            splits=data.dataset.hf_dataset_splits,
        )["train"]

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # We apply the Causal Language Modeling preprocessing
        train_dataset = clm_process(
            raw_dataset=raw_dataset,
            tokenizer=tokenizer,
            text_column_name=data.dataset.text_column_name,
            dataset_processing_num_proc_per_process=data.dataset.dataset_processing_num_proc_per_process,
            dataset_overwrite_cache=data.dataset.dataset_overwrite_cache,
            sequence_length=sequence_length,
        )

        dataloader = get_train_dataloader(
            train_dataset=train_dataset,
            sequence_length=sequence_length,
            parallel_context=parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=micro_batch_size,
            consumed_train_samples=consumed_train_samples,
            dataloader_num_workers=data.num_loading_workers,
            seed_worker=data.seed,
            dataloader_drop_last=True,
        )
    elif isinstance(data.dataset, NanosetDatasetsArgs):
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_name_or_path)
        token_dtype = np.int32 if len(tokenizer) > np.iinfo(np.uint16).max + 1 else np.uint16

        from nanotron.data.nanoset import Nanoset

        train_dataset = Nanoset(
            dataset_paths=data.dataset.dataset_path,
            dataset_weights=data.dataset.dataset_weights,
            sequence_length=sequence_length,
            token_dtype=token_dtype,
            train_split_num_samples=config.tokens.train_steps,
            random_seed=data.seed,
        )

        # Prepare dataloader
        dataloader = build_nanoset_dataloader(
            train_dataset,
            sequence_length,
            parallel_context=parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=micro_batch_size,
            consumed_train_samples=consumed_train_samples,
            dataloader_num_workers=data.num_loading_workers,
            dataloader_drop_last=True,
        )
    else:
        raise ValueError(f"Unhandled case of `self.config.data.dataset`. Got: {data.dataset}")
    return dataloader, tokenizer


def check_dataloader_from_config(config_or_config_file: Union[Config, str],
                                 config_class: Type[Config] = Config,
                                 n: int = 6):
    config = nanotron.trainer.get_config_from_file(config_or_config_file, config_class)
    parallel_context = ParallelContext(
            tensor_parallel_size=config.parallelism.tp,
            pipeline_parallel_size=config.parallelism.pp,
            data_parallel_size=config.parallelism.dp,
            expert_parallel_size=config.parallelism.expert_parallel_size,
        )
    for stage_idx, stage in enumerate(config.data_stages):
        stage = cast(DatasetStageArgs, stage)
        warn_once(f"{stage.name}: {stage.data}", logger=logger, rank=0)

        dataloader, tokenizer = _get_dataloader_and_tokenizer_from_data(stage.data, config, parallel_context)
        checked_dataloader = nanotron.trainer.sanity_check_dataloader(dataloader, parallel_context, config)

        if 0 == nanotron.distributed.get_rank(parallel_context.pp_pg):
            for _ in range(n):
                x = next(checked_dataloader)
                to_decode = x["input_ids"]
                warn_once(str(tokenizer.batch_decode(to_decode)), logger=logger, rank=0)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    parser.add_argument("-n", type=int, required=False, default=6, help="Number of rows to print")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file
    n = args.n
    dataloader = check_dataloader_from_config(config_file, Config, n)

