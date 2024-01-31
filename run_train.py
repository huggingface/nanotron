"""
Nanotron training script.

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=8 run_train.py --config-file examples/debug_run_train.yaml
```
"""
import argparse

from nanotron import logging
from nanotron.config import (
    PretrainDatasetsArgs,
)
from nanotron.dataloader import (
    clm_process,
    dummy_infinite_data_generator,
    get_datasets,
    get_train_dataloader,
)
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from nanotron.utils import (
    main_rank_first,
)

try:
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoTokenizer
    from transformers import __version__ as tf_version
except ImportError:
    hf_hub_version = None
    tf_version = None

logger = logging.get_logger(__name__)


def get_dataloader(trainer: DistributedTrainer):
    """Returns a dataloader for training."""

    # First, we need to know which ranks to feed the dataloader to
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    # Case 1: Dummy data generator
    if trainer.config.data.dataset is None:
        log_rank("Using dummy data generator", logger=logger, level=logging.INFO, rank=0)
        dataloader = dummy_infinite_data_generator(
            micro_batch_size=trainer.micro_batch_size,
            sequence_length=trainer.sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            vocab_size=trainer.model_config.vocab_size,
            seed=trainer.config.data.seed,
            parallel_context=trainer.parallel_context,
        )()

    # Case 2: HuggingFace datasets
    elif isinstance(trainer.config.data.dataset, PretrainDatasetsArgs):
        log_rank("Using `datasets` library", logger=logger, level=logging.INFO, rank=0)
        tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
        log_rank(
            f"Loading tokenizer from {tokenizer_path} and transformers/hf_hub versions {tf_version, hf_hub_version}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        # We need to the 1st device to process dataset and cache it, then other devices load from cache
        with main_rank_first(trainer.parallel_context.world_pg):
            # TODO @nouamanetazi: this may timeout before 1st device finishes processing dataset. Can we have a ctxmanager to modify timeout?
            # TODO: generalise to include  for validation/test splits

            # We load the raw dataset
            raw_dataset = get_datasets(
                hf_dataset_or_datasets=trainer.config.data.dataset.hf_dataset_or_datasets,
                splits=trainer.config.data.dataset.hf_dataset_splits,
            )["train"]

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            # We apply the Causal Language Modeling preprocessing
            train_dataset = clm_process(
                raw_dataset=raw_dataset,
                tokenizer=tokenizer,
                text_column_name=trainer.config.data.dataset.text_column_name,
                dataset_processing_num_proc_per_process=trainer.config.data.dataset.dataset_processing_num_proc_per_process,
                dataset_overwrite_cache=trainer.config.data.dataset.dataset_overwrite_cache,
                sequence_length=trainer.sequence_length,
            )

            # We load the processed dataset on the ranks requiring it
            dataloader = get_train_dataloader(
                train_dataset=train_dataset,
                sequence_length=trainer.sequence_length,
                parallel_context=trainer.parallel_context,
                input_pp_rank=input_pp_rank,
                output_pp_rank=output_pp_rank,
                micro_batch_size=trainer.micro_batch_size,
                consumed_train_samples=trainer.consumed_train_samples,
                dataloader_num_workers=trainer.config.data.num_loading_workers,
                seed_worker=trainer.config.data.seed,
                dataloader_drop_last=True,
            )
            # Check if we have enough samples for train_steps
            assert (
                trainer.config.tokens.train_steps - trainer.start_iteration_step
            ) * trainer.global_batch_size // trainer.parallel_context.dp_pg.size() <= len(dataloader), (
                f"Dataset is too small for steps ({len(dataloader)} < {(trainer.config.tokens.train_steps - trainer.start_iteration_step) * trainer.global_batch_size // trainer.parallel_context.dp_pg.size()}), "
                f"Try train_steps<={len(dataloader) * trainer.parallel_context.dp_pg.size() // trainer.global_batch_size + trainer.start_iteration_step}"
            )
    else:
        raise ValueError(f"Unhandled case of `self.config.data.dataset`. Got: {trainer.config.data.dataset}")

    return dataloader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load trainer and data
    trainer = DistributedTrainer(config_file)
    dataloader = get_dataloader(trainer)

    # Train
    trainer.train(dataloader)
