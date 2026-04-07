"""
Nanotron training script.

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml
```
"""
import argparse
import time
from pprint import pformat
from typing import Dict, Optional, cast

import nanotron.distributed as dist
from nanotron import logging
from nanotron.config import (
    DataArgs,
    DatasetStageArgs,
    NanosetDatasetsArgs,
    PretrainDatasetsArgs,
    Qwen2Config,
    SFTDatasetsArgs,
)
from nanotron.data.dataloader import (
    dummy_infinite_data_generator,
    get_train_dataloader,
)
from nanotron.data.processing import (
    clm_process,
    get_datasets,
)
from nanotron.data.sft_processing import prepare_sft_dataset
from nanotron.helpers import (
    compute_remain_train_steps_of_a_data_stage_from_ckp,
    get_consumed_train_samples_of_a_data_stage_from_ckp,
)
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.sanity_checks import sanity_check_dataloader
from nanotron.trainer import DistributedTrainer
from nanotron.utils import main_rank_first
from torch.utils.data import DataLoader
from nanotron.trainer import DataStageMetadata
from collections import defaultdict
try:
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoTokenizer
    from transformers import __version__ as tf_version
except ImportError:
    hf_hub_version = None
    tf_version = None

logger = logging.get_logger(__name__)

# import lovely_tensors as lt

# lt.monkey_patch()


def get_dataloader_from_data_stage(
    trainer: DistributedTrainer,
    data: DataArgs,
    consumed_train_samples_stage: int,
    consumed_tokens_per_dataset_folder: Dict[str, int],
    last_stages_consumed_tokens_per_dataset_folder: Dict[str, int],
    num_remaining_train_steps: int,
    sanity_check_dataloader_interval: Optional[int] = None,
):
    """
    Returns a dataloader for a given data stage.

    data: The data configuration for the current stage.
    consumed_train_samples_stage: The number of samples consumed by the model in the this stage (each stage starts from zero).
    consumed_tokens_per_dataset_folder: The number of tokens consumed by the model in previous stages to avoid reseeing them, because the sampler has restarted for this stage.
    num_remaining_train_steps: The number of remaining training steps for this stage.
    """
    assert consumed_train_samples_stage >= 0, "consumed_train_samples_stage should be greater than 0"
    assert num_remaining_train_steps >= 0, "num_remaining_train_steps should be greater than 0"

    # First, we need to know which ranks to feed the dataloader to
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    # Case 1: Dummy data generator
    if data.dataset is None:
        log_rank("Using dummy data generator", logger=logger, level=logging.INFO, rank=0)
        dataloader = dummy_infinite_data_generator(
            micro_batch_size=trainer.micro_batch_size,
            sequence_length=trainer.sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            vocab_size=trainer.model_config.vocab_size,
            seed=data.seed,
            parallel_context=trainer.parallel_context,
            use_position_ids=isinstance(
                trainer.model_config, Qwen2Config
            ),  # Simulate packed sequences to test SFT or inference
            cp_pg=trainer.parallel_context.cp_pg,
        )()

    # Case 2: HuggingFace datasets
    elif isinstance(data.dataset, PretrainDatasetsArgs) or isinstance(data.dataset, SFTDatasetsArgs):
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
                hf_dataset_or_datasets=data.dataset.hf_dataset_or_datasets,
                hf_dataset_config_name=data.dataset.hf_dataset_config_name,
                splits=data.dataset.hf_dataset_splits,
            )["train"]

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer.padding_side = "left"
            sequence_sep_tokens = [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token, tokenizer.unk_token]
            # assert bos or eos are present
            assert (
                tokenizer.bos_token is not None or tokenizer.eos_token is not None
            ), f"Tokenizer must have either bos or eos token, but found none for {tokenizer_path}"

            # Check that tokenizer's vocab size is smaller than the model's vocab size
            assert (
                tokenizer.vocab_size <= trainer.model_config.vocab_size
            ), f"Tokenizer's vocab size ({tokenizer.vocab_size}) is larger than the model's vocab size ({trainer.model_config.vocab_size})"

            # Different processing for SFT vs pretraining
            if isinstance(data.dataset, SFTDatasetsArgs):
                # For SFT, use the dedicated prepare_sft_dataset function
                # Get optional debug parameter to limit dataset size (for faster development)
                debug_max_samples = getattr(data.dataset, "debug_max_samples", None)

                # Process the dataset using our dedicated SFT processing module
                train_dataset = prepare_sft_dataset(
                    raw_dataset=raw_dataset,
                    tokenizer=tokenizer,
                    trainer_sequence_length=trainer.sequence_length,
                    debug_max_samples=debug_max_samples,
                    num_proc=data.dataset.dataset_processing_num_proc_per_process,
                )
            else:
                # For pretraining, use existing CLM processing
                train_dataset = clm_process(
                    raw_dataset=raw_dataset,
                    tokenizer=tokenizer,
                    text_column_name=data.dataset.text_column_name,
                    dataset_processing_num_proc_per_process=data.dataset.dataset_processing_num_proc_per_process,
                    dataset_overwrite_cache=data.dataset.dataset_overwrite_cache,
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
                consumed_train_samples=consumed_train_samples_stage,
                dataloader_num_workers=data.num_loading_workers,
                seed_worker=data.seed,
                dataloader_drop_last=True,
                use_position_ids=isinstance(trainer.model_config, Qwen2Config),
                sequence_sep_tokens=sequence_sep_tokens,  # Used to generate position ids
            )

            # Check if we have enough samples for train_steps
            total_tokens_dataset = len(dataloader.dataset) * trainer.sequence_length
            num_tokens_needed_for_training = (
                num_remaining_train_steps * trainer.global_batch_size * trainer.sequence_length
            )
            assert num_tokens_needed_for_training <= total_tokens_dataset, (
                f"Dataset is too small for steps ({total_tokens_dataset} < {num_tokens_needed_for_training}), "
                f"Try train_steps<={len(dataloader.dataset) // trainer.global_batch_size + trainer.iteration_step}"
            )

    # Case 3: Nanosets
    elif isinstance(data.dataset, NanosetDatasetsArgs):
        log_rank("Using TokenizedBytes Dataloader", logger=logger, level=logging.INFO, rank=0)
        from nanotron.data.tokenized_bytes import get_tb_dataloader, get_tb_datasets

        tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        assert (
            len(tokenizer) == trainer.model_config.vocab_size
        ), f"Tokenizer vocab size ({len(tokenizer)}) does not match model config vocab size ({trainer.model_config.vocab_size}). "
        log_rank(
            f"[TokenizedBytes] Creating TokenizedBytes with {len(data.dataset.dataset_folder)} dataset folders and {trainer.config.tokens.train_steps * trainer.global_batch_size} train samples",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        start_time = time.time()
        train_dataset, data_log = get_tb_datasets(
            config=data.dataset,
            global_batch_size=trainer.global_batch_size,
            sequence_length=trainer.sequence_length,
            train_steps=trainer.config.tokens.train_steps,
            current_iteration=trainer.iteration_step,
            parallel_context=trainer.parallel_context,
            shuffle=data.dataset.shuffle_files,
            eos_token_id=tokenizer.eos_token_id,
            seed=data.seed,
            consumed_samples=consumed_train_samples_stage,
            consumed_tokens_per_dataset_folder=consumed_tokens_per_dataset_folder,
            last_stages_consumed_tokens_per_dataset_folder=last_stages_consumed_tokens_per_dataset_folder,
        )
        dataloader = get_tb_dataloader(
            dataset=train_dataset,
            sequence_length=trainer.sequence_length,
            micro_batch_size=trainer.micro_batch_size,
            global_batch_size=trainer.global_batch_size,
            num_workers=data.num_loading_workers,
            cfg=data.dataset,
            consumed_samples=consumed_train_samples_stage,
            num_samples=trainer.config.tokens.train_steps * trainer.global_batch_size, # TODO: this overshoots what's needed by the current stage, but it doesnt matter?
            parallel_context=trainer.parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            dataloader_drop_last=True,
            dataloader_pin_memory=True,
            use_position_ids=isinstance(trainer.model_config, Qwen2Config),
            use_doc_masking=getattr(trainer.model_config, "_use_doc_masking", None),
        )
        log_rank(
            f"[TokenizedBytes] Time taken to create TokenizedBytes: {time.strftime('%M:%S', time.gmtime(time.time() - start_time))} (MM:SS)",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        dist.barrier()

        # Create Nanoset
        # from nanotron.data.nanoset import Nanoset

        # with main_rank_first(trainer.parallel_context.world_pg):
        #     tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
        #     tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        #     eos_token_id = tokenizer.eos_token_id
        #     assert (
        #         eos_token_id is not None or data.dataset.return_positions is False
        #     ), "Tokenizer must have an eos token if return_positions is True"
        #     log_rank(
        #         f"[Nanoset] Creating Nanoset with {len(data.dataset.dataset_folder)} dataset folders and {trainer.config.tokens.train_steps * trainer.global_batch_size} train samples",
        #         logger=logger,
        #         level=logging.INFO,
        #         rank=0,
        #     )
        #     start_time = time.time()
        #     train_dataset = Nanoset(
        #         dataset_folders=data.dataset.dataset_folder,
        #         sequence_length=trainer.sequence_length,
        #         token_size=data.dataset.token_size_in_bytes,
        #         train_split_num_samples=trainer.config.tokens.train_steps * trainer.global_batch_size,
        #         dataset_weights=data.dataset.dataset_weights,
        #         random_seed=data.seed,
        #         return_positions=data.dataset.return_positions,
        #         eos_token_id=eos_token_id,
        #     )
        #     end_time = time.time()
        #     log_rank(
        #         f"[Nanoset] Time taken to create Nanoset: {time.strftime('%M:%S', time.gmtime(end_time - start_time))} (MM:SS)",
        #         logger=logger,
        #         level=logging.INFO,
        #         rank=0,
        #     )
        # # Prepare dataloader
        # train_dataloader = build_nanoset_dataloader(
        #     train_dataset,
        #     trainer.sequence_length,
        #     parallel_context=trainer.parallel_context,
        #     input_pp_rank=input_pp_rank,
        #     output_pp_rank=output_pp_rank,
        #     micro_batch_size=trainer.micro_batch_size,
        #     consumed_train_samples=consumed_train_samples,
        #     dataloader_num_workers=data.num_loading_workers,
        #     dataloader_drop_last=True,
        #     use_position_ids=isinstance(trainer.model_config, Qwen2Config),
        #     use_doc_masking=False,
        #     dataloader_pin_memory=True,
        # )
        # dist.barrier()

    else:
        raise ValueError(f"Unhandled case of `self.config.data.dataset`. Got: {data.dataset}")

    if sanity_check_dataloader_interval is not None:
        sanity_check_dataloader(
            dataloader,
            tokenizer_path=trainer.config.tokenizer.tokenizer_name_or_path,
            sanity_check_dataloader_interval=sanity_check_dataloader_interval,
        )

    return dataloader


def get_dataloader(
    trainer: DistributedTrainer, sanity_check_dataloader_interval: Optional[int] = None
) -> Dict[str, DataLoader]:
    dataloaders = {}

    # Print training plan
    log_rank("Training plan", logger=logger, level=logging.INFO, rank=0, is_separator=True)
    stages_info = "".join(
        f"[Stage {stage.name}] start from step {stage.start_training_step} \n" for stage in trainer.config.data_stages
    )
    full_log_message = f"There are {len(trainer.config.data_stages)} training stages \n{stages_info}"
    log_rank(full_log_message, logger=logger, level=logging.INFO, rank=0)

    current_stage = None
    # WARNING: we assume we train on last stage
    stage_idx = len(trainer.config.data_stages) - 1
    stage_args = trainer.config.data_stages[stage_idx]
    if trainer.iteration_step+1 == stage_args.start_training_step:
        log_rank(f"Starting new stage {stage_args.name}", logger=logger, level=logging.INFO, rank=0)
        # we start a new stage
        if stage_idx >= len(trainer.metadata.data_stages):
            trainer.metadata.data_stages.append(DataStageMetadata(
                name=stage_args.name,
                start_training_step=stage_args.start_training_step,
                consumed_train_samples=0,
                consumed_tokens_per_dataset_folder={},
                sequence_length=trainer.sequence_length,
            ))
    elif len(trainer.metadata.data_stages) < len(trainer.config.data_stages):
        raise ValueError(f"If you're trying to start a new stage, you need to set `start_training_step` to the step after the last stage's: {trainer.iteration_step+1}")
    current_stage = trainer.metadata.data_stages[stage_idx]
    cur_stage_consumed_train_samples = current_stage.consumed_train_samples
    consumed_tokens_per_dataset_folder = current_stage.consumed_tokens_per_dataset_folder
    stage_args_data = trainer.config.data_stages[stage_idx].data

    num_remaining_train_steps = compute_remain_train_steps_of_a_data_stage_from_ckp(
        current_stage, trainer.config, trainer.metadata
    ) # TODO: check this
    log_rank(
        f"Current stage: {current_stage.name} has {num_remaining_train_steps} remaining training steps and has consumed {cur_stage_consumed_train_samples} samples"
        f"Consumed tokens per dataset folder: {pformat(consumed_tokens_per_dataset_folder)}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    # warn that if seqlen of stage - 1 has changed, consumed_train_samples=0 so we'll assume we're reading from new folder (so that we can resume training)
    if current_stage.sequence_length != trainer.metadata.data_stages[-1].sequence_length:
        raise NotImplementedError("We don't support changing sequence length between stages yet")
        if current_stage.consumed_train_samples == 0:
            log_rank(
                f"Warning: The sequence length of the last stage has changed from {trainer.metadata.data_stages[-1].sequence_length} to {current_stage.sequence_length}. We'll assume we're reading from the beginning of the dataset folders.",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
        else:
            # we're resuming training, so that's fine
            pass
        cur_stage_consumed_train_samples = current_stage.consumed_train_samples

    else:
        # Prepare last_stages_consumed_tokens_per_dataset_folder which will be used to offset BlendableDataset to avoid reseeing consumed tokens even when sampler has restarted for this stage
        last_stages_consumed_tokens_per_dataset_folder = {}
        for stage in trainer.metadata.data_stages[:-1]:
            for folder_path, consumed_tokens in stage.consumed_tokens_per_dataset_folder.items():
                last_stages_consumed_tokens_per_dataset_folder[folder_path] = last_stages_consumed_tokens_per_dataset_folder.get(folder_path, 0) + consumed_tokens  



    dataloaders[current_stage.name] = get_dataloader_from_data_stage(
        trainer,
        stage_args_data,
        consumed_train_samples_stage=cur_stage_consumed_train_samples,
        consumed_tokens_per_dataset_folder=consumed_tokens_per_dataset_folder,
        last_stages_consumed_tokens_per_dataset_folder=last_stages_consumed_tokens_per_dataset_folder,
        num_remaining_train_steps=num_remaining_train_steps,
        sanity_check_dataloader_interval=sanity_check_dataloader_interval,
    )
    return dataloaders


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
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
    trainer.train(dataloader)
