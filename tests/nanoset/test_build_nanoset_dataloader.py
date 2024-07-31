import sys
from math import isclose
from pathlib import Path
from typing import List

package_path = Path(__file__).parent.parent
sys.path.append(str(package_path))

import numpy as np
import pytest
from helpers.context import TestContext
from helpers.data import (
    assert_batch_dataloader,
    assert_nanoset_sync_across_all_ranks,
    compute_batch_hash,
    create_dataset_paths,
    create_dummy_json_dataset,
    preprocess_dummy_dataset,
)
from helpers.utils import available_gpus, get_all_3d_configurations, init_distributed, rerun_if_address_is_in_use
from nanotron.data.dataloader_builder import build_nanoset_dataloader
from nanotron.data.nanoset import Nanoset
from nanotron.data.utils import count_dataset_indexes, normalize
from nanotron.parallel import ParallelContext
from nanotron.utils import main_rank_first
from transformers import AutoTokenizer


@pytest.mark.parametrize(
    "tp,dp,pp",
    [
        pytest.param(*all_3d_configs)
        for gpus in range(1, min(available_gpus(), 4) + 1)
        for all_3d_configs in get_all_3d_configurations(gpus)
    ],
)
@pytest.mark.parametrize("train_steps", [500, 10000])
@pytest.mark.parametrize("sequence_length", [512, 8192])
@pytest.mark.parametrize("tokenizer_name_or_path", ["openai-community/gpt2", "unsloth/llama-3-8b-bnb-4bit"])
@rerun_if_address_is_in_use()
def test_build_nanoset_dataloader(
    tp: int, dp: int, pp: int, train_steps: int, sequence_length: int, tokenizer_name_or_path: str
):
    test_context = TestContext()

    # Create dataset folders
    json_paths, datatrove_tokenized_dataset_folders = create_dataset_paths(
        tmp_dir=test_context.get_auto_remove_tmp_dir(), quantity=2
    )

    # Create dummy json datasets
    for idx, json_path in enumerate(json_paths):
        create_dummy_json_dataset(path_to_json=json_path, dummy_text=f"Nanoset {idx}!", n_samples=(idx + 1) * 50000)

    # Preprocess json dataset with datatrove
    for json_path, datatrove_tokenized_dataset_folder in zip(json_paths, datatrove_tokenized_dataset_folders):
        preprocess_dummy_dataset(json_path, datatrove_tokenized_dataset_folder, tokenizer_name_or_path)

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_build_nanoset_dataloader)(
        datatrove_tokenized_dataset_folders=datatrove_tokenized_dataset_folders,
        train_steps=train_steps,
        sequence_length=sequence_length,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )


def _test_build_nanoset_dataloader(
    parallel_context: ParallelContext,
    datatrove_tokenized_dataset_folders: List[str],
    train_steps: int,
    sequence_length: int,
    tokenizer_name_or_path: str,
):
    SEED = 1234
    MICRO_BATCH_SIZE = 4
    N_MICRO_BATCHES_PER_BATCH = 8
    GLOBAL_BATCH_SIZE = MICRO_BATCH_SIZE * N_MICRO_BATCHES_PER_BATCH * parallel_context.dp_pg.size()

    input_pp_rank, output_pp_rank = 0, int(parallel_context.pp_pg.size() - 1)

    # Get tokenizer cardinality
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    token_size = 4 if len(tokenizer) > np.iinfo(np.uint16).max + 1 else 2
    del tokenizer

    # Create Nanoset configs: 1. Normal 2. Blended 3. Blended with weights
    nanoset_config = {
        "dataset_folders": [datatrove_tokenized_dataset_folders[0]],
        "dataset_weights": [1],
        "sequence_length": sequence_length,
        "token_size": token_size,
        "train_split_num_samples": train_steps * GLOBAL_BATCH_SIZE,
        "random_seed": SEED,
    }

    blended_nanoset_config = {
        "dataset_folders": datatrove_tokenized_dataset_folders,
        "dataset_weights": None,
        "sequence_length": sequence_length,
        "token_size": token_size,
        "train_split_num_samples": train_steps * GLOBAL_BATCH_SIZE,
        "random_seed": SEED,
    }

    blended_weighted_nanoset_config = {
        "dataset_folders": datatrove_tokenized_dataset_folders,
        "dataset_weights": [8, 2],
        "sequence_length": sequence_length,
        "token_size": token_size,
        "train_split_num_samples": train_steps * GLOBAL_BATCH_SIZE,
        "random_seed": SEED,
    }

    configs = [nanoset_config, blended_nanoset_config, blended_weighted_nanoset_config]

    for config in configs:
        # Create Nanoset
        with main_rank_first(parallel_context.world_pg):
            train_dataset = Nanoset(**config)

        # Assert we have the same Nanoset in all ranks
        assert_nanoset_sync_across_all_ranks(train_dataset, parallel_context)
        dataset_sample_count = count_dataset_indexes(train_dataset.dataset_index, len(train_dataset.dataset_folders))
        for idx, ds_length in enumerate(train_dataset.dataset_lengths):
            # Assert Nanoset doesn't sample indexes greater than the datasets
            assert (
                np.max(train_dataset.dataset_sample_index, where=train_dataset.dataset_index == idx, initial=-1)
                < ds_length
            ), f"Error building Nanoset Indexes: Tryng to access sample {np.max(train_dataset.dataset_sample_index, where=train_dataset.dataset_index==idx, initial = -1)} of a {ds_length} sample dataset"
            # Assert Nanoset builds up the correct blend WRT the dataset_weights
            assert isclose(
                normalize(dataset_sample_count).tolist()[idx], train_dataset.dataset_weights[idx], abs_tol=0.05
            ), f"Requested Nanoset to contain {round(train_dataset.dataset_weights[idx]*100, 2)}% of samples from {train_dataset.dataset_folders[idx]} but got {round(normalize(dataset_sample_count).tolist()[idx]*100, 2)}%"
        # Create Dataloaders
        dataloader = build_nanoset_dataloader(
            train_dataset,
            sequence_length=sequence_length,
            parallel_context=parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=MICRO_BATCH_SIZE,
            dataloader_num_workers=0,
            dataloader_drop_last=True,
        )

        # Check a batch produced by the Dataloader
        batch = next(iter(dataloader))
        assert_batch_dataloader(
            batch=batch,
            parallel_context=parallel_context,
            micro_batch_size=MICRO_BATCH_SIZE,
            sequence_length=sequence_length,
        )

    parallel_context.destroy()


@pytest.mark.parametrize(
    "tp,dp,pp",
    [
        pytest.param(*all_3d_configs)
        for gpus in range(1, min(available_gpus(), 4) + 1)
        for all_3d_configs in get_all_3d_configurations(gpus)
    ],
)
@pytest.mark.parametrize("skipped_batches", [20, 5555])
@pytest.mark.parametrize("tokenizer_name_or_path", ["openai-community/gpt2", "unsloth/llama-3-8b-bnb-4bit"])
@rerun_if_address_is_in_use()
def test_recover_nanoset_dataloader(tp: int, dp: int, pp: int, skipped_batches: int, tokenizer_name_or_path: str):
    test_context = TestContext()

    # Create dataset folders
    json_paths, datatrove_tokenized_dataset_folders = create_dataset_paths(
        tmp_dir=test_context.get_auto_remove_tmp_dir(), quantity=2
    )

    # Create dummy json datasets
    for idx, json_path in enumerate(json_paths):
        create_dummy_json_dataset(path_to_json=json_path, dummy_text=f"Nanoset {idx}!", n_samples=(idx + 1) * 50000)

    # Preprocess json dataset with datatrove
    for json_path, datatrove_tokenized_dataset_folder in zip(json_paths, datatrove_tokenized_dataset_folders):
        preprocess_dummy_dataset(json_path, datatrove_tokenized_dataset_folder, tokenizer_name_or_path)

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_recover_nanoset_dataloader)(
        datatrove_tokenized_dataset_folders=datatrove_tokenized_dataset_folders,
        skipped_batches=skipped_batches,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )


def _test_recover_nanoset_dataloader(
    parallel_context: ParallelContext,
    datatrove_tokenized_dataset_folders: List[str],
    skipped_batches: int,
    tokenizer_name_or_path: str,
):
    SEED = 1234
    MICRO_BATCH_SIZE = 4
    N_MICRO_BATCHES_PER_BATCH = 8
    GLOBAL_BATCH_SIZE = MICRO_BATCH_SIZE * N_MICRO_BATCHES_PER_BATCH * parallel_context.dp_pg.size()
    SEQUENCE_LENGTH = 1024
    TRAIN_STEPS = 10000

    input_pp_rank, output_pp_rank = 0, int(parallel_context.pp_pg.size() - 1)

    # Get tokenizer cardinality
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    token_size = 4 if len(tokenizer) > np.iinfo(np.uint16).max + 1 else 2
    del tokenizer

    # Create Nanoset configs: 1. Normal 2. Blended 3. Blended with weights
    nanoset_config = {
        "dataset_folders": [datatrove_tokenized_dataset_folders[0]],
        "dataset_weights": [1],
        "sequence_length": SEQUENCE_LENGTH,
        "token_size": token_size,
        "train_split_num_samples": TRAIN_STEPS * GLOBAL_BATCH_SIZE,
        "random_seed": SEED,
    }

    blended_nanoset_config = {
        "dataset_folders": datatrove_tokenized_dataset_folders,
        "dataset_weights": None,
        "sequence_length": SEQUENCE_LENGTH,
        "token_size": token_size,
        "train_split_num_samples": TRAIN_STEPS * GLOBAL_BATCH_SIZE,
        "random_seed": SEED,
    }

    blended_weighted_nanoset_config = {
        "dataset_folders": datatrove_tokenized_dataset_folders,
        "dataset_weights": [8, 2],
        "sequence_length": SEQUENCE_LENGTH,
        "token_size": token_size,
        "train_split_num_samples": TRAIN_STEPS * GLOBAL_BATCH_SIZE,
        "random_seed": SEED,
    }

    configs = [nanoset_config, blended_nanoset_config, blended_weighted_nanoset_config]

    for config in configs:
        # Create Nanoset
        with main_rank_first(parallel_context.world_pg):
            train_dataset = Nanoset(**config)

        # Create initial Dataloader
        dataloader = build_nanoset_dataloader(
            train_dataset,
            sequence_length=SEQUENCE_LENGTH,
            parallel_context=parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=MICRO_BATCH_SIZE,
            dataloader_num_workers=0,
            dataloader_drop_last=True,
        )

        # Recover from failures
        dataloader = iter(dataloader)
        for _ in range(skipped_batches + 1):  # In order to compare with the first batch of the recovered DataLoader
            batch = next(dataloader)

        # Create recover Dataloader
        recovered_dataloader = build_nanoset_dataloader(
            train_dataset,
            sequence_length=SEQUENCE_LENGTH,
            parallel_context=parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=MICRO_BATCH_SIZE,
            dataloader_num_workers=0,
            dataloader_drop_last=True,
            # NOTE The dataloader serves batches of micro_batch_size despite of batch_accumulation_per_replica
            consumed_train_samples=skipped_batches * MICRO_BATCH_SIZE * parallel_context.dp_pg.size(),
        )

        recovered_first_batch = next(iter(recovered_dataloader))

        assert compute_batch_hash(batch) == compute_batch_hash(recovered_first_batch)

    parallel_context.destroy()
