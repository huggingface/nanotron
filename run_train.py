"""
Nanotron training script.

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
python -u -m torch.distributed.run --nproc_per_node=2 run_train.py --config-file examples/tinyllama.yaml
```
"""
import argparse
from typing import Dict, cast

import numpy as np
import os
from nanotron import logging
from nanotron.config import DataArgs, DatasetStageArgs, NanosetDatasetsArgs, PretrainDatasetsArgs, MixteraDatasetArgs
from nanotron.data.dataloader_builder import build_nanoset_dataloader
from nanotron.dataloader import (
    clm_process,
    dummy_infinite_data_generator,
    get_datasets,
    get_train_dataloader,
)
from nanotron.helpers import (
    compute_remain_train_steps_of_a_data_stage_from_ckp,
    get_consumed_train_samples_of_a_data_stage_from_ckp,
)
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from nanotron.utils import main_rank_first, has_length
from torch.utils.data import DataLoader

try:
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoTokenizer
    from transformers import __version__ as tf_version
except ImportError:
    hf_hub_version = None
    tf_version = None

logger = logging.get_logger(__name__)

# Query execution in Mixtera takes long, and NCCL would time out otherwise.
os.environ["NCCL_TIMEOUT"] = str(30 * 60 * 1000)

def get_dataloader_from_data_stage(
    trainer: DistributedTrainer,
    data: DataArgs,
    consumed_train_samples: int,
    num_remaining_train_steps: int,
):
    """
    Returns a dataloader for a given data stage.

    data: The data configuration for the current stage.
    consumed_train_samples: The number of samples consumed by the model in the this stage (each stage starts from zero).
    num_remaining_train_steps: The number of remaining training steps for this stage.
    """
    assert consumed_train_samples >= 0, "consumed_train_samples should be greater than 0"
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
        )()

    # Case 2: HuggingFace datasets
    elif isinstance(data.dataset, PretrainDatasetsArgs):
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
                streaming_hf = data.dataset.use_streaming_interface
            )["train"]

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            # Check that tokenizer's vocab size is smaller than the model's vocab size
            assert (
                tokenizer.vocab_size <= trainer.model_config.vocab_size
            ), f"Tokenizer's vocab size ({tokenizer.vocab_size}) is larger than the model's vocab size ({trainer.model_config.vocab_size})"

            # We apply the Causal Language Modeling preprocessing
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
                consumed_train_samples=consumed_train_samples,
                dataloader_num_workers=data.num_loading_workers,
                seed_worker=data.seed,
                dataloader_drop_last=True,
            )

            # Check if we have enough samples for train_steps
            if has_length(train_dataset):
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
        # Get tokenizer cardinality
        tokenizer = AutoTokenizer.from_pretrained(trainer.config.tokenizer.tokenizer_name_or_path)
        token_size = 4 if len(tokenizer) > np.iinfo(np.uint16).max + 1 else 2
        del tokenizer
        # Create Nanoset
        from nanotron.data.nanoset import Nanoset

        with main_rank_first(trainer.parallel_context.world_pg):
            train_dataset = Nanoset(
                dataset_folders=data.dataset.dataset_folder,
                dataset_weights=data.dataset.dataset_weights,
                sequence_length=trainer.sequence_length,
                token_size=token_size,
                train_split_num_samples=trainer.config.tokens.train_steps * trainer.global_batch_size,
                random_seed=data.seed,
            )

        # Prepare dataloader
        train_dataloader = build_nanoset_dataloader(
            train_dataset,
            trainer.sequence_length,
            parallel_context=trainer.parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=trainer.micro_batch_size,
            consumed_train_samples=consumed_train_samples,
            dataloader_num_workers=data.num_loading_workers,
            dataloader_drop_last=True,
        )

        return train_dataloader
    
    # Case 4: Mixtera
    elif isinstance(data.dataset, MixteraDatasetArgs):
        # Query execution in Mixtera takes long, and NCCL would time out otherwise.
        os.environ["NCCL_TIMEOUT"] = str(30 * 60 * 1000)
        tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
        log_rank(
            f"Loading tokenizer from {tokenizer_path} and transformers/hf_hub versions {tf_version, hf_hub_version}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        
        from mixtera.hf import MixteraHFDataset
        from mixtera.core.client import MixteraClient, QueryExecutionArgs, ResultStreamingArgs
        from mixtera.core.query import Query
        from mixtera.core.query.mixture import InferringMixture

        if data.dataset.port:
            client = MixteraClient.from_remote(data.dataset.path, data.dataset.port)
        else:
            client = MixteraClient.from_directory(data.dataset.path)
        
        job_id = data.dataset.job_id
        chunk_size = data.dataset.chunk_size
        tunnel_via_server = data.dataset.tunnel_via_server
        chunk_reading_degree_of_parallelism = data.dataset.chunk_reading_degree_of_parallelism
        chunk_reading_per_window_mixture = data.dataset.chunk_reading_per_window_mixture
        chunk_reading_window_size = data.dataset.chunk_reading_window_size

        total_nodes = trainer.parallel_context.world_pg.size()
        data_parallel_size = trainer.parallel_context.data_parallel_size
        assert data_parallel_size == trainer.parallel_context.dp_pg.size(), f"num_nodes_per_dp_group = {data_parallel_size} != trainer.parallel_context.dp_pg.size() = {trainer.parallel_context.dp_pg.size()}"
        assert total_nodes % data_parallel_size == 0, f"total_nodes = {total_nodes} is not a multiple of data_parallel_size = {data_parallel_size}"
        nodes_per_dp_group = total_nodes // data_parallel_size
        assert nodes_per_dp_group == trainer.parallel_context.mp_pg.size(), f"nodes_per_dp_group = {nodes_per_dp_group} != trainer.parallel_context.mp_pg.size() = {trainer.parallel_context.mp_pg.size()}"
        dp_group_id = trainer.parallel_context.dp_pg.rank()
        assert dp_group_id < data_parallel_size, f"dp_group_id = {dp_group_id} NOT < data_parallel_size = {data_parallel_size}"
        node_id = trainer.parallel_context.mp_pg.rank()
        logger.info(f"There are {total_nodes} total nodes, {data_parallel_size} dp size => {nodes_per_dp_group} nodes per DP group. My dp group is {dp_group_id}, my node id is {node_id}")
        assert node_id < nodes_per_dp_group, f"node_id = {node_id} NOT < nodes_per_dp_group = {nodes_per_dp_group}"

        query_execution_args = QueryExecutionArgs(mixture=InferringMixture(chunk_size), dp_groups=data_parallel_size, nodes_per_group=nodes_per_dp_group, num_workers=data.num_loading_workers)
        streaming_args = ResultStreamingArgs(job_id=job_id, dp_group_id=dp_group_id, node_id=node_id, tunnel_via_server=tunnel_via_server, chunk_reading_degree_of_parallelism=chunk_reading_degree_of_parallelism, chunk_reading_per_window_mixture=chunk_reading_per_window_mixture, chunk_reading_window_size=chunk_reading_window_size)

        query = Query.for_job(job_id)
        if data.dataset.query is not None and data.dataset.query.strip() != "":
            query = query.select(tuple(data.dataset.query.split(" ")))
        else:
            query = query.select(None)

        raw_dataset = MixteraHFDataset(client, query, query_execution_args, streaming_args, checkpoint_path=trainer.config.checkpoints.resume_checkpoint_path)

        # The following is mostly copy&pasted from the huggingface case above, since `MixteraHFDataset` is a datasets.IterableDataset
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # Check that tokenizer's vocab size is smaller than the model's vocab size
        assert (
            tokenizer.vocab_size <= trainer.model_config.vocab_size
        ), f"Tokenizer's vocab size ({tokenizer.vocab_size}) is larger than the model's vocab size ({trainer.model_config.vocab_size})"

        # We apply the Causal Language Modeling preprocessing
        train_dataset = clm_process(
            raw_dataset=raw_dataset,
            tokenizer=tokenizer,
            text_column_name="text", # by MixteraHFDataset implementation
            dataset_processing_num_proc_per_process=-1, # will be ignored
            dataset_overwrite_cache=False, # will be ignored
            sequence_length=trainer.sequence_length,
            batch_size=chunk_size // 4 # We set the batch size to 25% of chunk size. We don't want this to be higher than the chunk size because otherwise we will prefetch chunks implicitly!
        )

        # We load the processed dataset on the ranks requiring it
        dataloader = get_train_dataloader(
            train_dataset=train_dataset,
            sequence_length=trainer.sequence_length,
            parallel_context=trainer.parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=trainer.micro_batch_size,
            consumed_train_samples=consumed_train_samples, # TODO the whole restarting/checkpoint thing is not supported currently in Mixtera. Not sure what happens currently.
            dataloader_num_workers=data.num_loading_workers,
            seed_worker=data.seed,
            dataloader_drop_last=True,
        )

    else:
        raise ValueError(f"Unhandled case of `self.config.data.dataset`. Got: {data.dataset}")

    return dataloader


def get_dataloader(trainer: DistributedTrainer) -> Dict[str, DataLoader]:
    dataloaders = {}

    for stage_idx, stage in enumerate(trainer.config.data_stages):
        # NOTE: we only create the dataloader for the first stage,
        # then we lazy initialize the dataloader for the other stages
        stage = cast(DatasetStageArgs, stage)
        consumed_train_samples = get_consumed_train_samples_of_a_data_stage_from_ckp(stage, trainer.metadata)
        assert (
            consumed_train_samples is not None
        ), f"Cannot find consumed_train_samples for stage {stage.start_training_step} in the checkpoint"

        num_remaining_train_steps = compute_remain_train_steps_of_a_data_stage_from_ckp(
            stage, trainer.config, trainer.metadata
        )
        log_rank(
            f"[Training Plan] Stage {stage.name} has {num_remaining_train_steps} remaining training steps and has consumed {consumed_train_samples} samples",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        dataloader = (
            get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed_train_samples,
                num_remaining_train_steps=num_remaining_train_steps,
            )
            if stage_idx == 0
            else lambda stage=stage: get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed_train_samples,
                num_remaining_train_steps=num_remaining_train_steps,
            )
        )
        dataloaders[stage.name] = dataloader
    return dataloaders


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
