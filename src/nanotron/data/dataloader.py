from typing import Dict, Iterator, Union

import datasets
import torch
from torch.utils.data import DataLoader

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import Config
from nanotron.data.clm_collator import DataCollatorForCLM, DataCollatorForCLMWithPositionIds
from nanotron.data.samplers import EmptyInfiniteDataset, get_sampler
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.random import set_random_seed
from nanotron.sanity_checks import (
    assert_fail_except_rank_with,
    assert_tensor_synced_across_pg,
)

logger = logging.get_logger(__name__)


def sanity_check_dataloader(
    dataloader: Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]],
    parallel_context: ParallelContext,
    config: Config,
) -> Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]:
    """
    Run sanity checks on batches from the dataloader to ensure data is properly distributed and synchronized.

    Args:
        dataloader: Iterator that yields batches
        parallel_context: Object containing process groups information
        config: Configuration object

    Yields:
        The same batches after performing sanity checks
    """
    for batch in dataloader:
        micro_batch = {
            k: v if isinstance(v, TensorPointer) else v.to("cuda", memory_format=torch.contiguous_format)
            for k, v in batch.items()
        }

        if not config.general.ignore_sanity_checks:
            # SANITY CHECK: Check input are not the same across DP
            for key, value in sorted(micro_batch.items(), key=lambda x: x[0]):
                if isinstance(value, TensorPointer):
                    continue

                if "mask" in key:
                    # It's fine if mask is the same across DP
                    continue

                with assert_fail_except_rank_with(AssertionError, rank_exception=0, pg=parallel_context.dp_pg):
                    assert_tensor_synced_across_pg(
                        tensor=value, pg=parallel_context.dp_pg, msg=lambda err: f"{key} {err}"
                    )

            # SANITY CHECK: Check input are synchronized throughout TP
            for key, value in sorted(micro_batch.items(), key=lambda x: x[0]):
                if isinstance(value, TensorPointer):
                    continue
                assert_tensor_synced_across_pg(
                    tensor=value,
                    pg=parallel_context.tp_pg,
                    msg=lambda err: f"{key} are not synchronized throughout TP {err}",
                )

            # SANITY CHECK: Check that input are synchronized throughout PP
            # TODO @thomasw21: That's really hard to test as input gets sharded across the PP, let's assume it works for now.

            # SANITY CHECK: Check that an input only exists on the PP rank responsible for it
            # TODO @nouamanetazi: add this test
        yield micro_batch


def dummy_infinite_data_generator(
    micro_batch_size: int,
    sequence_length: int,
    input_pp_rank: int,
    output_pp_rank: int,
    vocab_size: int,
    seed: int,
    parallel_context: ParallelContext,
    use_position_ids: bool = False,
):
    """
    Generate dummy data for testing or benchmark purposes.

    Args:
        micro_batch_size: Size of each micro batch
        sequence_length: Maximum sequence length
        input_pp_rank: Rank responsible for input data
        output_pp_rank: Rank responsible for output/label data
        vocab_size: Size of the vocabulary
        seed: Random seed for reproducibility
        parallel_context: Object containing process groups information
        use_position_ids: Whether to use packed sequences

    Returns:
        Generator function that yields infinite random data batches
    """

    def data_generator() -> Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]:
        # Random generator
        generator = torch.Generator(device="cuda")
        # Make sure that TP are synced always
        generator.manual_seed(
            seed * (1 + dist.get_rank(parallel_context.dp_pg)) * (1 + dist.get_rank(parallel_context.pp_pg))
        )

        if use_position_ids:
            document_lengths = [[4, 6, 12], [sequence_length]]
            position_ids = torch.full(
                (micro_batch_size, sequence_length), fill_value=-1, dtype=torch.long, device="cuda"
            )
            for i in range(micro_batch_size):
                prev_idx = 0
                for doc_idx, doc_len in enumerate(document_lengths[i]):
                    position_ids[i, prev_idx : prev_idx + doc_len] = torch.arange(
                        0, doc_len, dtype=torch.long, device="cuda"
                    )
                    prev_idx += doc_len
            while True:
                yield {
                    "input_ids": torch.randint(
                        0,
                        vocab_size,
                        (micro_batch_size * sequence_length,),
                        dtype=torch.long,
                        device="cuda",
                        generator=generator,
                    )
                    if dist.get_rank(parallel_context.pp_pg) == input_pp_rank
                    else TensorPointer(group_rank=input_pp_rank),
                    "position_ids": position_ids
                    if dist.get_rank(parallel_context.pp_pg) == input_pp_rank
                    else TensorPointer(group_rank=input_pp_rank),
                    "label_ids": torch.randint(
                        0,
                        vocab_size,
                        (micro_batch_size, sequence_length),
                        dtype=torch.long,
                        device="cuda",
                        generator=generator,
                    )
                    if dist.get_rank(parallel_context.pp_pg) == output_pp_rank
                    else TensorPointer(group_rank=output_pp_rank),
                    "label_mask": torch.ones(
                        micro_batch_size,
                        sequence_length,
                        dtype=torch.bool,
                        device="cuda",
                    )
                    if dist.get_rank(parallel_context.pp_pg) == output_pp_rank
                    else TensorPointer(group_rank=output_pp_rank),
                }
        else:
            while True:
                yield {
                    "input_ids": torch.randint(
                        0,
                        vocab_size,
                        (micro_batch_size, sequence_length),
                        dtype=torch.long,
                        device="cuda",
                        generator=generator,
                    )
                    if dist.get_rank(parallel_context.pp_pg) == input_pp_rank
                    else TensorPointer(group_rank=input_pp_rank),
                    "input_mask": torch.ones(
                        micro_batch_size,
                        sequence_length,
                        dtype=torch.bool,
                        device="cuda",
                    )
                    if dist.get_rank(parallel_context.pp_pg) == input_pp_rank
                    else TensorPointer(group_rank=input_pp_rank),
                    "label_ids": torch.randint(
                        0,
                        vocab_size,
                        (micro_batch_size, sequence_length),
                        dtype=torch.long,
                        device="cuda",
                        generator=generator,
                    )
                    if dist.get_rank(parallel_context.pp_pg) == output_pp_rank
                    else TensorPointer(group_rank=output_pp_rank),
                    "label_mask": torch.ones(
                        micro_batch_size,
                        sequence_length,
                        dtype=torch.bool,
                        device="cuda",
                    )
                    if dist.get_rank(parallel_context.pp_pg) == output_pp_rank
                    else TensorPointer(group_rank=output_pp_rank),
                }

    return data_generator


def set_tensor_pointers(
    input_dict: Dict[str, Union[torch.Tensor, TensorPointer]], group: dist.ProcessGroup, group_rank: int
) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
    """
    Make sure only the group_rank rank has the data, others have TensorPointers.

    Args:
        input_dict: Dictionary of tensors or tensor pointers
        group: Process group
        group_rank: Rank in the group that should have the actual data

    Returns:
        Dictionary with tensors or pointers based on current rank
    """
    return {
        k: v if dist.get_rank(group) == group_rank else TensorPointer(group_rank=group_rank)
        for k, v in input_dict.items()
    }


def get_dataloader_worker_init(dp_rank: int):
    """
    Creates random states for each worker in order to get different state in each workers.

    Args:
        dp_rank: Data parallel rank

    Returns:
        Worker initialization function to be used by DataLoader
    """

    def dataloader_worker_init(worker_id):
        # Dataloader is TP/PP synced in random states
        seed = 2 ** (1 + worker_id) * 3 ** (1 + dp_rank) % (2**32)
        set_random_seed(seed)

    return dataloader_worker_init


def get_train_dataloader(
    train_dataset: "datasets.Dataset",
    sequence_length: int,
    parallel_context: ParallelContext,
    input_pp_rank: int,
    output_pp_rank: int,
    micro_batch_size: int,
    consumed_train_samples: int,
    dataloader_num_workers: int,
    seed_worker: int,
    dataloader_drop_last: bool = True,
    dataloader_pin_memory: bool = True,
    use_loop_to_round_batch_size: bool = False,
    use_position_ids: bool = False,
) -> DataLoader:
    """
    Get a PyTorch DataLoader for training.

    Args:
        train_dataset: Dataset to use for training
        sequence_length: Maximum sequence length
        parallel_context: Object containing process groups information
        input_pp_rank: Rank responsible for input data
        output_pp_rank: Rank responsible for output/label data
        micro_batch_size: Size of each micro batch
        consumed_train_samples: Number of samples already consumed
        dataloader_num_workers: Number of workers for the DataLoader
        seed_worker: Random seed for workers
        dataloader_drop_last: Whether to drop the last incomplete batch
        dataloader_pin_memory: Whether to use pinned memory for faster data transfer
        use_loop_to_round_batch_size: Whether to loop at the end of dataset to ensure batch size multiple
        use_position_ids: Whether to use position IDs in the collator

    Returns:
        PyTorch DataLoader for training
    """
    if not isinstance(train_dataset, datasets.Dataset):
        raise ValueError(f"training requires a datasets.Dataset, but got {type(train_dataset)}")

    # Case of ranks requiring data
    if dist.get_rank(parallel_context.pp_pg) in [
        input_pp_rank,
        output_pp_rank,
    ]:
        train_dataset = train_dataset.with_format(type="numpy", columns=["input_ids"], output_all_columns=True)

    # Case of ranks not requiring data. We give them an infinite dummy dataloader
    else:
        #
        assert train_dataset.column_names == ["input_ids"], (
            f"Dataset has to have a single column, with `input_ids` as the column name. "
            f"Current dataset: {train_dataset}"
        )
        dataset_length = len(train_dataset)
        train_dataset = train_dataset.remove_columns(column_names="input_ids")
        assert (
            len(train_dataset) == 0
        ), f"Dataset has to be empty after removing the `input_ids` column. Current dataset: {train_dataset}"
        # HACK as if we remove the last column of a train_dataset, it becomes empty and it's number of rows becomes empty.
        train_dataset = EmptyInfiniteDataset(length=dataset_length)
        # No need to spawn a lot of workers, we can just use main
        dataloader_num_workers = 0

    if use_position_ids:
        data_collator = DataCollatorForCLMWithPositionIds(
            sequence_length=sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            parallel_context=parallel_context,
        )
    else:
        data_collator = DataCollatorForCLM(
            sequence_length=sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            parallel_context=parallel_context,
        )

    # Compute size and rank of dataloader workers
    dp_ranks_size = parallel_context.dp_pg.size()
    dp_rank = parallel_context.dp_pg.rank()

    train_sampler = get_sampler(
        dl_rank=dp_rank,
        dl_ranks_size=dp_ranks_size,
        train_dataset=train_dataset,
        seed=seed_worker,
        use_loop_to_round_batch_size=use_loop_to_round_batch_size,
        micro_batch_size=micro_batch_size,
        drop_last=dataloader_drop_last,
        consumed_train_samples=consumed_train_samples,
    )

    return DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        drop_last=dataloader_drop_last,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dp_rank),
    )
