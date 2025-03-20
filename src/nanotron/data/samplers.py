from typing import Optional, Union

import datasets
import torch
import torch.utils.data
from torch.utils.data import BatchSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.trainer_pt_utils import DistributedSamplerWithLoop


class SkipBatchSampler(BatchSampler):
    """
    A `torch.utils.data.BatchSampler` that skips the first `n` batches of another `torch.utils.data.BatchSampler`.

    In case of distributed training, we skip batches on each rank, so a total of `skip_batches * dp_size`
    batches are skipped globally.

    Args:
        batch_sampler: Base batch sampler
        skip_batches: Number of batches to skip
        dp_size: Data parallel world size
    """

    def __init__(self, batch_sampler: BatchSampler, skip_batches: int, dp_size: int):
        self.batch_sampler = batch_sampler
        # In case of DDP, we skip batches on each rank, so a total of `skip_batches * dp_size` batches
        self.skip_batches = skip_batches // dp_size

    def __iter__(self):
        for index, samples in enumerate(self.batch_sampler):
            if index >= self.skip_batches:
                yield samples

    @property
    def total_length(self):
        return len(self.batch_sampler)

    def __len__(self):
        return len(self.batch_sampler) - self.skip_batches


def get_sampler(
    dl_ranks_size: int,
    dl_rank: int,
    train_dataset: Union["torch.utils.data.Dataset", "datasets.Dataset"],
    consumed_train_samples: int,
    seed: int = 42,
    use_loop_to_round_batch_size: bool = False,
    micro_batch_size: Optional[int] = None,
    drop_last: Optional[bool] = True,
    shuffle: bool = True,
) -> Optional[torch.utils.data.Sampler]:
    """
    Returns sampler that restricts data loading to a subset of the dataset proper to the DP rank.

    Args:
        dl_ranks_size: Number of data parallel ranks
        dl_rank: Current data parallel rank
        train_dataset: Dataset to sample from
        consumed_train_samples: Number of samples already consumed
        seed: Random seed
        use_loop_to_round_batch_size: Whether to loop at the end back to beginning to ensure
                                     each process has a batch size multiple of micro_batch_size
        micro_batch_size: Batch size for each iteration
        drop_last: Whether to drop the last incomplete batch
        shuffle: Whether to shuffle the dataset

    Returns:
        Sampler for distributed training with appropriate skip behavior
    """
    if use_loop_to_round_batch_size:
        assert micro_batch_size is not None
        # loops at the end back to the beginning of the shuffled samples to make each process have a round multiple of batch_size samples.
        sampler = DistributedSamplerWithLoop(
            train_dataset,
            batch_size=micro_batch_size,
            num_replicas=dl_ranks_size,
            rank=dl_rank,
            seed=seed,
            drop_last=drop_last,
        )
    else:
        sampler = DistributedSampler(
            train_dataset, num_replicas=dl_ranks_size, rank=dl_rank, seed=seed, drop_last=drop_last, shuffle=shuffle
        )

    if consumed_train_samples > 0:
        sampler = SkipBatchSampler(sampler, skip_batches=consumed_train_samples, dp_size=dl_ranks_size)

    return sampler


class EmptyInfiniteDataset:
    """
    Hack as removing all columns from a datasets.Dataset makes the number of rows 0.
    This provides a dataset that returns empty dicts but maintains the original length.

    Args:
        length: Number of items in the dataset
    """

    def __init__(self, length: int):
        self._length = length

    def __getitem__(self, item) -> dict:
        if isinstance(item, int):
            return {}
        raise NotImplementedError(f"{item} of type {type(item)} is not supported yet")

    def __len__(self) -> int:
        return self._length
