# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from dataclasses import dataclass
from typing import Optional, Union

import datasets
import torch
import torch.utils.data
from torch.utils.data import BatchSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.trainer_pt_utils import DistributedSamplerWithLoop

from nanotron import logging
from nanotron.logging import log_rank

logger = logging.get_logger(__name__)


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
    shuffle: bool = False,
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


@dataclass
class BaseMegatronSampler:
    total_samples: int
    consumed_samples: int
    micro_batch_size: int
    data_parallel_rank: int
    data_parallel_size: int
    global_batch_size: int
    drop_last: bool = True
    pad_samples_to_global_batch_size: Optional[bool] = False

    def __post_init__(self):
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * self.data_parallel_size

        # Sanity checks.
        if self.total_samples <= 0:
            raise RuntimeError("no sample to consume: {}".format(self.total_samples))
        if self.consumed_samples >= self.total_samples:
            raise RuntimeError("no samples left to consume: {}, {}".format(self.consumed_samples, self.total_samples))
        if self.micro_batch_size <= 0:
            raise RuntimeError(f"micro_batch_size size must be greater than 0, but {self.micro_batch_size}")
        if self.data_parallel_size <= 0:
            raise RuntimeError(f"data parallel size must be greater than 0, but {self.data_parallel_size}")
        if self.data_parallel_rank >= self.data_parallel_size:
            raise RuntimeError(
                "data_parallel_rank should be smaller than data size, but {} >= {}".format(
                    self.data_parallel_rank, self.data_parallel_size
                )
            )
        if self.global_batch_size % (self.micro_batch_size * self.data_parallel_size) != 0:
            raise RuntimeError(
                f"`global_batch_size` ({self.global_batch_size}) is not divisible by "
                f"`micro_batch_size ({self.micro_batch_size}) x data_parallel_size "
                f"({self.data_parallel_size})`"
            )
        if self.pad_samples_to_global_batch_size and self.global_batch_size is None:
            raise RuntimeError(
                "`pad_samples_to_global_batch_size` can be `True` only when "
                "`global_batch_size` is set to an integer value"
            )
        log_rank(
            f"Instantiating MegatronPretrainingSampler with total_samples: {self.total_samples} and consumed_samples: {self.consumed_samples}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

    @abc.abstractmethod
    def __iter__(self):
        ...


@dataclass
class MegatronPretrainingSampler(BaseMegatronSampler):
    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __len__(self):
        num_available_samples: int = self.total_samples - self.consumed_samples
        if self.global_batch_size is not None:
            if self.drop_last:
                return num_available_samples // self.global_batch_size
            else:
                return (num_available_samples + self.global_batch_size - 1) // self.global_batch_size
        else:
            if self.drop_last:
                return num_available_samples // self.micro_batch_times_data_parallel_size
            else:
                return (num_available_samples - 1) // self.micro_batch_times_data_parallel_size + 1

    def __iter__(self):
        batch = []
        batch_idx = 0
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                log_rank(
                    f"DP {self.data_parallel_rank} batch {batch_idx} {batch[start_idx:end_idx]} self.consumed_samples {self.consumed_samples}",
                    logger=logger,
                    level=logging.DEBUG,
                )
                # self.last_consumed_sample_all_ranks = batch[-1] # = self.consumed_samples?
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch[start_idx:end_idx]
                batch = []
                batch_idx += 1

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            if self.pad_samples_to_global_batch_size:
                for i in range(
                    self.data_parallel_rank, self.global_batch_size, self.micro_batch_times_data_parallel_size
                ):
                    indices = [batch[j] for j in range(i, max(len(batch), i + self.micro_batch_size))]
                    num_pad = self.micro_batch_size - len(indices)
                    indices = indices + [-1] * num_pad
                    yield indices
            else:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]


@dataclass
class MegatronPretrainingRandomSampler(BaseMegatronSampler):
    def __len__(self):
        num_available_samples: int = self.total_samples - self.consumed_samples
        if self.global_batch_size is not None:
            if self.drop_last:
                return num_available_samples // self.global_batch_size
            else:
                return (num_available_samples + self.global_batch_size - 1) // self.global_batch_size
        else:
            if self.drop_last:
                return num_available_samples // self.micro_batch_times_data_parallel_size
            else:
                return (num_available_samples - 1) // self.micro_batch_times_data_parallel_size + 1

    def __iter__(self):
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # data sharding and random sampling
        bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size

        g = torch.Generator()
        g.manual_seed(self.epoch)
        random_idx = torch.randperm(bucket_size, generator=g).tolist()
        idx_range = [start_idx + x for x in random_idx[bucket_offset:]]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            yield batch


class MegatronPretrainingCyclicSampler(BaseMegatronSampler):
    """Cyclic sampler

    This sampler is used for the cyclic pretraining. It will go through the dataset
    once and then start over again without shuffling.

    For data parallelism, the dataset is sharded into `data_parallel_size` chunks at the full dataset level.
    Each rank will then sample from its own shard starting from a different offset in the full dataset.

    Args:
        total_samples (int): total number of samples in the dataset
        consumed_samples (int): number of samples already consumed across all dataparallel ranks
        micro_batch_size (int): number of samples in a micro batch
        data_parallel_rank (int): rank of the data parallel group
        data_parallel_size (int): size of the data parallel group
        drop_last (bool): drop the last batch if it is not complete
        global_batch_size (int): global batch size
        pad_samples_to_global_batch_size (bool): pad the last batch to global batch size
    """

    def __init__(
        self,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        global_batch_size: int,
        drop_last: bool = True,
        pad_samples_to_global_batch_size: Optional[bool] = False,
    ) -> None:
        super().__init__(
            total_samples=total_samples,
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            drop_last=drop_last,
            global_batch_size=global_batch_size,
            pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
        )
        assert (
            pad_samples_to_global_batch_size is False
        ), "`MegatronPretrainingCyclicSampler` does not support sample padding"
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size

    def __len__(self):
        num_available_samples: int = self.total_samples - self.consumed_samples
        if self.global_batch_size is not None:
            if self.drop_last:
                return num_available_samples // self.global_batch_size
            else:
                return (num_available_samples + self.global_batch_size - 1) // self.global_batch_size
        else:
            if self.drop_last:
                return num_available_samples // self.micro_batch_times_data_parallel_size
            else:
                return (num_available_samples - 1) // self.micro_batch_times_data_parallel_size + 1

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # data sharding and random sampling
        bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size

        batch = []
        # Last batch if not complete will be dropped.
        for idx in range(bucket_offset, bucket_size):
            batch.append(idx + start_idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            yield batch
