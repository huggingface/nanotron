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

"""Blendable dataset."""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch

from nanotron import logging
from nanotron.logging import log_rank
from nanotron.parallel import ParallelContext
from nanotron.utils import main_rank_first
from pprint import pformat

if TYPE_CHECKING:
    from . import GPTDataset, SubsetSplitLog

logger = logging.get_logger(__name__)


@dataclass
class BlendedSubsetSplitLog:
    blended_total_num_samples: int
    blended_per_subset_samples: List[int]
    blended_subset: List["SubsetSplitLog"]


class BlendableDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: List["GPTDataset"],
        weights: List[float],
        size: int,
        parallel_context: ParallelContext,
        seed: int,
        consumed_tokens_per_dataset_folder: Optional[Dict[str, int]] = None,
        offsets_in_samples: Optional[Dict[str, int]] = None,
    ):
        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = size

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indices.
        start_time = time.time()
        # from https://github.com/NVIDIA/Megatron-LM/commit/c6e65b2e96e8376ccc84225dd1a9b60dd242fc48
        assert num_datasets < 32767
        self.dataset_index = np.zeros(self.size, dtype=np.int16)
        self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)
        self.dataset_num_samples = np.zeros(num_datasets, dtype=np.int64)
        self.random_seed = seed

        with main_rank_first(parallel_context.world_pg):
            try:
                from . import helpers
            except ImportError:
                try:
                    from .dataset_utils import compile_helper

                    compile_helper()
                    from . import helpers
                except ImportError:
                    raise ImportError(
                        "Could not compile megatron dataset C++ helper functions and therefore cannot import helpers python file."
                    )

        helpers.build_blending_indices(
            self.dataset_index,
            self.dataset_sample_index,  # sequential for each dataset_source
            self.dataset_num_samples,
            weights,
            num_datasets,
            self.size,
            torch.distributed.get_rank() == 0,
        )

        log_rank(
            "> elapsed time for building blendable dataset indices: " "{:.2f} (sec)".format(time.time() - start_time),
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        self.subset_log = BlendedSubsetSplitLog(
            blended_total_num_samples=self.size,
            blended_per_subset_samples=self.dataset_num_samples.tolist(),
            blended_subset=[d.subset_log for d in self.datasets],
        )

        # Track last 16 items
        # self.history_size = 16
        # self.last_dataset_idx = np.full(self.history_size, -1, dtype=np.int16)  # -1 indicates no data
        # self.last_dataset_sample_idx = np.full(self.history_size, -1, dtype=np.int64)
        # self.last_item_idx = np.full(self.history_size, -1, dtype=np.int64)

        # Initialize consumption tracking
        self.consumed_tokens = {idx: 0 for idx in range(len(datasets))} # current stage's consumed_tokens_per_dataset_folder
        if consumed_tokens_per_dataset_folder is not None:
            # find idx of dataset that matches the folder path
            for idx, dataset in enumerate(datasets):
                for folder_path, consumed_tokens in consumed_tokens_per_dataset_folder.items():
                    if dataset.folder_path == folder_path:
                        self.consumed_tokens[idx] = consumed_tokens
                        log_rank(f"[BlendableDataset] Setting consumed_tokens for dataset {idx} ({dataset.folder_path}) to {consumed_tokens}", logger=logger, level=logging.INFO, rank=0)
        
        self.sequence_length = None  # Will be set when first batch is processed

        # Setup offsets for already consumed tokens from previous stages
        self.offsets_in_samples = {idx: 0 for idx in range(len(datasets))} # last stage's consumed_tokens_per_dataset_folder
        if offsets_in_samples is not None:
            for idx, dataset in enumerate(datasets):
                for folder_path, offset in offsets_in_samples.items():
                    if dataset.folder_path == folder_path:
                        self.offsets_in_samples[idx] = offset
                        log_rank(f"[BlendableDataset] Applying offset {offset} samples to dataset {idx} ({dataset.folder_path})", logger=logger, level=logging.INFO, rank=0)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]

        return self.datasets[dataset_idx][sample_idx + self.offsets_in_samples[dataset_idx]] # TODO: is it okay to not respect dataset_sample_index? Since it's sequential it's okay for now

    # @property
    # def last_file_idx(self):
    #     return None if self.last_dataset_idx[-1] == -1 else self.datasets[self.last_dataset_idx[-1]].current_file

    # @property
    # def last_file_path(self):
    #     return None if self.last_dataset_idx[-1] == -1 else self.datasets[self.last_dataset_idx[-1]].current_file_path

    # @property
    # def last_dataset_path(self):
    #     return None if self.last_dataset_idx[-1] == -1 else self.datasets[self.last_dataset_idx[-1]].folder_path

    def update_consumption_metrics(self, start_idx: int, end_idx: int, sequence_length: int):
        """Update consumed samples/tokens for the current batch.

        Args:
            start_idx: Starting index of current batch for all dp ranks
            end_idx: Ending index of current batch for all dp ranks
            sequence_length: Sequence length for token calculation
        """
        if self.sequence_length is None:
            self.sequence_length = sequence_length

        # Get dataset indices for current batch
        batch_indices = self.dataset_index[start_idx:end_idx]
        unique_indices, counts = np.unique(batch_indices, return_counts=True)

        # Update consumption dictionaries
        for dataset_idx, count in zip(unique_indices, counts):
            self.consumed_tokens[dataset_idx] += int(count * sequence_length)

    def get_consumption_stats(self):
        """Get current consumption statistics for all datasets.

        Returns:
            dict: Dictionary containing samples and tokens consumed per dataset
        """
        stats = {}
        for dataset_idx, dataset in enumerate(self.datasets):
            assert (
                "s3" in dataset.folder_path
            ), "Only S3 paths are supported for consumption stats"  # TODO: remove this
            stats[dataset.folder_path] = {"tokens": self.consumed_tokens[dataset_idx]}
        return stats


class MemoryEfficientBlendableDataset(torch.utils.data.Dataset):
    """
    A BlendableDataset implementation that uses less memory than the original implementation.
    Indices are computed algorithmically instead of storing them in memory.

    To test call: MemoryEfficientBlendableDataset.test_index_blending()
    """

    def __init__(self, datasets, weights, size, weight_bins=100):
        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        weight_bins = min(weight_bins, size)

        self.size = size
        self.weight_bins = weight_bins

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        assert (weights > 0.0).all()
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        self.weights = weights / sum_weights

        # create ds index based on weights
        ds_index = []
        ds_bias = []
        for i, w in enumerate(self.weights):
            n = int(w * weight_bins)
            ds_index.extend([i] * n)
            ds_bias.extend(range(n))
        # make sure arrays have length of weight_bins
        n = weight_bins - len(ds_index)
        ds_index.extend([i] * n)
        ds_bias.extend(range(ds_bias[-1], ds_bias[-1] + n))

        self.ds_index = np.array(ds_index, dtype=np.uint32)
        self.ds_index_size = np.array([(self.ds_index == i).sum() for i in range(num_datasets)], dtype=np.uint32)
        assert (
            self.ds_index_size > 0
        ).all(), f"Some datasets have no samples in the blendable dataset, increase weight_bins or the offending weight. ds_index_size = {self.ds_index_size}"
        self.ds_bias = np.array(ds_bias, dtype=np.uint32)

        self.ds_size = np.array([len(ds) for ds in datasets], dtype=np.uint32)

    def get_ds_sample_idx(self, idx):
        """Returns ds index and sample index (within the ds) for the given index in the blendable dataset."""

        bin = idx % self.weight_bins
        ds_idx = self.ds_index[bin]
        sample_idx = (self.ds_bias[bin] + (idx // self.weight_bins) * self.ds_index_size[ds_idx]) % self.ds_size[
            ds_idx
        ]

        return ds_idx, sample_idx

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        ds_idx, sample_idx = self.get_ds_sample_idx(idx)

        return self.datasets[ds_idx][sample_idx]

    @classmethod
    def test_index_blending(cls):
        """Visualize indices of blended dataset"""

        import matplotlib.pyplot as plt

        plt.ion()

        class DS(torch.utils.data.Dataset):
            def __init__(self, size, data):
                self.size = size
                self.data = data

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return self.data[idx]

        for weight_bins in [10, 100]:
            blend_ds = MemoryEfficientBlendableDataset(
                [DS(10, "a"), DS(10, "b"), DS(10, "c")], [0.5, 0.3, 0.2], 50, weight_bins=weight_bins
            )

            ds_sample_idx_list = [blend_ds.get_ds_sample_idx(i) for i in range(50)]
            ds_list = list(zip(*ds_sample_idx_list))[0]
            sample_list = list(zip(*ds_sample_idx_list))[1]

            plt.figure()
            plt.plot(ds_list, label="ds idx")
            plt.plot(sample_list, label="sample")
            plt.legend()
            plt.grid()
            plt.title(f"weight_bins={weight_bins}")
