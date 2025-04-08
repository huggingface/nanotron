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
from typing import TYPE_CHECKING, List

import numpy as np
import torch

from nanotron import logging
from nanotron.logging import log_rank
from nanotron.parallel import ParallelContext
from nanotron.utils import main_rank_first

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
            self.dataset_sample_index,
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

        # numpy_random_state = np.random.RandomState(self.random_seed)
        # numpy_random_state.shuffle(self.dataset_index)
        # numpy_random_state = np.random.RandomState(self.random_seed)
        # numpy_random_state.shuffle(self.dataset_sample_index)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]
        return self.datasets[dataset_idx][sample_idx]


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
