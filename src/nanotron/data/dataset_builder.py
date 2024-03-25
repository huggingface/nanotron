import math
from typing import Any, List, Type, Union

import numpy
import torch

from nanotron import logging
from nanotron.data.blended_nanoset import BlendedNanoset
from nanotron.data.indexed_dataset import MMapIndexedDataset
from nanotron.data.nanoset import Nanoset
from nanotron.data.nanoset_configs import NanosetConfig
from nanotron.data.utils import Split, normalize

logger = logging.get_logger(__name__)

DistributedDataset = Union[Nanoset, MMapIndexedDataset, BlendedNanoset]


class NanosetBuilder(object):
    """Builder class for the Nanoset classes

    Args:

        config (NanosetConfig): The config object which informs dataset creation
    """

    def __init__(
        self,
        config: NanosetConfig,
    ):
        self.config = config

    def build(self) -> List[Union[BlendedNanoset, Nanoset]]:
        """Build all dataset splits according to the provided data_path(s)

        This method is distributed-aware and must be called on all ranks.

        The dataset splits returned can vary according to the config. Supply config.data_path and
        config.split to build BlendedNanoset and/or Nanoset splits from the same
        distribution.

        Returns:
            List[Union[BlendedNanoset, Nanoset]]: A list of either Nanoset or BlendedNanoset per split
        """

        data_path = self.config.data_path
        split = self.config.split_vector

        # Single Nanoset
        if isinstance(data_path, str):
            return self.build_nanoset_dataset_splits(data_path, split, self.config.split_num_samples)

        # Blended Nanoset
        prefix_per_dataset = list(data_path.keys())
        weight_per_dataset = normalize(list(data_path.values()))

        # NOTE: Use 0.5% target margin to ensure that that we do not exhaust the indices of any dataset in the Blend
        sizes_per_dataset = [
            [
                int(math.ceil(target_num_samples * weight * 1.005))
                for target_num_samples in self.config.split_num_samples
            ]
            for weight in weight_per_dataset
        ]

        nanoset_datasets = [[] for _ in range(len(Split))]

        for i in range(len(prefix_per_dataset)):
            nanoset_datasets_split = self.build_nanoset_dataset_splits(
                prefix_per_dataset[i], split, sizes_per_dataset[i]
            )

            for j in range(len(nanoset_datasets_split)):
                nanoset_datasets[j].append(nanoset_datasets_split[j])

        # Sum over all contributing datasets, per split
        size_per_split = list(map(sum, zip(*sizes_per_dataset)))

        blended_nanosets = []

        for i in range(len(nanoset_datasets)):
            is_none = (_ is None for _ in nanoset_datasets[i])

            if split[i] == 0.0:
                assert all(is_none)
                blended_nanosets.append(None)
            else:
                assert not any(is_none)
                blended_nanosets.append(
                    self.build_generic_dataset(
                        BlendedNanoset,
                        nanoset_datasets[i],
                        weight_per_dataset,
                        size_per_split[i],
                        self.config,
                    )
                )

        return blended_nanosets

    def build_nanoset_dataset_splits(
        self,
        path_prefix: str,
        split: List[float],
        sizes: List[int],
    ) -> List[Nanoset]:
        """Build each Nanoset split from a single MMapIndexedDataset

        Args:
            path_prefix (str): The MMapIndexedDataset .bin and .idx file prefix

            split (List[float]): The dataset split ratios (must sum to 1.00)

            sizes (List[int]): The number of total samples to draw from each split

        Returns:
            List[Nanoset]: The Nanoset per split. Always returns Nanosets because we build them in each and every rank
        """
        indexed_dataset = self.build_generic_dataset(MMapIndexedDataset, path_prefix)

        split_idx_bounds = get_split_indices(split, indexed_dataset.sequence_lengths.shape[0])

        split_indices = [
            numpy.arange(
                start=split_idx_bounds[i],
                stop=split_idx_bounds[i + 1],
                step=1,
                dtype=numpy.int32,
            )
            for i, _ in enumerate(Split)
        ]

        nanoset_datasets = []
        for i, _split in enumerate(Split):
            if split[i] == 0.0:
                nanoset_datasets.append(None)
            else:
                nanoset_datasets.append(
                    self.build_generic_dataset(
                        Nanoset, indexed_dataset, split_indices[i], sizes[i], _split, self.config
                    )
                )

        return nanoset_datasets

    def build_generic_dataset(
        self,
        cls: Type[DistributedDataset],
        *args: Any,
    ) -> DistributedDataset:
        """Build the DistributedDataset

        Args:
            cls (Type[DistributedDataset]): The DistributedDataset class to be built

            args (Tuple[Any]): The positional arguments used to build the provided
            DistributedDataset class

        Raises:
            Exception: When the dataset constructor raises an OSError

        Returns:
            DistributedDataset: The DistributedDataset instantion
        """
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()

            dataset = None

            # First, build on rank 0
            if rank == 0:
                try:
                    dataset = cls(*args)
                except OSError as err:
                    log = (
                        "Failed to write dataset materials to the data cache directory. "
                        + "Please supply a directory to which you have write access via "
                        + "the path_to_cache attribute in NanosetConfig and "
                        + "retry. Refer to the preserved traceback above for more information."
                    )
                    raise Exception(log) from err

            torch.distributed.barrier()

            if rank != 0:
                dataset = cls(*args)

            return dataset

        return cls(*args)


def get_split_indices(split: List[float], num_elements: int) -> List[int]:
    """Determine the document index bounds per split

    Args:
        split (List[float]): The dataset split ratios (must sum to 1.00)

        num_elements (int): The number of elements, e.g. sequences or documents, available for
        the split

    Returns:
        List[int]: The indices for all three splits e.g. [0, 900, 990, 1000] for a 1000-document
        set and a [90.0, 9.0, 1.0] split
    """
    split_indices = [0]
    for split_pct in split:
        split_indices.append(split_indices[-1] + int(round(split_pct * float(num_elements))))
    split_indices[1:] = [_ - (split_indices[-1] - num_elements) for _ in split_indices[1:]]

    assert len(split_indices) == len(split) + 1
    assert split_indices[-1] == num_elements

    return split_indices
