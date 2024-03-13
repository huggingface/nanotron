from nanotron import logging
import math
from typing import Any, List, Optional, Tuple, Type, Union

import numpy
import torch

from nanotron.data.nanoset import NanosetConfig, Nanoset
from nanotron.data.indexed_dataset import MMapIndexedDataset
from nanotron.data.utils import Split, normalize

logger = logging.get_logger(__name__)

DistributedDataset = Union[Nanoset, MMapIndexedDataset]

class NanosetBuilder(object):
    """Builder class for the Nanoset classes

    Args:

        config (NanosetConfig): The config object which informs dataset creation
    """

    def __init__(
        self, config: NanosetConfig,
    ):
        self.config = config
        self.sizes = config.split_sizes 
        self.cls = Nanoset # NOTE: keep it like that to support BlendedNanoset in the future

    def build(self) -> List[Nanoset]:
        """Build all dataset splits according to the provided data_path(s)
        
        This method is distributed-aware and must be called on all ranks.
        
        The dataset splits returned can vary according to the config. Supply config.data_path and
        config.split to build BlendedNanoset and/or Nanoset splits from the same
        distribution. Supply config.data_path_per_split to build BlendedNanoset and/or Nanoset
        splits from separate distributions.

        Returns:
            List[Union[BlendedNanoset, Nanoset]]: A list of either
            Nanoset or BlendedNanoset per split
        """
        return self._build_blended_dataset_splits()

    def _build_blended_dataset_splits(
        self,
    ) -> List[Nanoset]:
        """Build all dataset splits according to the provided data_path(s)
        
        See the NanosetBuilder.build alias for more information.

        Returns:
            List[Optional[Union[BlendedNanoset, Nanoset]]]: A list of either
            Nanoset or BlendedNanoset per split
        """

        data_path = getattr(self.config, "data_path")
        split = getattr(self.config, "split_vector")

        # NOTE: For including blended datasets (BlendedNanoset)
        # NOTE: Refer to Megatron to check how to work with multiple data_paths datasets. For now we don't support it
        # NOTE: https://github.com/TJ-Solergibert/Megatron-debug/blob/c5eda947d9728d21b03d77b7db56cb71513d5636/megatron/core/datasets/blended_megatron_dataset_builder.py#L81
        data_path = [data_path]
        if len(data_path) == 1:
            return self._build_nanoset_dataset_splits(data_path[0], split, self.sizes)

    def _build_nanoset_dataset_splits(
        self, path_prefix: str, split: List[float], sizes: List[int],
    ) -> List[Nanoset]:
        """Build each Nanoset split from a single MMapIndexedDataset

        Args:
            path_prefix (str): The MMapIndexedDataset .bin and .idx file prefix

            split (List[float]): The dataset split ratios (must sum to 1.00)

            sizes (List[int]): The number of total samples to draw from each split

        Returns:
            List[Nanoset]: The Nanoset per split. Always returns Nanosets because we build them in each and every rank 
        """
        
        indexed_dataset = self._build_generic_dataset(
            MMapIndexedDataset, path_prefix, False
        )
        
        split_idx_bounds = _get_split_indices(
            split, indexed_dataset.sequence_lengths.shape[0]
        )
        
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
                    self._build_generic_dataset(
                        self.cls, indexed_dataset, split_indices[i], sizes[i], _split, self.config
                    )
                )

        return nanoset_datasets

    def _build_generic_dataset(
        self, cls: Type[DistributedDataset], *args: Any,
    ) -> Optional[DistributedDataset]:
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
                        f"Failed to write dataset materials to the data cache directory. "
                        + f"Please supply a directory to which you have write access via "
                        + f"the path_to_cache attribute in NanosetConfig and "
                        + f"retry. Refer to the preserved traceback above for more information."
                    )
                    raise Exception(log) from err

            torch.distributed.barrier()

            if rank != 0:
                dataset = cls(*args)

            return dataset

        return cls(*args)


def _get_split_indices(split: List[float], num_elements: int) -> List[int]:
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
    split_indices[1:] = list(
        map(lambda _: _ - (split_indices[-1] - num_elements), split_indices[1:])
    )

    assert len(split_indices) == len(split) + 1
    assert split_indices[-1] == num_elements

    return split_indices

# NOTE: Keep for BlendedNanoset
def _get_prefixes_weights_and_sizes_for_blend(
    data_path: List[str], target_num_samples_per_split: List[int]
) -> Tuple[List[str], List[float], List[List[int]]]:
    """Determine the contribution of the Nanoset splits to the BlendedNanoset splits
    
    Args:
        data_path (List[str]): e.g. ["30", "path/to/dataset_1_prefix", "70", 
        "path/to/dataset_2_prefix"]

        target_num_samples_per_split (List[int]): The number of samples to target for each
        BlendedNanoset split

    Returns:
        Tuple[List[str], List[float], List[List[int]]]: The prefix strings e.g.
        ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], the normalized weights e.g.
        [0.3, 0.7], and the number of samples to request per Nanoset per split
    """
    weights, prefixes = zip(
        *[(float(data_path[i]), data_path[i + 1].strip()) for i in range(0, len(data_path), 2)]
    )

    weights = normalize(weights)

    # NOTE: Use 0.5% target margin to ensure we satiate the network
    sizes_per_dataset = [
        [
            int(math.ceil(target_num_samples * weight * 1.005))
            for target_num_samples in target_num_samples_per_split
        ]
        for weight in weights
    ]

    return prefixes, weights, sizes_per_dataset