import hashlib
import json
import os
import time
from collections import OrderedDict
from typing import Dict, Union

import numpy
import torch

from nanotron import logging
from nanotron.data.indexed_dataset import MMapIndexedDataset
from nanotron.data.nanoset_configs import NanosetConfig
from nanotron.data.utils import Split
from nanotron.logging import log_rank

logger = logging.get_logger(__name__)


class Nanoset(torch.utils.data.Dataset):
    """
    The base Nanoset dataset

    Args:
        indexed_dataset (MMapIndexedDataset): The MMapIndexedDataset around which to build the
        Nanoset

        indexed_indices (numpy.ndarray): The set of indexes to sample from the MMapIndexedDataset dataset

        num_samples (int): Number of samples that we will consume from the dataset. If it is None, we will
                           consume all the samples only 1 time (valid and test splits). For the train split,
                           we will introduce train steps * global batch size and compute the number of epochs
                           based on the number of samples of the dataset.

        index_split (Split): The indexed_indices Split (train, valid, test)

        config (NanosetConfig): The Nanoset-specific container for all config sourced parameters
    """

    def __init__(
        self,
        indexed_dataset: MMapIndexedDataset,
        indexed_indices: numpy.ndarray,
        num_samples: Union[int, None],
        index_split: Split,
        config: NanosetConfig,
    ) -> None:

        self.indexed_dataset = indexed_dataset
        self.indexed_indices = indexed_indices
        self.num_samples = num_samples
        self.index_split = index_split
        self.config = config

        # Create unique identifier

        self.unique_identifiers = OrderedDict()
        self.unique_identifiers["class"] = type(self).__name__
        self.unique_identifiers["path_to_mmap"] = self.indexed_dataset.path_to_mmap
        self.unique_identifiers["index_split"] = self.index_split.name
        self.unique_identifiers["split"] = self.config.split
        self.unique_identifiers["random_seed"] = self.config.random_seed
        self.unique_identifiers["sequence_length"] = self.config.sequence_length

        self.unique_description = json.dumps(self.unique_identifiers, indent=4)
        self.unique_description_hash = hashlib.md5(self.unique_description.encode("utf-8")).hexdigest()

        self.shuffle_index = self.build_shuffle_indices()

        # Check size
        _ = self[len(self) - 1]
        try:
            _ = self[len(self)]
            raise RuntimeError("Nanoset size is improperly bounded")
        except IndexError:
            pass

    def __len__(self) -> int:
        """
        Returns:
            int: The number of samples of the Nanoset
        """

        return self.shuffle_index.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, numpy.ndarray]:
        """Get the input ids for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, numpy.ndarray]: The input ids wrapped in a dictionary
        """

        # Do the shuffle mapping
        idx = self.shuffle_index[idx]
        # torch can't convert np.ndarray of type numpy.uint16
        return {"input_ids": self.indexed_dataset.get(idx, self.config.sequence_length).astype(numpy.int32)}

    def build_shuffle_indices(self) -> numpy.ndarray:
        """
        Build the shuffle index

        Shuffle index:
            -- 1-D
            -- A random permutation of index range of the indexed indices of the memmap dataset for this split

        Returns:
            numpy.ndarray: The shuffle index
        """

        path_to_cache = self.config.path_to_cache
        if path_to_cache is None:
            path_to_cache = os.path.join(
                self.indexed_dataset.path_to_mmap[:-4], "cache", f"{type(self).__name__}_indices"
            )

        def get_path_to(suffix):
            return os.path.join(path_to_cache, f"{self.unique_description_hash}-{type(self).__name__}-{suffix}")

        path_to_shuffle_index = get_path_to("shuffle_index.npy")
        cache_hit = os.path.isfile(path_to_shuffle_index)

        if not cache_hit and torch.distributed.get_rank() == 0:

            os.makedirs(path_to_cache, exist_ok=True)

            numpy_random_state = numpy.random.RandomState(self.config.random_seed)

            # Build the shuffle index
            log_rank(
                f"Build and save the shuffle index to {os.path.basename(path_to_shuffle_index)} for the {type(self).__name__} {self.index_split.name} split",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
            t_beg = time.time()
            shuffle_index = self.indexed_indices.copy()
            numpy_random_state.shuffle(shuffle_index)
            if self.num_samples is not None:  # For the train split, concatenate shuffle Indexes
                n_concatenations = (
                    int(self.num_samples / shuffle_index.shape[0]) + 1
                )  # NOTE: To ensure that we always generate more samples than requested in num_samples
                shuffle_index = numpy.concatenate([shuffle_index for _ in range(n_concatenations)])
            numpy.save(
                path_to_shuffle_index,
                shuffle_index,
                allow_pickle=True,
            )
            t_end = time.time()
            log_rank(f"\t> Time elapsed: {t_end - t_beg:4f} seconds", logger=logger, level=logging.DEBUG, rank=0)

        log_rank(
            f"Load the shuffle index from {os.path.basename(path_to_shuffle_index)} for the {type(self).__name__} {self.index_split.name} split",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        t_beg = time.time()
        shuffle_index = numpy.load(path_to_shuffle_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_rank(f"\t> Time elapsed: {t_end - t_beg:4f} seconds", logger=logger, level=logging.DEBUG, rank=0)

        log_rank(f"> Total number of samples: {len(self.indexed_indices)}", logger=logger, level=logging.INFO, rank=0)

        if self.num_samples is not None:
            # Compute number of epochs we will iterate over this Nanoset. Just for training
            num_epochs = round(self.num_samples / (len(self.indexed_indices)), 2)
            log_rank(f"> Total number of epochs: {num_epochs}", logger=logger, level=logging.INFO, rank=0)

        return shuffle_index
