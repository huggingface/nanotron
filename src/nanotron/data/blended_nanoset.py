import hashlib
import json
import os
import time
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import numpy
import torch

from nanotron import logging
from nanotron.data.nanoset import Nanoset
from nanotron.data.nanoset_configs import NanosetConfig
from nanotron.logging import log_rank

logger = logging.get_logger(__name__)


class BlendedNanoset(torch.utils.data.Dataset):
    """
    Conjugating class for a set of Nanoset instances

    Args:
        datasets (List[Nanoset]): The Nanoset instances to blend

        weights (List[float]): The weights which determines the dataset blend ratios

        size (int): The number of samples to draw from the blend

        config (NanosetConfig): The config object which informs dataset creation
    """

    def __init__(
        self,
        datasets: List[Nanoset],
        weights: List[float],
        size: int,
        config: NanosetConfig,
    ) -> None:
        assert len(datasets) == len(weights)
        assert numpy.isclose(sum(weights), 1.0)
        assert all((isinstance(dataset, Nanoset) for dataset in datasets))

        self.datasets = datasets
        self.weights = weights
        self.config = config

        # For the train split, we will have global batch size * train steps samples
        # For the valid and test splits, we will consume entirely both datasets
        self.dataset_sizes = [len(dataset) for dataset in self.datasets]
        self.size = size if size is not None else sum(self.dataset_sizes)

        # Create unique identifier

        unique_identifiers = OrderedDict()
        unique_identifiers["class"] = type(self).__name__
        unique_identifiers["datasets"] = [dataset.unique_identifiers for dataset in self.datasets]
        unique_identifiers["weights"] = self.weights
        unique_identifiers["size"] = self.size
        unique_identifiers["dataset_sizes"] = self.dataset_sizes

        self.unique_description = json.dumps(unique_identifiers, indent=4)
        self.unique_description_hash = hashlib.md5(self.unique_description.encode("utf-8")).hexdigest()

        self.dataset_index, self.dataset_sample_index = self.build_indices()

        # Check size
        _ = self[self.size - 1]
        try:
            _ = self[self.size]
            raise RuntimeError("BlendedNanoset size is improperly bounded")
        except IndexError:
            pass

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Union[int, numpy.ndarray]]:
        dataset_id = self.dataset_index[idx]
        dataset_sample_id = self.dataset_sample_index[idx]

        return self.datasets[dataset_id][dataset_sample_id]

    def build_indices(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Build and optionally cache the dataset index and the dataset sample index

        The dataset index is a 1-D mapping which determines the dataset to query. The dataset
        sample index is a 1-D mapping which determines the sample to request from the queried
        dataset.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The dataset index and the dataset sample index
        """
        path_to_cache = self.config.path_to_cache

        if path_to_cache:

            def get_path_to(suffix):
                return os.path.join(path_to_cache, f"{self.unique_description_hash}-{type(self).__name__}-{suffix}")

            path_to_dataset_index = get_path_to("dataset_index.npy")
            path_to_dataset_sample_index = get_path_to("dataset_sample_index.npy")
            cache_hit = all(map(os.path.isfile, [path_to_dataset_index, path_to_dataset_sample_index]))

        else:
            cache_hit = False

        if not path_to_cache or (not cache_hit and torch.distributed.get_rank() == 0):

            log_rank(f"Build and save the {type(self).__name__} indices", logger=logger, level=logging.INFO, rank=0)

            t_beg = time.time()

            dataset_index, dataset_sample_index = build_blending_indices(
                n_samples=self.size, weights=numpy.array(self.weights), dataset_sizes=self.dataset_sizes
            )

            if path_to_cache:
                os.makedirs(path_to_cache, exist_ok=True)

                # Save the indexes
                numpy.save(path_to_dataset_index, dataset_index, allow_pickle=True)
                numpy.save(path_to_dataset_sample_index, dataset_sample_index, allow_pickle=True)
            else:
                log_rank(
                    "Unable to save the indexes because path_to_cache is None",
                    logger=logger,
                    level=logging.WARNING,
                    rank=0,
                )

            t_end = time.time()
            log_rank(f"\t> Time elapsed: {t_end - t_beg:4f} seconds", logger=logger, level=logging.DEBUG, rank=0)

            return dataset_index, dataset_sample_index

        log_rank(f"Load the {type(self).__name__} indices", logger=logger, level=logging.INFO, rank=0)

        log_rank(f"\tLoad the dataset index from {path_to_dataset_index}", logger=logger, level=logging.INFO, rank=0)
        t_beg = time.time()
        dataset_index = numpy.load(path_to_dataset_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_rank(f"\t> Time elapsed: {t_end - t_beg:4f} seconds", logger=logger, level=logging.DEBUG, rank=0)

        log_rank(
            f"\tLoad the dataset sample index from {path_to_dataset_sample_index}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        t_beg = time.time()
        dataset_sample_index = numpy.load(path_to_dataset_sample_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_rank(f"\t> Time elapsed: {t_end - t_beg:4f} seconds", logger=logger, level=logging.DEBUG, rank=0)

        return dataset_index, dataset_sample_index


def build_blending_indices(
    n_samples: int, weights: numpy.ndarray, dataset_sizes: List
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Given multiple datasets and a weighting array, build samples indexes
    such that it follows those weights."""
    # Create empty arrays for dataset indices and dataset sample indices
    dataset_index = numpy.empty((n_samples,), dtype="uint")
    dataset_sample_index = numpy.empty((n_samples,), dtype="long")

    # Initialize buffer for number of samples used for each dataset
    current_samples = numpy.zeros((len(weights),), dtype="long")

    # Iterate over all samples
    for sample_idx in range(n_samples):

        # Convert sample index to float for comparison against weights
        sample_idx_float = max(sample_idx, 1.0)

        # Find the dataset with the highest error
        errors = weights * sample_idx_float - current_samples
        max_error_index = numpy.argmax(errors)

        # Assign the dataset index and update the sample index
        dataset_index[sample_idx] = max_error_index
        dataset_sample_index[sample_idx] = current_samples[max_error_index] % dataset_sizes[max_error_index]

        # Update the total samples for the selected dataset
        current_samples[max_error_index] += 1

    return dataset_index, dataset_sample_index
