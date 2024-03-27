import re
from enum import Enum
from typing import List

import numpy

from nanotron import logging
from nanotron.logging import log_rank

logger = logging.get_logger(__name__)


class Split(Enum):
    train = 0
    valid = 1
    test = 2


def normalize(weights: List[float]) -> List[float]:
    """Do non-exponentiated normalization

    Args:
        weights (List[float]): The weights

    Returns:
        List[float]: The normalized weights
    """
    w = numpy.array(weights, dtype=numpy.float64)
    w_sum = numpy.sum(w)
    w = (w / w_sum).tolist()
    return w


def parse_and_normalize_split(split: str) -> List[float]:
    """Parse the dataset split ratios from a string

    Args:
        split (str): The train valid test split string e.g. "99,1,0"

    Returns:
        List[float]: The trian valid test split ratios e.g. [99.0, 1.0, 0.0]
    """
    split = list(map(float, re.findall(r"[.0-9]+", split)))
    split = split + [0.0 for _ in range(len(Split) - len(split))]

    assert len(split) == len(
        Split
    ), "Introduce the split ratios of the datasets in a comma separated string (e.g. 99,1,0)"
    assert all((_ >= 0.0 for _ in split)), "Don't introduce negative values in the split config. Given: {split}"

    split = normalize(split)

    return split


def count_blending_indexes(dataset_idx: numpy.ndarray, n_datasets: int):
    counts = []

    for dataset in range(n_datasets):
        counts.append(numpy.count_nonzero(dataset_idx == dataset))

    return counts


def log_BlendedNanoset_stats(blended_nanosets):
    for blended_nanoset, split_name in zip(blended_nanosets, Split):
        log_rank(
            f"> Total number of samples of the {split_name._name_} BlendedNanoset: {len(blended_nanoset)}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        nanoset_sample_count = count_blending_indexes(blended_nanoset.dataset_index, len(blended_nanoset.datasets))
        for idx, nanoset in enumerate(blended_nanoset.datasets):
            log_rank(
                f">   Total number of samples from the {nanoset.indexed_dataset.path_prefix.rsplit('/', 1)[-1]} Nanoset: {nanoset_sample_count[idx]} ({round(normalize(nanoset_sample_count)[idx], 2)})",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
