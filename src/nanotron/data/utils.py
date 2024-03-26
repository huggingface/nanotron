import re
from enum import Enum
from typing import List

import numpy

from nanotron import logging

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
