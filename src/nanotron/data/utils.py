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


def compile_helpers():
    """Compile C++ helper functions at runtime. Make sure this is invoked on a single process."""
    import os
    import subprocess

    command = ["make", "-C", os.path.abspath(os.path.dirname(__file__))]
    if subprocess.run(command).returncode != 0:
        import sys

        log_rank("Failed to compile the C++ dataset helper functions", logger=logger, level=logging.ERROR, rank=0)
        sys.exit(1)


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

    assert len(split) == len(Split)
    assert all((_ >= 0.0 for _ in split))

    split = normalize(split)

    return split


def compute_datasets_num_samples(train_iters, eval_interval, eval_iters, global_batch_size):

    train_samples = train_iters * global_batch_size
    test_iters = eval_iters
    eval_iters = (train_iters // eval_interval + 1) * eval_iters

    datasets_num_samples = [train_samples, eval_iters * global_batch_size, test_iters * global_batch_size]

    log_rank(" > Datasets target sizes (minimum size):", logger=logger, level=logging.INFO, rank=0)
    log_rank("    Train:      {}".format(datasets_num_samples[0]), logger=logger, level=logging.INFO, rank=0)
    log_rank("    Validation: {}".format(datasets_num_samples[1]), logger=logger, level=logging.INFO, rank=0)
    log_rank("    Test:       {}".format(datasets_num_samples[2]), logger=logger, level=logging.INFO, rank=0)

    return datasets_num_samples
