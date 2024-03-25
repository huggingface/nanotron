import re
from enum import Enum
from typing import List, Tuple

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


def build_blending_indices(size: int, weights: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Given multiple datasets and a weighting array, build samples
    such that it follows those weights."""
    # Create empty arrays for dataset indices and dataset sample indices
    dataset_index = numpy.empty((size,), dtype="uint")
    dataset_sample_index = numpy.empty((size,), dtype="long")

    # Initialize buffer for number of samples used for each dataset
    current_samples = numpy.zeros((len(weights),), dtype="long")

    # Iterate over all samples
    for sample_idx in range(size):

        # Convert sample index to float for comparison against weights
        sample_idx_float = max(sample_idx, 1.0)

        # Find the dataset with the highest error
        errors = weights * sample_idx_float - current_samples
        max_error_index = numpy.argmax(errors)

        # Assign the dataset index and update the sample index
        dataset_index[sample_idx] = max_error_index
        dataset_sample_index[sample_idx] = current_samples[max_error_index]

        # Update the total samples for the selected dataset
        current_samples[max_error_index] += 1

    return dataset_index, dataset_sample_index


def build_sample_idx(sizes, doc_idx, seq_length, tokens_per_epoch):
    # Check validity of inumpyut args.
    assert seq_length > 1
    assert tokens_per_epoch > 1

    # Compute the number of samples.
    num_samples = (tokens_per_epoch - 1) // seq_length

    # Allocate memory for the mapping table.
    sample_idx = numpy.full([num_samples + 1, 2], fill_value=-999, dtype=numpy.int32)

    # Setup helper vars.
    sample_index = 0
    doc_idx_index = 0
    doc_offset = 0

    # Add the first entry to the mapping table.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1

    # Loop over the rest of the samples.
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + 1

        # Keep adding docs until we reach the end of the sequence.
        while remaining_seq_length != 0:
            # Look up the current document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset

            # Try to add it to the current sequence.
            remaining_seq_length -= doc_length

            # If it fits, adjust offset and break out of inner loop.
            if remaining_seq_length <= 0:
                doc_offset += remaining_seq_length + doc_length - 1
                remaining_seq_length = 0
            else:
                # Otherwise move to the next document.
                doc_idx_index += 1
                doc_offset = 0

        # Store the current sequence in the mapping table.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    assert not numpy.any(sample_idx == -999)
    return sample_idx
