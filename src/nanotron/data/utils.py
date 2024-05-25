from typing import List

import numpy as np


def normalize(weights: List[float]) -> List[np.array]:
    """
    Normalize elements of a list

    Args:
        weights (List[float]): The weights

    Returns:
        List[numpy.array]: The normalized weights
    """
    w = np.array(weights, dtype=np.float64)
    w_sum = np.sum(w)
    w = w / w_sum
    return w


def count_dataset_indexes(dataset_idx: np.ndarray, n_datasets: int):
    counts = []

    for dataset in range(n_datasets):
        counts.append(np.count_nonzero(dataset_idx == dataset))

    return counts
