from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from nanotron import logging
from nanotron.data.utils import count_dataset_indexes, normalize
from nanotron.logging import log_rank
from numba import jit

logger = logging.get_logger(__name__)


class Nanoset(torch.utils.data.Dataset):
    """
    The Nanoset dataset

    Args:
        dataset_paths (List[str]): List of paths to tokenized datasets
        dataset_weights (List[float]): List with the weights for weighted datasets. If None, consume all samples from all datasets without weighting. Weights are normalized in __init__
        sequence_length (int): Sequence length of the built samples
        token_dtype (Union[np.uint16, np.int32]): dtype of the tokens stored in the processed dataset files. np.uin16 for vocab sizes < 65535, np.int32 otherwise
        train_split_num_samples (int): Number of samples the dataset needs. It's the training steps * global batch size
    """

    def __init__(
        self,
        dataset_paths: List[str],
        dataset_weights: Union[List[float], None],
        sequence_length: int,
        token_dtype: Union[np.uint16, np.int32],
        train_split_num_samples: int,
        random_seed: int = 1234,
    ) -> None:

        # Init
        self.dataset_paths = dataset_paths
        self.dataset_weights = dataset_weights
        self.sequence_length = sequence_length
        self.token_dtype = token_dtype
        self.train_split_num_samples = train_split_num_samples
        self.random_seed = random_seed

        # Build Nanoset Index
        ## To build the index we need the length of each dataset
        self.dataset_lengths = []
        for dataset_path in self.dataset_paths:
            self.dataset_buffer_mmap = np.memmap(dataset_path, mode="r", order="C", dtype=self.token_dtype)
            self.dataset_buffer = memoryview(self.dataset_buffer_mmap)
            dataset_number_of_tokens = int(len(self.dataset_buffer))
            number_of_samples = int(
                (dataset_number_of_tokens - 1) / sequence_length
            )  # Discard last sample if length < sequence_length
            self.dataset_lengths.append(number_of_samples)
        ## Set dataset weights
        if (
            self.dataset_weights is None
        ):  # Case of training with > 1 datasets without weighting them: Consume both datasets entirely on each epoch
            self.dataset_weights = normalize(self.dataset_lengths)
        else:
            self.dataset_weights = normalize(dataset_weights)
        ## Build dataset index and dataset sample index
        self.dataset_index, self.dataset_sample_index = self.build_nanoset_index()

        self.print_nanoset_info()

    def __len__(self) -> int:
        """
        Returns:
            int: The number of samples of the Nanoset
        """

        return len(self.dataset_index)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns sequence_length + 1 tokens from the memmap dataset

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, numpy.ndarray]: The input ids wrapped in a dictionary
        """

        dataset = self.dataset_index[idx]
        dataset_sample = self.dataset_sample_index[idx]

        # Rebuild the memmap in every access to free memory
        # https://stackoverflow.com/a/61472122
        self.dataset_buffer_mmap = np.memmap(self.dataset_paths[dataset], mode="r", order="C", dtype=self.token_dtype)
        self.dataset_buffer = memoryview(self.dataset_buffer_mmap)

        # uint16 -> 2 bytes per token, int32 -> 4 bytes per token
        offset = dataset_sample * self.sequence_length * (np.iinfo(self.token_dtype).bits / 8)
        input_ids_tokens = np.frombuffer(
            self.dataset_buffer, dtype=self.token_dtype, count=(self.sequence_length + 1), offset=int(offset)
        )

        # Return tokens as np.int32 as Torch can't handle uint16
        return {"input_ids": input_ids_tokens.astype(np.int32)}

    def build_nanoset_index(self) -> np.ndarray:
        """
        Build dataset index and dataset sample index
        """
        # Compute samples per epoch and number of epochs
        samples_per_epoch = sum(self.dataset_lengths)
        num_epochs = int(self.train_split_num_samples / samples_per_epoch) + 1
        # Build the dataset indexes for 1 epoch
        dataset_index, dataset_sample_index = build_nanoset_index_helper(
            n_samples=samples_per_epoch, weights=self.dataset_weights, dataset_sizes=self.dataset_lengths
        )
        # Shuffle the indexes the same way
        numpy_random_state = np.random.RandomState(self.random_seed)
        numpy_random_state.shuffle(dataset_index)
        numpy_random_state = np.random.RandomState(self.random_seed)
        numpy_random_state.shuffle(dataset_sample_index)
        # Concatenate num_epochs the shuffled indexes
        dataset_index = np.concatenate([dataset_index for _ in range(num_epochs)])
        dataset_sample_index = np.concatenate([dataset_sample_index for _ in range(num_epochs)])
        # Just keep the necessary samples
        dataset_index = dataset_index[: self.train_split_num_samples]
        dataset_sample_index = dataset_sample_index[: self.train_split_num_samples]

        return dataset_index, dataset_sample_index

    def __del__(self) -> None:
        """
        Clean up Nanoset
        """

        if hasattr(self, "dataset_buffer_mmap"):
            self.dataset_buffer_mmap._mmap.close()
        del self.dataset_buffer_mmap

    def print_nanoset_info(self):

        log_rank(f"> Total number of samples: {len(self)}", logger=logger, level=logging.INFO, rank=0)
        log_rank(
            f"> Total number of tokens: {len(self) * self.sequence_length}", logger=logger, level=logging.INFO, rank=0
        )

        # Print samples from each dataset + weight
        dataset_sample_count = count_dataset_indexes(self.dataset_index, len(self.dataset_paths))
        for index, sample_count in enumerate(dataset_sample_count):
            log_rank(
                f">   Total number of samples from the {self.dataset_paths[index].rsplit('/', 1)[-1]} dataset: {sample_count} ({round(normalize(dataset_sample_count).tolist()[index], 2)})",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )


@jit(nopython=True, cache=True)
def build_nanoset_index_helper(
    n_samples: int, weights: np.ndarray, dataset_sizes: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given multiple datasets and a weighting array, build samples indexes
    such that it follows those weights
    """
    # Create empty arrays for dataset indices and dataset sample indices
    dataset_index = np.empty((n_samples,), dtype="uint")
    dataset_sample_index = np.empty((n_samples,), dtype="long")  # Supports dataset with up to 2**64 samples

    # Initialize buffer for number of samples used for each dataset
    current_samples = np.zeros((len(weights),), dtype="long")

    # Iterate over all samples
    for sample_idx in range(n_samples):

        # Convert sample index to float for comparison against weights
        sample_idx_float = max(sample_idx, 1.0)

        # Find the dataset with the highest error
        errors = weights * sample_idx_float - current_samples
        max_error_index = np.argmax(errors)

        # Assign the dataset index and update the sample index
        dataset_index[sample_idx] = max_error_index
        dataset_sample_index[sample_idx] = current_samples[max_error_index] % dataset_sizes[max_error_index]

        # Update the total samples for the selected dataset
        current_samples[max_error_index] += 1

    return dataset_index, dataset_sample_index
