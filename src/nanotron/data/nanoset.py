import hashlib
import json
import os
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from datatrove.utils.dataset import DatatroveFolderDataset
from numba import jit
from tqdm import tqdm

from nanotron import logging
from nanotron.data.utils import count_dataset_indexes, normalize
from nanotron.logging import log_rank

logger = logging.get_logger(__name__)


class Nanoset(torch.utils.data.Dataset):
    """
    The Nanoset dataset

    Args:
        dataset_folders (List[str]): List of folders with tokenized datasets
        dataset_weights (Union[List[float], None]): List with the weights for weighted datasets. If None, consume all samples from all datasets without weighting. Weights are normalized in __init__
        sequence_length (int): Sequence length of the built samples
        vocab_size (int): Vocab size of the tokenizer used to tokenize the dataset
        train_split_num_samples (int): Number of samples the dataset needs. It's the training steps * global batch size
    """

    def __init__(
        self,
        dataset_folders: List[str],
        sequence_length: int,
        token_size: int,
        train_split_num_samples: int,
        dataset_weights: Union[List[float], None] = None,
        random_seed: int = 1234,
        use_cache: bool = True,
        eos_token_id: int = None,
        return_positions: bool = True,
    ) -> None:

        # Checks
        if isinstance(dataset_folders, str):
            warnings.warn("dataset_folders should be of type List[str] but str was provided. Converting to List[str]")
            dataset_folders = [dataset_folders]

        # Init
        self.dataset_folders = dataset_folders
        self.sequence_length = sequence_length
        self.eos_token_id = eos_token_id
        self.return_positions = return_positions
        assert (
            self.return_positions or self.eos_token_id is not None
        ), "If return_positions is True, eos_token_id must be defined"
        # Number of bytes for the tokens stored in the processed dataset files. 2 for vocab sizes < 65535, 4 otherwise
        self.token_size = token_size
        self.train_split_num_samples = train_split_num_samples
        self.random_seed = random_seed
        self.use_cache = use_cache
        self.cache_dir = "./.nanoset_cache"
        self.datatrove_datasets = []
        for dataset_folder in self.dataset_folders:
            self.datatrove_datasets.append(
                DatatroveFolderDataset(
                    folder_path=dataset_folder,
                    filename_pattern=os.path.join(dataset_folder, "*.ds"),
                    seq_len=sequence_length,
                    recursive=False,
                    token_size=self.token_size,
                    shuffle=True,
                    return_positions=self.return_positions,  # if set to True, the position ids are directly build datatrove
                    eos_token_id=self.eos_token_id,
                )
            )

        # Build Nanoset Index
        ## To build the index we need the length of each dataset
        self.dataset_lengths = [len(datatrove_dataset) for datatrove_dataset in self.datatrove_datasets]
        ## Set dataset weights
        if (
            dataset_weights is None
        ):  # Case of training with > 1 datasets without weighting them: Consume both datasets entirely on each epoch
            self.dataset_weights = normalize(self.dataset_lengths)
        else:
            self.dataset_weights = normalize(dataset_weights)
        assert len(dataset_folders) == len(
            self.dataset_weights
        ), f"Specified {len(self.dataset_weights)} weights but {len(dataset_folders)} datasets were provided."
        ## Build dataset index and dataset sample index
        self.dataset_index, self.dataset_sample_index = self.build_nanoset_index()
        # self.dataset_index, self.dataset_sample_index = self.new_build_nanoset_index() # TODO: Fix this

        self.print_nanoset_info()

    def __len__(self) -> int:
        """
        Returns:
            int: The number of samples of the Nanoset
        """

        return len(self.dataset_index)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns sequence_length + 1 tokens from the memmap dataset using sequential access
        """
        dataset = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]

        # Get actual sample index by wrapping around dataset length
        actual_sample = sample_idx % self.dataset_lengths[dataset]

        return self.datatrove_datasets[dataset][actual_sample]

    def new_build_nanoset_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build dataset index that enables sequential reading while respecting weights.
        Uses cache if available and parameters match.
        """
        # Create a cache key based on the parameters that affect the index
        cache_params = {
            "dataset_folders": self.dataset_folders,
            "dataset_lengths": self.dataset_lengths,
            "dataset_weights": self.dataset_weights.tolist(),
            "train_split_num_samples": self.train_split_num_samples,
            "random_seed": self.random_seed,
            "token_size": self.token_size,
            "sequence_length": self.sequence_length,
        }

        # Create a deterministic cache key
        cache_key = hashlib.md5(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"index_{cache_key}.npz")

        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                logger.info(f"[Nanoset] Loading index from cache: {cache_file}")
                cached_data = np.load(cache_file)
                return cached_data["dataset_index"], cached_data["dataset_sample_index"]
            except Exception as e:
                logger.warning(f"[Nanoset] Failed to load cache, rebuilding index: {e}")

        logger.info(f"[Nanoset] Building sequential Nanoset index for {len(self.dataset_folders)} datasets")

        # Original index building logic
        total_weighted_samples = np.array(self.dataset_weights) * self.train_split_num_samples
        samples_per_dataset = np.floor(total_weighted_samples).astype(np.int64)

        remaining = self.train_split_num_samples - samples_per_dataset.sum()
        if remaining > 0:
            fractional_parts = total_weighted_samples - samples_per_dataset
            indices = np.argsort(fractional_parts)[-remaining:]
            samples_per_dataset[indices] += 1

        dataset_positions = np.zeros(len(self.dataset_folders), dtype=np.int64)
        dataset_index = np.zeros(self.train_split_num_samples, dtype=np.int64)
        dataset_sample_index = np.zeros(self.train_split_num_samples, dtype=np.int64)

        dataset_order = np.repeat(np.arange(len(self.dataset_folders)), samples_per_dataset)
        rng = np.random.RandomState(self.random_seed)
        rng.shuffle(dataset_order)

        for idx, dataset_idx in tqdm(enumerate(dataset_order), desc="Building Nanoset index"):
            dataset_index[idx] = dataset_idx
            dataset_sample_index[idx] = dataset_positions[dataset_idx]
            dataset_positions[
                dataset_idx
            ] += 1  # Read samples sequentially from each datatrove_dataset assuming they're already shuffled

        # Save to cache
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.savez(cache_file, dataset_index=dataset_index, dataset_sample_index=dataset_sample_index)
            logger.info(f"[Nanoset] Saved index to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"[Nanoset] Failed to save cache: {e}")

        return dataset_index, dataset_sample_index

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

    def print_nanoset_info(self):

        log_rank(f"> Total number of samples: {len(self)}", logger=logger, level=logging.INFO, rank=0)
        log_rank(
            f"> Total number of tokens: {len(self) * self.sequence_length}", logger=logger, level=logging.INFO, rank=0
        )

        # Print samples from each dataset + weight
        dataset_sample_count = count_dataset_indexes(self.dataset_index, len(self.dataset_folders))
        for index, sample_count in enumerate(dataset_sample_count):
            log_rank(
                f">   Total number of samples from the {self.dataset_folders[index]} dataset: {sample_count} ({round(normalize(dataset_sample_count).tolist()[index], 2)})",
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

    # TODO: Add 0.5% (the 1.005 factor) so in case the bleding dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network

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
