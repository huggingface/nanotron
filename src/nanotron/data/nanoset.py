import os
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from datatrove.utils.dataset import DatatroveFolderDataset
from numba import jit

from nanotron import logging
from nanotron.data.utils import count_dataset_indexes, normalize
from nanotron.logging import log_rank

logger = logging.get_logger(__name__)


class Nanoset(torch.utils.data.Dataset):
    """
    The Nanoset dataset with support for sequential access optimization.

    Args:
        dataset_folders (List[str]): List of folders with tokenized datasets
        sequence_length (int): Sequence length of the built samples
        token_size (int): Size of a single token in bytes (2 for vocab sizes < 65535, 4 otherwise)
        train_split_num_samples (int): Number of samples the dataset needs (training steps * global batch size)
        dataset_weights (Union[List[float], None]): Optional list with weights for weighted datasets
                                                    If None, weights are based on dataset sizes
        random_seed (int): Seed for reproducibility
        sequential_mode (bool): Whether to use sequential access optimization
        chunk_size (int): Size of sequential chunks when using sequential_mode
    """

    def __init__(
        self,
        dataset_folders: List[str],
        sequence_length: int,
        token_size: int,
        train_split_num_samples: int,
        dataset_weights: Union[List[float], None] = None,
        random_seed: int = 1234,
        sequential_mode: bool = True,
        chunk_size: int = 1000,
    ) -> None:

        # Checks
        if isinstance(dataset_folders, str):
            warnings.warn("dataset_folders should be of type List[str] but str was provided. Converting to List[str]")
            dataset_folders = [dataset_folders]

        # Init
        self.dataset_folders = dataset_folders
        self.sequence_length = sequence_length
        self.token_size = token_size
        self.train_split_num_samples = train_split_num_samples
        self.random_seed = random_seed
        self.sequential_mode = sequential_mode
        self.chunk_size = chunk_size
        
        # Initialize datasets with shuffle=False for sequential access
        self.datatrove_datasets = []
        for dataset_folder in self.dataset_folders:
            self.datatrove_datasets.append(
                DatatroveFolderDataset(
                    folder_path=dataset_folder,
                    filename_pattern=os.path.join(dataset_folder, "*.ds"),
                    seq_len=sequence_length,
                    recursive=False,
                    token_size=self.token_size,
                    shuffle=False,  # No shuffling for sequential access
                    return_positions=True,
                )
            )

        # Build Nanoset Index
        self.dataset_lengths = [len(datatrove_dataset) for datatrove_dataset in self.datatrove_datasets]
        
        # Set dataset weights
        if dataset_weights is None:
            self.dataset_weights = normalize(self.dataset_lengths)
        else:
            self.dataset_weights = normalize(dataset_weights)
        
        assert len(dataset_folders) == len(
            self.dataset_weights
        ), f"Specified {len(self.dataset_weights)} weights but {len(dataset_folders)} datasets were provided."

        # Build dataset index and dataset sample index
        if self.sequential_mode:
            self.dataset_index, self.dataset_sample_index = self.build_sequential_index()
        else:
            self.dataset_index, self.dataset_sample_index = self.build_nanoset_index()

        # Print information about the dataset
        self.print_nanoset_info()
        
        # Analyze batch diversity if using sequential mode
        if self.sequential_mode:
            self.analyze_batch_diversity()

    def __len__(self) -> int:
        """Returns the number of samples in the dataset"""
        return len(self.dataset_index)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns sequence_length + 1 tokens from the dataset

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, torch.LongTensor]: Dictionary containing input_ids and optionally positions
        """
        dataset = self.dataset_index[idx]
        dataset_sample = self.dataset_sample_index[idx]

        return self.datatrove_datasets[dataset][dataset_sample]

    def build_sequential_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build dataset index and sample index optimized for sequential access
        while maintaining diversity across datasets within batches.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: dataset_index and dataset_sample_index arrays
        """
        # Calculate total samples needed and samples per dataset based on weights
        total_samples = self.train_split_num_samples
        samples_per_dataset = [int(np.ceil(w * total_samples)) for w in self.dataset_weights]
        
        # Adjust to ensure we don't exceed total samples
        total_assigned = sum(samples_per_dataset)
        if total_assigned > total_samples:
            largest_idx = np.argmax(samples_per_dataset)
            samples_per_dataset[largest_idx] -= (total_assigned - total_samples)
        
        # Create random number generator
        rng = np.random.RandomState(self.random_seed)
        
        # Build chunks for each dataset
        dataset_chunks = []
        for dataset_idx, num_samples in enumerate(samples_per_dataset):
            if num_samples <= 0:
                continue
                
            # How many samples are available in this dataset
            available_samples = min(self.dataset_lengths[dataset_idx], num_samples)
            
            # Use a smaller chunk size to ensure better interleaving between datasets
            effective_chunk_size = min(self.chunk_size, 128)
            num_full_chunks = available_samples // effective_chunk_size
            final_chunk_size = available_samples % effective_chunk_size
            
            # Generate chunk start positions
            chunk_starts = np.arange(num_full_chunks) * effective_chunk_size
            
            # Shuffle the starting positions of chunks 
            # (samples within chunks remain sequential)
            rng.shuffle(chunk_starts)
            
            # Create chunks for this dataset
            chunks = []
            for start in chunk_starts:
                dataset_indices = np.array([dataset_idx] * effective_chunk_size)
                sample_indices = np.arange(start, start + effective_chunk_size)
                chunks.append((dataset_indices, sample_indices))
            
            # Add final partial chunk if needed
            if final_chunk_size > 0:
                final_start = num_full_chunks * effective_chunk_size
                dataset_indices = np.array([dataset_idx] * final_chunk_size)
                sample_indices = np.arange(final_start, final_start + final_chunk_size)
                chunks.append((dataset_indices, sample_indices))
            
            # Add all chunks to our collection
            dataset_chunks.append(chunks)
        
        # Interleave chunks from different datasets to ensure diversity
        interleaved_chunks = []
        dataset_iterators = [iter(chunks) for chunks in dataset_chunks if chunks]
        active_iterators = list(range(len(dataset_iterators)))
        
        while active_iterators:
            for i in list(active_iterators):  # Make a copy to allow safe removal
                try:
                    chunk = next(dataset_iterators[i])
                    interleaved_chunks.append(chunk)
                except StopIteration:
                    active_iterators.remove(i)
        
        # Shuffle the order of chunks for randomness
        # (sequential access within chunks is preserved)
        rng.shuffle(interleaved_chunks)
        
        # Flatten all chunks into our final arrays
        dataset_index = []
        dataset_sample_index = []
        
        for dataset_indices, sample_indices in interleaved_chunks:
            dataset_index.extend(dataset_indices)
            dataset_sample_index.extend(sample_indices)
        
        # Convert to numpy arrays
        dataset_index = np.array(dataset_index, dtype=np.int64)
        dataset_sample_index = np.array(dataset_sample_index, dtype=np.int64)
        
        # If we need to repeat data to reach target size
        if len(dataset_index) < total_samples:
            repeats = int(np.ceil(total_samples / len(dataset_index)))
            dataset_index = np.tile(dataset_index, repeats)[:total_samples]
            dataset_sample_index = np.tile(dataset_sample_index, repeats)[:total_samples]
        else:
            # Trim if we have too many samples
            dataset_index = dataset_index[:total_samples]
            dataset_sample_index = dataset_sample_index[:total_samples]
        
        # Final small-batch-sized shuffling for batch diversity
        # This maintains sequential chunks but ensures diversity within batches
        batch_size = 8  # Typical batch size per GPU - adjust as needed
        if len(dataset_index) > batch_size:
            # Reshape and shuffle mini-batches
            num_batches = len(dataset_index) // batch_size
            indices = np.arange(num_batches * batch_size).reshape(-1, batch_size)
            rng.shuffle(indices)  # Shuffle the batch order
            indices = indices.flatten()
            
            # Apply the batch shuffling
            dataset_index_batch = dataset_index[:len(indices)]
            dataset_sample_index_batch = dataset_sample_index[:len(indices)]
            dataset_index[:len(indices)] = dataset_index_batch[indices]
            dataset_sample_index[:len(indices)] = dataset_sample_index_batch[indices]
        
        return dataset_index, dataset_sample_index

    def build_nanoset_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Original method - build dataset index and dataset sample index with random access.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: dataset_index and dataset_sample_index arrays
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
        """Print information about the dataset"""
        sequential_status = "sequential" if self.sequential_mode else "random"
        log_rank(f"> Using {sequential_status} access pattern with chunk size {self.chunk_size}", 
                 logger=logger, level=logging.INFO, rank=0)
        log_rank(f"> Total number of samples: {len(self)}", logger=logger, level=logging.INFO, rank=0)
        log_rank(
            f"> Total number of tokens: {len(self) * self.sequence_length}", 
            logger=logger, level=logging.INFO, rank=0
        )

        # Print samples from each dataset + weight
        dataset_sample_count = count_dataset_indexes(self.dataset_index, len(self.dataset_folders))
        for index, sample_count in enumerate(dataset_sample_count):
            log_rank(
                f">   Total number of samples from the {self.dataset_folders[index]} dataset: "
                f"{sample_count} ({round(normalize(dataset_sample_count).tolist()[index], 2)})",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

    def analyze_batch_diversity(self, batch_size=8):
        """
        Analyze and print batch diversity statistics
        
        Args:
            batch_size (int): Batch size to use for analysis
        """
        if not hasattr(self, 'dataset_index'):
            return
        
        # Calculate number of full batches
        num_batches = len(self.dataset_index) // batch_size
        if num_batches == 0:
            return
            
        # Calculate diversity stats
        batch_diversity = []
        for i in range(num_batches):
            batch_indices = self.dataset_index[i * batch_size:(i + 1) * batch_size]
            unique_datasets = len(np.unique(batch_indices))
            batch_diversity.append(unique_datasets)
        
        # Compute statistics
        avg_diversity = np.mean(batch_diversity)
        median_diversity = np.median(batch_diversity)
        min_diversity = np.min(batch_diversity)
        max_diversity = np.max(batch_diversity)
        
        # Log results
        log_rank(f"> Batch diversity statistics (batch_size={batch_size}):", 
                 logger=logger, level=logging.INFO, rank=0)
        log_rank(f">   Average unique datasets per batch: {avg_diversity:.2f}", 
                 logger=logger, level=logging.INFO, rank=0)
        log_rank(f">   Median unique datasets per batch: {median_diversity}", 
                 logger=logger, level=logging.INFO, rank=0)
        log_rank(f">   Min/Max unique datasets per batch: {min_diversity}/{max_diversity}", 
                 logger=logger, level=logging.INFO, rank=0)


@jit(nopython=True, cache=True)
def build_nanoset_index_helper(
    n_samples: int, weights: np.ndarray, dataset_sizes: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given multiple datasets and a weighting array, build samples indexes
    such that it follows those weights
    
    Args:
        n_samples (int): Number of samples to generate
        weights (np.ndarray): Normalized weights for each dataset
        dataset_sizes (List[int]): Size of each dataset
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: dataset_index and dataset_sample_index arrays
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
