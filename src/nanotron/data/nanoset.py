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
    ) -> None:

        # Checks
        if isinstance(dataset_folders, str):
            warnings.warn("dataset_folders should be of type List[str] but str was provided. Converting to List[str]")
            dataset_folders = [dataset_folders]

        # Init
        self.dataset_folders = dataset_folders
        self.sequence_length = sequence_length
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
                    return_positions=True,
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
        # self.dataset_index, self.dataset_sample_index = self.build_nanoset_index()
        self.dataset_index, self.dataset_sample_index = self.new_build_nanoset_index()

        self.print_nanoset_info()
        self.analyze_batch_diversity()

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
        }

        # Create a deterministic cache key
        cache_key = hashlib.md5(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"index_{cache_key}.npz")

        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                log_rank(f"[Nanoset] Loading index from cache: {cache_file}", logger=logger, level=logging.INFO, rank=0)
                cached_data = np.load(cache_file)
                return cached_data["dataset_index"], cached_data["dataset_sample_index"]
            except Exception as e:
                log_rank(f"[Nanoset] Failed to load cache, rebuilding index: {e}", logger=logger, level=logging.WARNING, rank=0)

        log_rank(f"[Nanoset] Building sequential Nanoset index for {len(self.dataset_folders)} datasets", logger=logger, level=logging.INFO, rank=0)

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
            log_rank(f"[Nanoset] Saved index to cache: {cache_file}", logger=logger, level=logging.INFO, rank=0)
        except Exception as e:
            log_rank(f"[Nanoset] Failed to save cache: {e}", logger=logger, level=logging.WARNING, rank=0)

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

    def analyze_batch_diversity(self, num_random_batches=64, batch_size=8, num_detailed_batches=5):
        """
        Analyzes dataset distribution within random batches, providing global stats and detailed examples.

        Args:
            num_random_batches (int): The number of random batches to sample for global statistics.
            batch_size (int): The batch size to use for analysis.
            num_detailed_batches (int): The number of sampled batches for which to print detailed distribution.
        """
        if not hasattr(self, 'dataset_index') or len(self.dataset_index) == 0:
            log_rank("Dataset index not available or empty. Skipping diversity analysis.", logger=logger, level=logging.INFO, rank=0)
            return

        if batch_size <= 0:
            log_rank("Batch size must be positive. Skipping diversity analysis.", logger=logger, level=logging.WARNING, rank=0)
            return

        # Calculate total number of possible full batches
        total_possible_batches = len(self.dataset_index) // batch_size

        if total_possible_batches == 0:
            log_rank(f"Not enough samples ({len(self.dataset_index)}) for a single batch of size {batch_size}. Skipping diversity analysis.", logger=logger, level=logging.INFO, rank=0)
            return

        if num_random_batches <= 0:
            log_rank("Number of random batches must be positive. Skipping diversity analysis.", logger=logger, level=logging.WARNING, rank=0)
            return
            
        num_detailed_batches = max(0, num_detailed_batches) # Ensure non-negative

        # Adjust num_random_batches if it exceeds the total possible batches
        actual_num_batches_to_analyze = min(num_random_batches, total_possible_batches)
        if actual_num_batches_to_analyze < num_random_batches:
            log_rank(f"Requested {num_random_batches} random batches, but only {total_possible_batches} possible batches exist. Analyzing {actual_num_batches_to_analyze} batches.", logger=logger, level=logging.INFO, rank=0)
            
        # Adjust num_detailed_batches if it exceeds the number we actually analyze
        actual_num_detailed_batches = min(num_detailed_batches, actual_num_batches_to_analyze)

        # Select unique random batch indices
        rng = np.random.RandomState(self.random_seed) # Use the instance seed for reproducibility
        random_batch_indices = rng.choice(total_possible_batches, size=actual_num_batches_to_analyze, replace=False)

        # --- Data Collection Phase ---
        batch_diversities = []
        all_batch_dataset_counts = np.zeros((actual_num_batches_to_analyze, len(self.dataset_folders)), dtype=int)

        for i, batch_idx in enumerate(random_batch_indices):
            batch_indices = self.dataset_index[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            unique_datasets_in_batch = set()
            current_batch_counts = np.zeros(len(self.dataset_folders), dtype=int)

            for dataset_id in batch_indices:
                current_batch_counts[dataset_id] += 1
                unique_datasets_in_batch.add(dataset_id)
            
            batch_diversities.append(len(unique_datasets_in_batch))
            all_batch_dataset_counts[i, :] = current_batch_counts
            
        # --- Global Statistics Calculation ---
        avg_diversity = np.mean(batch_diversities)
        median_diversity = np.median(batch_diversities)
        min_diversity = np.min(batch_diversities)
        max_diversity = np.max(batch_diversities)

        # Calculate per-batch percentages for the sampled batches
        sampled_batch_percentages = all_batch_dataset_counts / batch_size * 100
        avg_sampled_batch_percentages = np.mean(sampled_batch_percentages, axis=0)
        std_sampled_percentages = np.std(sampled_batch_percentages, axis=0)

        # --- Global Statistics Reporting ---
        log_rank("=" * 80, logger=logger, level=logging.INFO, rank=0)
        log_rank(f"RANDOM BATCH DIVERSITY ANALYSIS (Sampled {actual_num_batches_to_analyze} random batches of size {batch_size})", 
                 logger=logger, level=logging.INFO, rank=0)
        log_rank(f"Avg. datasets/batch in sample: {avg_diversity:.2f} (Median: {int(median_diversity)}, Min: {min_diversity}, Max: {max_diversity})", 
                 logger=logger, level=logging.INFO, rank=0)
        log_rank("-" * 80, logger=logger, level=logging.INFO, rank=0)
        log_rank(f"{'Dataset Path':<45} | {'Avg Batch %':<12} | {'Expected %':<10} | {'Std Dev (%)':<12}", 
                 logger=logger, level=logging.INFO, rank=0)
        log_rank(f"{'-'*45}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}", 
                 logger=logger, level=logging.INFO, rank=0)

        for dataset_idx, folder in enumerate(self.dataset_folders):
            avg_pct = avg_sampled_batch_percentages[dataset_idx]
            expected_pct = self.dataset_weights[dataset_idx] * 100
            std_pct = std_sampled_percentages[dataset_idx]
            
            # Format path
            if len(folder) > 45:
                parts = folder.split('/')
                if len(parts) > 3:
                    dataset_path = f"{parts[0]}/.../{'/'.join(parts[-2:])}"
                    if len(dataset_path) > 45:
                        dataset_path = "..." + dataset_path[-42:]
                else:
                    dataset_path = "..." + folder[-(45-3):]
            else:
                dataset_path = folder
            
            log_rank(f"{dataset_path:<45} | {avg_pct:12.2f} | {expected_pct:10.2f} | {std_pct:12.2f}", 
                     logger=logger, level=logging.INFO, rank=0)

        # --- Detailed Batch Reporting --- 
        if actual_num_detailed_batches > 0:
            log_rank("=" * 80, logger=logger, level=logging.INFO, rank=0)
            log_rank(f"Detailed Distribution for {actual_num_detailed_batches} Sampled Batches:", logger=logger, level=logging.INFO, rank=0)
            log_rank("=" * 80, logger=logger, level=logging.INFO, rank=0)
            
            # Select which batches to detail (e.g., the first few sampled)
            detailed_batch_indices_in_sample = range(actual_num_detailed_batches) # Indices within our sample (0 to N-1)

            for i, sample_idx in enumerate(detailed_batch_indices_in_sample):
                original_batch_idx = random_batch_indices[sample_idx]
                dataset_counts = all_batch_dataset_counts[sample_idx, :]
                diversity = batch_diversities[sample_idx]
                
                log_rank(f"--- Detailed Batch {i+1}/{actual_num_detailed_batches} (Original Index: {original_batch_idx}) ---", 
                         logger=logger, level=logging.INFO, rank=0)
                log_rank(f"  Diversity: {diversity}/{len(self.dataset_folders)} datasets present", logger=logger, level=logging.INFO, rank=0)
                log_rank(f"  {'Dataset Path':<45} | {'Count':<7} | {'Actual %':<10} | {'Expected %':<10} | {'Diff %':<8}", 
                         logger=logger, level=logging.INFO, rank=0)
                log_rank(f"  {'-'*45}-+-{'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}", 
                         logger=logger, level=logging.INFO, rank=0)

                # Find which datasets are actually present in this batch
                present_dataset_indices = np.where(dataset_counts > 0)[0]

                for dataset_idx in present_dataset_indices:
                    folder = self.dataset_folders[dataset_idx]
                    count = int(dataset_counts[dataset_idx])
                    actual_pct = count / batch_size * 100
                    expected_pct = self.dataset_weights[dataset_idx] * 100
                    diff_pct = actual_pct - expected_pct

                    # Format path
                    if len(folder) > 45:
                        parts = folder.split('/')
                        if len(parts) > 3:
                            dataset_path = f"{parts[0]}/.../{'/'.join(parts[-2:])}"
                            if len(dataset_path) > 45:
                                dataset_path = "..." + dataset_path[-42:]
                        else:
                            dataset_path = "..." + folder[-(45-3):]
                    else:
                        dataset_path = folder
                    
                    log_rank(f"  {dataset_path:<45} | {count:<7} | {actual_pct:10.2f} | {expected_pct:10.2f} | {diff_pct:+8.2f}", 
                             logger=logger, level=logging.INFO, rank=0)
                
                # Add a small separator between detailed batches
                if i < actual_num_detailed_batches - 1:
                     log_rank("-" * 80, logger=logger, level=logging.INFO, rank=0)

        log_rank("=" * 80, logger=logger, level=logging.INFO, rank=0)
        log_rank("End of Random Batch Diversity Analysis", logger=logger, level=logging.INFO, rank=0)
        log_rank("=" * 80, logger=logger, level=logging.INFO, rank=0)


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
