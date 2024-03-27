import hashlib
import json
import os
import time
from collections import OrderedDict
from typing import Dict, Tuple, Union

import numpy
import torch

from nanotron import logging
from nanotron.data.indexed_dataset import MMapIndexedDataset
from nanotron.data.nanoset_configs import NanosetConfig
from nanotron.data.utils import Split
from nanotron.logging import log_rank

logger = logging.get_logger(__name__)


class Nanoset(torch.utils.data.Dataset):
    """
    The base Nanoset dataset

    Args:
        indexed_dataset (MMapIndexedDataset): The MMapIndexedDataset around which to build the
        Nanoset

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (int): Number of samples that we will consume from the dataset. If it is None, we will consume all the samples only 1 time (valid and test splits). For the train split, we will introduce train steps * global batch size and compute the number of epochs based on the length of the dataset.

        index_split (Split): The indexed_indices Split (train, valid, test)

        config (NanosetConfig): The Nanoset-specific container for all config sourced parameters
    """

    def __init__(
        self,
        indexed_dataset: MMapIndexedDataset,
        indexed_indices: numpy.ndarray,
        num_samples: Union[int, None],
        index_split: Split,
        config: NanosetConfig,
    ) -> None:

        assert indexed_indices.size > 0

        self.indexed_dataset = indexed_dataset
        self.indexed_indices = indexed_indices
        self.num_samples = num_samples
        self.index_split = index_split
        self.config = config

        # Create unique identifier

        self.unique_identifiers = OrderedDict()
        self.unique_identifiers["class"] = type(self).__name__
        self.unique_identifiers["path_prefix"] = self.indexed_dataset.path_prefix
        self.unique_identifiers["index_split"] = self.index_split.name
        self.unique_identifiers["split"] = self.config.split
        self.unique_identifiers["random_seed"] = self.config.random_seed
        self.unique_identifiers["sequence_length"] = self.config.sequence_length

        self.unique_description = json.dumps(self.unique_identifiers, indent=4)
        self.unique_description_hash = hashlib.md5(self.unique_description.encode("utf-8")).hexdigest()

        # Load or build/cache the document, sample, and shuffle indices

        (
            self.document_index,
            self.sample_index,
            self.shuffle_index,
        ) = self.build_document_sample_shuffle_indices()

    def __len__(self) -> int:
        """
        Returns:
            int: The length of the dataset
        """
        return self.shuffle_index.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, numpy.ndarray]:
        """Get the text (token ids) for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, numpy.ndarray]: The token ids wrapped in a dictionary
        """

        # Do the shuffle mapping
        idx = self.shuffle_index[idx]

        # Get the beginning and end documents and offsets
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        tokens = []

        # Sample spans a single document
        if doc_index_beg == doc_index_end:

            # Add the entire sample
            tokens.append(
                self.indexed_dataset.get(
                    self.document_index[doc_index_beg],
                    offset=doc_index_beg_offset,
                    length=doc_index_end_offset - doc_index_beg_offset + 1,
                )
            )

        # Sample spans multiple documents
        else:
            for i in range(doc_index_beg, doc_index_end + 1):

                # Add the sample part
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = None if i < doc_index_end else doc_index_end_offset + 1
                tokens.append(self.indexed_dataset.get(self.document_index[i], offset=offset, length=length))

        return {"input_ids": numpy.array(numpy.concatenate(tokens), dtype=numpy.int64)}

    def build_document_sample_shuffle_indices(
        self,
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Build the document index, the sample index, and the shuffle index

        The document index:
            -- 1-D
            -- An ordered array of document ids

        The sample index:
            -- 2-D
            -- The document indices and offsets which mark the start of every sample to generate samples of sequence length length

        The shuffle index:
            -- 1-D
            -- A random permutation of index range of the sample index

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: The document index, the sample index, and the
            shuffle index

        """
        path_to_cache = self.config.path_to_cache
        if path_to_cache is None:
            path_to_cache = os.path.join(self.indexed_dataset.path_prefix, "cache", f"{type(self).__name__}_indices")

        def get_path_to(suffix):
            return os.path.join(path_to_cache, f"{self.unique_description_hash}-{type(self).__name__}-{suffix}")

        path_to_description = get_path_to("description.txt")
        path_to_document_index = get_path_to("document_index.npy")
        path_to_sample_index = get_path_to("sample_index.npy")
        path_to_shuffle_index = get_path_to("shuffle_index.npy")
        cache_hit = all(
            map(
                os.path.isfile,
                [
                    path_to_description,
                    path_to_document_index,
                    path_to_sample_index,
                    path_to_shuffle_index,
                ],
            )
        )

        num_tokens_per_epoch = compute_num_tokens_per_epoch(self.indexed_dataset, self.indexed_indices)

        if not cache_hit and torch.distributed.get_rank() == 0:
            log_rank(
                f"Build and save the {type(self).__name__} {self.index_split.name} indices",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

            numpy_random_state = numpy.random.RandomState(self.config.random_seed)

            os.makedirs(path_to_cache, exist_ok=True)

            # Write the description
            with open(path_to_description, "wt") as writer:
                writer.write(self.unique_description)

            # Build the document index
            log_rank(
                f"\tBuild and save the document index to {os.path.basename(path_to_document_index)}",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
            t_beg = time.time()
            document_index = numpy.copy(self.indexed_indices)
            numpy_random_state.shuffle(document_index)
            numpy.save(path_to_document_index, document_index, allow_pickle=True)
            t_end = time.time()
            log_rank(f"\t> Time elapsed: {t_end - t_beg:4f} seconds", logger=logger, level=logging.DEBUG, rank=0)

            # Build the sample index
            log_rank(
                f"\tBuild and save the sample index to {os.path.basename(path_to_sample_index)}",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
            t_beg = time.time()

            assert document_index.dtype == numpy.int32
            assert self.indexed_dataset.sequence_lengths.dtype == numpy.int32
            sample_index = build_sample_idx(
                sizes=self.indexed_dataset.sequence_lengths,
                doc_idx=document_index,
                seq_length=self.config.sequence_length,
                tokens_per_epoch=num_tokens_per_epoch,
            )
            numpy.save(path_to_sample_index, sample_index, allow_pickle=True)
            t_end = time.time()
            log_rank(f"\t> Time elapsed: {t_end - t_beg:4f} seconds", logger=logger, level=logging.DEBUG, rank=0)

            # Build the shuffle index
            log_rank(
                f"\tBuild and save the shuffle index to {os.path.basename(path_to_shuffle_index)}",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
            t_beg = time.time()
            shuffle_index = numpy.arange(start=0, stop=(sample_index.shape[0] - 1), step=1, dtype=numpy.uint32)
            numpy_random_state.shuffle(shuffle_index)
            if self.num_samples is not None:  # For the train split, concatenate shuffle Indexes
                n_concatenations = (
                    int(self.num_samples / shuffle_index.shape[0]) + 1
                )  # NOTE: To ensure that we always generate more samples than requested in num_samples
                shuffle_index = numpy.concatenate([shuffle_index for _ in range(n_concatenations)])
            numpy.save(
                path_to_shuffle_index,
                shuffle_index,
                allow_pickle=True,
            )
            t_end = time.time()
            log_rank(f"\t> Time elapsed: {t_end - t_beg:4f} seconds", logger=logger, level=logging.DEBUG, rank=0)

        log_rank(
            f"Load the {type(self).__name__} {self.index_split.name} indices",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        log_rank(
            f"\tLoad the document index from {os.path.basename(path_to_document_index)}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        t_beg = time.time()
        document_index = numpy.load(path_to_document_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_rank(f"\t> time elapsed: {t_end - t_beg:4f} seconds", logger=logger, level=logging.DEBUG, rank=0)

        log_rank(
            f"\tLoad the sample index from {os.path.basename(path_to_sample_index)}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        t_beg = time.time()
        sample_index = numpy.load(path_to_sample_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_rank(f"\t> Time elapsed: {t_end - t_beg:4f} seconds", logger=logger, level=logging.DEBUG, rank=0)

        log_rank(
            f"\tLoad the shuffle index from {os.path.basename(path_to_shuffle_index)}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        t_beg = time.time()
        shuffle_index = numpy.load(path_to_shuffle_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_rank(f"\t> Time elapsed: {t_end - t_beg:4f} seconds", logger=logger, level=logging.DEBUG, rank=0)

        log_rank(f"> Total number of samples: {sample_index.shape[0] - 1}", logger=logger, level=logging.INFO, rank=0)

        if (
            self.num_samples is not None
        ):  # Compute number of epochs we will iterate over this Nanoset. Just for training
            num_epochs = round(self.num_samples / (sample_index.shape[0] - 1), 2)
            log_rank(f"> Total number of epochs: {num_epochs}", logger=logger, level=logging.INFO, rank=0)

        return document_index, sample_index, shuffle_index


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


def compute_num_tokens_per_epoch(indexed_dataset: MMapIndexedDataset, indices: numpy.ndarray) -> int:
    """Compute the number of tokens in a single epoch

    Args:
        indexed_dataset (MMapIndexedDataset): The underlying MMapIndexedDataset

        indices (numpy.ndarray): The subset of indices into the underlying MMapIndexedDataset

    Returns:
        int: The number of tokens in a single epoch
    """
    return numpy.sum(indexed_dataset.sequence_lengths[indices])
