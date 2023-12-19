# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT style dataset."""

import os
import time

import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset

from nanotron.config import PretrainNemoArgs
from nanotron.core import distributed as dist
from nanotron.core import logging
from nanotron.core.logging import log_rank
from nanotron.core.process_groups_initializer import DistributedProcessGroups
from nanotron.core.utils import main_rank_first

from .blendable_dataset import BlendableDataset
from .dataset_utils import (
    compile_helper,
    get_datasets_weights_and_num_samples,
    get_train_valid_test_split_,
)
from .indexed_dataset import make_indexed_dataset

logger = logging.get_logger(__name__)

@dataclass
class SubsetSplitLog:
    name: str
    data_prefix: str
    doc_idx_filename: str
    sample_idx_filename: str
    shuffle_idx_filename: str
    tokens_per_epoch: int
    num_epochs: int
    num_samples: int
    seq_length: int


def build_dataset(
    cfg: PretrainNemoArgs,
    data_prefix,
    num_samples,
    seq_length,
    seed,
    skip_warmup,
    name,
    dpg: DistributedProcessGroups,
):
    def _build_dataset(current_data_prefix, current_num_samples):
        indexed_dataset = get_indexed_dataset_(current_data_prefix, skip_warmup)
        total_num_of_documents = indexed_dataset.sizes.shape[0]
        # Print stats about the splits.

        log_rank(" > dataset split:", logger=logger, level=logging.INFO, rank=0)
        log_rank(
            "     Total {} documents is : {} ".format(name, total_num_of_documents),
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        drop_last = True
        if name == "valid":
            drop_last = cfg.validation_drop_last
        dataset = GPTDataset(
            cfg,
            name,
            current_data_prefix,
            np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32),
            indexed_dataset,
            current_num_samples,
            seq_length,
            seed,
            dpg,
            drop_last=drop_last,
        )
        return dataset

    if len(data_prefix) == 1:
        return _build_dataset(data_prefix[0], num_samples)

    else:
        output = get_datasets_weights_and_num_samples(data_prefix, num_samples)
        prefixes, weights, datasets_num_samples = output
        datasets = []
        for i in range(len(prefixes)):
            dataset = _build_dataset(prefixes[i], datasets_num_samples[i])
            datasets.append(dataset)
        return BlendableDataset(datasets, weights, num_samples, dpg=dpg)


def build_train_valid_test_datasets(
    cfg: PretrainNemoArgs,
    data_prefix,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    dpg: DistributedProcessGroups,
    skip_warmup,
):
    if isinstance(data_prefix, dict):
        assert (
            data_prefix.get("train") is not None
            and data_prefix.get("test") is not None
            and data_prefix.get("validation") is not None
        ), f"Data prefix dictionary should have train, test and validation keys.  data_prefix currently has only {data_prefix.keys()}"
        if cfg.splits_string is not None:
            log_rank(
                cfg.splits_string + " ignored since data prefix is of type dictionary.",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
        train_ds = build_dataset(
            cfg,
            data_prefix["train"],
            int(train_valid_test_num_samples[0]),
            seq_length,
            seed,
            skip_warmup,
            "train",
            dpg,
        )
        validation_ds = build_dataset(
            cfg,
            data_prefix["validation"],
            int(train_valid_test_num_samples[1]),
            seq_length,
            seed,
            skip_warmup,
            "valid",
            dpg,
        )
        test_ds = build_dataset(
            cfg,
            data_prefix["test"],
            int(train_valid_test_num_samples[2]),
            seq_length,
            seed,
            skip_warmup,
            "test",
            dpg,
        )
        return train_ds, validation_ds, test_ds

    else:
        # No data
        if len(data_prefix) == 0:
            return (None, None, None)
        # Single dataset.
        if len(data_prefix) == 1:
            return _build_train_valid_test_datasets(
                cfg,
                data_prefix[0],
                splits_string,
                train_valid_test_num_samples,
                seq_length,
                seed,
                dpg,
                skip_warmup,
            )

        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix, train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        train_datasets = []
        valid_datasets = []
        test_datasets = []
        for i in range(len(prefixes)):
            train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
                cfg,
                prefixes[i],
                splits_string,
                datasets_train_valid_test_num_samples[i],
                seq_length,
                seed,
                dpg,
                skip_warmup,
            )
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)

        train_n, valid_n, test_n = map(sum, zip(*datasets_train_valid_test_num_samples))

        # Blend.
        blending_train_dataset = None
        if train_datasets:
            blending_train_dataset = BlendableDataset(train_datasets, weights, train_n, dpg)
        blending_valid_dataset = None
        if valid_datasets:
            blending_valid_dataset = BlendableDataset(valid_datasets, weights, valid_n, dpg)
        blending_test_dataset = None
        if test_datasets:
            blending_test_dataset = BlendableDataset(test_datasets, weights, test_n, dpg)

        return (blending_train_dataset, blending_valid_dataset, blending_test_dataset)


def _build_train_valid_test_datasets(
    cfg: PretrainNemoArgs,
    data_prefix,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    dpg: DistributedProcessGroups,
    skip_warmup,
):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix, skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    log_rank(" > dataset split:", logger=logger, level=logging.INFO, rank=0)

    def print_split_stats(name, index):
        log_rank("    {}:".format(name), logger=logger, level=logging.INFO, rank=0)
        log_rank(
            "     document indices in [{}, {}) total of {} "
            "documents".format(splits[index], splits[index + 1], splits[index + 1] - splits[index]),
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

    print_split_stats("train", 0)
    print_split_stats("validation", 1)
    print_split_stats("test", 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1], step=1, dtype=np.int32)
            drop_last = True
            if name == "valid":
                drop_last = cfg.validation_drop_last
            dataset = GPTDataset(
                cfg,
                name,
                data_prefix,
                documents,
                indexed_dataset,
                train_valid_test_num_samples[index],
                seq_length,
                seed,
                dpg,
                drop_last=drop_last,
            )
        return dataset

    train_dataset = build_dataset(0, "train")
    valid_dataset = build_dataset(1, "valid")
    test_dataset = build_dataset(2, "test")

    return (train_dataset, valid_dataset, test_dataset)


def get_indexed_dataset_(data_prefix, skip_warmup):
    """Build indexed dataset."""
    log_rank(" > building dataset index ...", logger=logger, level=logging.INFO, rank=0)

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix, skip_warmup)
    log_rank(
        " > finished creating indexed dataset in {:4f} " "seconds".format(time.time() - start_time),
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    log_rank(
        "    number of documents: {}".format(indexed_dataset.sizes.shape[0]), logger=logger, level=logging.INFO, rank=0
    )

    return indexed_dataset


class GPTDataset(Dataset):
    def __init__(
        self,
        cfg: PretrainNemoArgs,
        name: str,
        data_prefix: str,
        documents: np.array,
        indexed_dataset,
        num_samples: int,
        seq_length: int,
        seed: int,
        dpg: DistributedProcessGroups,
        drop_last=True,
    ):
        super().__init__()
        self.name = name
        self.indexed_dataset = indexed_dataset
        self.drop_last = drop_last
        self.seq_length = seq_length

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        self.eod_mask_loss = cfg.eod_mask_loss
        self.no_seqlen_plus_one_input_tokens = cfg.no_seqlen_plus_one_input_tokens
        self.add_extra_token = 1
        if self.no_seqlen_plus_one_input_tokens:
            self.add_extra_token = 0

        # save index mappings to a configurable dir
        self.index_mapping_dir = cfg.index_mapping_dir

        # create index_mapping_dir on rank 0
        if dist.is_available() and dist.is_initialized():
            with main_rank_first(dpg.world_pg):
                if self.index_mapping_dir is not None and not os.path.isdir(self.index_mapping_dir):
                    os.makedirs(self.index_mapping_dir)

        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
            self.name,
            data_prefix,
            documents,
            self.indexed_dataset.sizes,
            num_samples,
            seq_length,
            seed,
            dpg,
            index_mapping_dir=self.index_mapping_dir,
            drop_last=drop_last,
            add_extra_token=self.add_extra_token,
        )
        self.indexed_dataset.deallocate_indexed_dataset_memory()

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def _get_text(self, idx: int) -> np.ndarray:

        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f, offset_f = self.sample_idx[idx]
        doc_index_l, offset_l = self.sample_idx[idx + 1]
        # offset_f = self.sample_idx[idx][1]
        # offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(
                self.doc_idx[doc_index_f], offset=offset_f, length=offset_l - offset_f + self.add_extra_token
            )
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f], offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            sample_list.append(
                self.indexed_dataset.get(self.doc_idx[doc_index_l], length=offset_l + self.add_extra_token)
            )
            sample = np.concatenate(sample_list)
        if len(sample) != (self.seq_length + self.add_extra_token):
            log_rank(
                f" > WARNING: Got sample of length: {len(sample)} for sequence length={self.seq_length+self.add_extra_token}, padding the sample to match sequence length",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )

            sample = np.array(sample, dtype=np.int64)
            sample = np.pad(
                sample, (0, self.seq_length + self.add_extra_token - len(sample)), mode="constant", constant_values=-1
            )
        return sample.astype(np.int64)

    def __getitem__(self, idx):
        text = self._get_text(idx)
        return {"input_ids": text}


def _build_index_mappings(
    name,
    data_prefix,
    documents,
    sizes,
    num_samples,
    seq_length,
    seed,
    dpg: DistributedProcessGroups,
    index_mapping_dir: str = None,
    drop_last: bool = True,
    add_extra_token: int = 1,
):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples, add_extra_token)
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    if index_mapping_dir is not None:
        _filename = os.path.join(index_mapping_dir, os.path.basename(data_prefix))
    else:
        _filename = data_prefix
    _filename += "_{}_indexmap".format(name)
    _filename += "_{}ns".format(num_samples)
    _filename += "_{}sl".format(seq_length)
    _filename += "_{}s".format(seed)
    doc_idx_filename = _filename + "_doc_idx.npy"
    sample_idx_filename = _filename + "_sample_idx.npy"
    shuffle_idx_filename = _filename + "_shuffle_idx.npy"

    # Build the indexed mapping if not exist.
    with main_rank_first(dpg.world_pg):
        if (
            (not os.path.isfile(doc_idx_filename))
            or (not os.path.isfile(sample_idx_filename))
            or (not os.path.isfile(shuffle_idx_filename))
        ):

            log_rank(
                " > WARNING: could not find index map files, building " "the indices on rank 0 ...",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

            # For the last epoch, decide whether include the entire epoch
            # in the global shuffle or not.

            # If we need only one epoch, then separating last epoch  does
            # not mean anything.
            if num_epochs == 1:
                separate_last_epoch = False
                log_rank(
                    " > only one epoch required, setting " "separate_last_epoch to False",
                    logger=logger,
                    level=logging.INFO,
                    rank=0,
                )
            else:
                # Get the number of samples for the last epoch
                num_samples_from_epochs_minus_one = (
                    (num_epochs - 1) * tokens_per_epoch - add_extra_token
                ) // seq_length
                last_epoch_num_samples = num_samples - num_samples_from_epochs_minus_one
                assert last_epoch_num_samples >= 0, "last epoch number of samples should be non-negative."
                num_samples_per_epoch = (tokens_per_epoch - add_extra_token) // seq_length
                # For very small datasets, `last_epoch_num_samples` can be equal to
                # (num_samples_per_epoch + 1).
                # TODO: check that this is not problematic indeed
                #  https://github.com/bigcode-project/Megatron-LM/commit/3a6286ba11181899cccfb11d2e508eca9fd15bea
                assert last_epoch_num_samples <= (
                    num_samples_per_epoch + 1
                ), "last epoch number of samples exceeded max value."
                # If we have less than 80% of the samples for the last epoch,
                # seperate out the epoch and treat it differently.
                # Note: the 80% number is just based on common sense and can
                # be adjusted if needed.
                separate_last_epoch = last_epoch_num_samples < int(0.80 * num_samples_per_epoch)
                if separate_last_epoch:
                    string = (
                        " > last epoch number of samples ({}) is smaller "
                        "than 80% of number of samples per epoch ({}), "
                        "setting separate_last_epoch to True"
                    )
                else:
                    string = (
                        " > last epoch number of samples ({}) is larger "
                        "than 80% of number of samples per epoch ({}), "
                        "setting separate_last_epoch to False"
                    )
                log_rank(
                    string.format(last_epoch_num_samples, num_samples_per_epoch),
                    logger=logger,
                    level=logging.INFO,
                    rank=0,
                )

            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            log_rank(
                " > elasped time to build and save doc-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time),
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            # First compile and then import.
            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32
            try:
                compile_helper()

                from . import helpers
            except ImportError:
                raise ImportError(
                    "Could not compile megatron dataset C++ helper functions and therefore cannot import helpers python file."
                )

            sample_idx = helpers.build_sample_idx(
                sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch, drop_last, add_extra_token
            )
            # sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
            #                              num_epochs, tokens_per_epoch, drop_last, add_extra_token)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            log_rank(
                " > elasped time to build and save sample-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time),
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1
            shuffle_idx = _build_shuffle_idx(num_samples_, sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            log_rank(
                " > elasped time to build and save shuffle-idx mapping"
                " (seconds): {:4f}".format(time.time() - start_time),
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
    # counts = torch.cuda.LongTensor([1])
    # dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=dpg.dp_pg)
    # dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=dpg.pp_pg)
    # assert counts[0].item() == (
    #     dist.get_world_size()
    #     // dist.get_world_size(group=dpg.tp_pg)
    # )

    # Load mappings.
    start_time = time.time()
    log_rank(" > loading doc-idx mapping from {}".format(doc_idx_filename), logger=logger, level=logging.INFO, rank=0)
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode="r")
    log_rank(
        " > loading sample-idx mapping from {}".format(sample_idx_filename), logger=logger, level=logging.INFO, rank=0
    )
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode="r")
    log_rank(
        " > loading shuffle-idx mapping from {}".format(shuffle_idx_filename),
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode="r")
    log_rank(
        "    loaded indexed file in {:3.3f} seconds".format(time.time() - start_time),
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    log_rank("    total number of samples: {}".format(sample_idx.shape[0]), logger=logger, level=logging.INFO, rank=0)
    log_rank("    total number of epochs: {}".format(num_epochs), logger=logger, level=logging.INFO, rank=0)

    return doc_idx, sample_idx, shuffle_idx


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples, add_extra_token=1):
    """Based on number of samples and sequence lenght, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - add_extra_token) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0 : len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs - 1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))


def _build_sample_idx(sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch, drop_last=True, add_extra_token=1):
    """Sample index mapping is a 2D array with sizes
    [number-of-samples + 1, 2] where [..., 0] contains
    the index into `doc_idx` and [..., 1] is the
    starting offset in that document."""

    # Total number of samples. For -1 see comments in `_num_epochs`.
    if not drop_last:
        num_samples = -(-(num_epochs * tokens_per_epoch - add_extra_token) // seq_length)
    else:
        num_samples = (num_epochs * tokens_per_epoch - add_extra_token) // seq_length
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)

    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Begining offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + add_extra_token
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += remaining_seq_length + doc_length - add_extra_token
                remaining_seq_length = 0
            else:
                # Otherwise, start from the begining of the next document.
                if doc_idx_index == (len(doc_idx) - 1):
                    assert (
                        sample_index == num_samples
                    ), f"sample_index={sample_index} and num_samples={num_samples} should be the same"
                    doc_offset = sizes[doc_idx[doc_idx_index]] - add_extra_token
                    break
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx


def _build_shuffle_idx(num_samples, total_size, np_rng):
    """Build the range [0, size) and shuffle."""
    logger.info(
        " > building shuffle index with split [0, {}) and [{}, {}) "
        "...".format(num_samples, num_samples, total_size),
    )

    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))
