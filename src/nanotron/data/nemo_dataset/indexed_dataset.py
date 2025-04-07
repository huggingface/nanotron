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

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Most of the code here has been copied from:
#   fairseq/fairseq/data/indexed_dataset.py

# with some modifications:

# Removed IndexedRawTextDataset since it relied on Fairseq dictionary
# other slight modifications to remove fairseq dependencies
# Added document index to index file and made it accessible.
#    An empty sentence no longer separates documents.

import os
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
from typing import Tuple

import numpy as np
import torch

from nanotron import logging
from nanotron.logging import log_rank

logger = logging.get_logger(__name__)
# logging.getLogger('botocore').setLevel(logging.WARNING)


def __best_fitting_dtype(vocab_size=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def make_builder(out_file, vocab_size=None, chunk_size=64, pad_id=0, retrieval_db=False, stride=64):
    return MMapIndexedDatasetBuilder(out_file, dtype=__best_fitting_dtype(vocab_size))


def deallocate_indexed_dataset_memory(indexed_dataset):
    """Deallocate memory of an IndexedDataset."""
    if isinstance(indexed_dataset, MMapIndexedDataset):
        # for MMapIndexedDataset we cannot release any memory of sizes
        indexed_dataset._index._doc_idx = None
    else:
        indexed_dataset.sizes = None
        indexed_dataset.doc_idx = None


def make_indexed_dataset(path: str, skip_warmup: bool = False):
    # now handle bin memap
    if not MMapIndexedDataset.exists(path):
        raise ValueError(
            f"Dataset does not exist: {path}"
            f"Path should be a basename that both .idx and .bin can be appended to get full filenames."
        )
    if MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path, skip_warmup)
    raise ValueError("MMapIndexedDataset doesn't exist")


dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float64, 7: np.double, 8: np.uint16}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path: str) -> str:
    return prefix_path + ".idx"


def data_file_path(prefix_path: str) -> str:
    return prefix_path + ".bin"


def _warmup_mmap_file(path: str):
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, "wb")

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack("<Q", 1))
                    self._file.write(struct.pack("<B", code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes, doc_idx):
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack("<Q", len(sizes)))
                    self._file.write(struct.pack("<Q", len(doc_idx)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order="C"))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path: str, skip_warmup: bool = False):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                log_rank("    warming up index mmap file...", logger=logger, level=logging.INFO, rank=0)
                _warmup_mmap_file(path)

            # self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            log_rank("    reading sizes...", logger=logger, level=logging.INFO, rank=0)
            self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
            log_rank("    reading pointers...", logger=logger, level=logging.INFO, rank=0)
            self._pointers = np.frombuffer(
                self._bin_buffer, dtype=np.int64, count=self._len, offset=offset + self._sizes.nbytes
            )
            log_rank("    reading document index...", logger=logger, level=logging.INFO, rank=0)
            self._doc_idx = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )

        def __del__(self):
            if self._bin_buffer_mmap is not None:
                self._bin_buffer_mmap._mmap.close()
                del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i: int) -> Tuple[int, int]:
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path: str, skip_warmup: bool = False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    # def __setstate__(self, state):
    #     self._do_init(state)

    def _do_init(self, path: str, skip_warmup: bool):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            log_rank("    warming up data mmap file...", logger=logger, level=logging.INFO, rank=0)
            _warmup_mmap_file(data_file_path(self._path))
        log_rank("    creating numpy buffer of mmap...", logger=logger, level=logging.INFO, rank=0)
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode="r", order="C")
        log_rank("    creating memory view of numpy buffer...", logger=logger, level=logging.INFO, rank=0)
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr)
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        logger.debug(
            f"Checking file path: {path}, index_file_path: {index_file_path(path)}, data_file_path: {data_file_path(path)}"
        )
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))

    def deallocate_indexed_dataset_memory(self):
        """Deallocate memory of an IndexedDataset."""
        self._index._doc_idx = None


class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    def add_item(self, tensor):
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def merge_file_(self, another_file):
        # Concatenate index
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        for size in index.sizes:
            self._sizes.append(size)

        # Concatenate data
        with open(data_file_path(another_file), "rb") as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        self._data_file.close()

        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)
