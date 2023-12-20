import re
import time
from bisect import bisect
from dataclasses import dataclass
from re import Pattern
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from nanotron.config import TokenizedBytesDatasetArgs, TokenizedBytesDatasetFileArgs, TokenizedBytesDatasetFolderArgs
from nanotron.core import distributed as dist
from nanotron import logging
from nanotron.core.dataclass import DistributedProcessGroups
from nanotron.logging import log_rank
from nanotron.core.utils import human_format
from nanotron.dataloaders.data_samplers import MegatronPretrainingSampler
from nanotron.dataloaders.dataloader import DataCollatorForCLM, EmptyInfiniteDataset, get_dataloader_worker_init
from nanotron.dataloaders.nemo_dataset import BlendableDataset
from nanotron.dataloaders.nemo_dataset.dataset_utils import compile_helper
from nanotron.dataloaders.s3_utils import BOTO3_AVAILABLE, _get_s3_file_list, _get_s3_object, _stream_s3_file

try:
    tb_logger_available = True
except ImportError:
    tb_logger_available = False

logger = logging.get_logger(__name__)


@dataclass
class S3FileDatasetLog:
    dataset_type: str
    s3_path: str
    seq_len: int
    dtype: str
    skip_in_stream: bool
    num_samples: int
    num_tokens: int
    human_format_num_tokens: str
    num_epochs: Optional[int]


@dataclass
class S3FolderDatasetLog:
    dataset_type: str
    s3_folder: str
    s3_filename_pattern: str
    recursive: bool
    seq_len: int
    dtype: str
    skip_in_stream: bool
    num_samples: int
    num_tokens: int
    human_format_num_tokens: str
    num_epochs: Optional[int]


@dataclass
class TrainDataLog:
    global_batch_size: int
    sequence_length: int
    total_training_tokens: int
    human_total_train_tokens: str
    train_num_samples: int
    eval_num_samples: int
    test_num_samples: int
    train_subset: Union[S3FileDatasetLog, S3FolderDatasetLog]
    eval_subset: Union[S3FileDatasetLog, S3FolderDatasetLog]
    test_subset: Union[S3FileDatasetLog, S3FolderDatasetLog]


class S3TokenizedBytesFileDataset(Dataset):
    def __init__(
        self,
        s3_path: str,
        seq_len: int,
        dtype: np.dtype = np.uint16,
        skip_in_stream: bool = True,
        num_samples: Optional[int] = None,
        max_tokens: Optional[int] = None,
        skip_tokens: Optional[int] = None,
    ):
        """Streaming dataset for a single S3 file
        We loop on the dataset if asking for an index larger than the dataset size

        Args:
            s3_path (str): S3 path to file
            seq_len (int): sequence length
            dtype (np.dtype, optional): numpy dtype. Defaults to np.uint16.
            skip_in_stream (bool, optional): skip ahead in stream. Defaults to True.
            num_samples (Optional[int], optional): number of samples. Defaults to None. Only indicative for the number of epoch
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required: pip install boto3")
        self.s3_path = s3_path
        self.seq_len = seq_len
        self.dtype = dtype
        self.dtype_size = np.dtype(dtype).itemsize
        self.skip_in_stream = skip_in_stream
        self.skip_tokens = skip_tokens or 0
        # total number of full contexts in this file
        s3_tokens = _get_s3_object(s3_path).content_length // self.dtype_size - self.skip_tokens
        self._len = (min(max_tokens, s3_tokens) if max_tokens else s3_tokens) // (seq_len + 1)
        self._stream = None
        self._last_item_requested = None

        self.subset_log = S3FileDatasetLog(
            dataset_type=self.__class__.__name__,
            s3_path=s3_path,
            seq_len=seq_len,
            dtype=np.dtype(dtype).name,
            skip_in_stream=skip_in_stream,
            num_samples=self._len,
            num_tokens=self._len * (seq_len + 1),
            human_format_num_tokens=human_format(self._len * (seq_len + 1)),
            num_epochs=num_samples // self._len if num_samples else 0,
        )

    def _get_new_stream(self, index):
        """Get a new stream starting from index (in sequence length contexts)

        Note: we pick chunks of seq_len + 1 to account for the label/target of the last tokens
              This means that we drop one token of training per sample.
        """
        chunk_size = self.dtype_size * (self.seq_len + 1)
        index += self.skip_tokens
        for chunk in _stream_s3_file(self.s3_path, chunk_size, index):
            assert len(chunk) == self.dtype_size * (self.seq_len + 1), (
                f"Expected {chunk_size} bytes from S3 but got " f"{len(chunk)}"
            )
            # careful with type conversions here
            yield torch.as_tensor(np.frombuffer(chunk, self.dtype).astype(np.int64), dtype=torch.long)

    def __getitem__(self, item):
        # We loop on the dataset if asking for an index larger than the dataset size
        epoch_item = item % len(self)
        # if item >= len(self):
        #     raise IndexError(f"Index {item} requested for file {self.s3_path} but it only has size {len(self)}")
        # skip ahead without creating a new stream
        if self._stream and epoch_item > self._last_item_requested and self.skip_in_stream:
            while self._last_item_requested < epoch_item - 1:
                self._last_item_requested += 1
                self._get_next_from_stream()  # consume stream
        # new stream starting from "epoch_item"
        elif not self._stream or epoch_item != self._last_item_requested + 1:
            self._stream = self._get_new_stream(epoch_item)

        self._last_item_requested = epoch_item

        return {"input_ids": self._get_next_from_stream()}

    def _get_next_from_stream(self):
        sleep_time = 0.01
        while True:
            try:
                return next(self._stream)
            except Exception as e:
                if sleep_time >= 2.0:
                    logger.error("Giving up on re-establishing boto3 stream.")
                    raise e
                time.sleep(sleep_time)
                self._stream = self._get_new_stream(self._last_item_requested)
                sleep_time *= 2

    def __len__(self):
        return self._len


class S3TokenizedBytesFolderDataset(Dataset):
    def __init__(
        self,
        s3_folder: str,
        seq_len: int,
        s3_filename_pattern: Union[Pattern, str] = None,
        recursive: bool = True,
        dtype: np.dtype = np.uint16,
        skip_in_stream: bool = True,
        num_samples: Optional[int] = None,
        max_tokens: Optional[int] = None,
        skip_tokens: Optional[int] = None,
    ):
        """Dataset for a folder of S3 files
        We loop on the dataset if asking for an index larger than the dataset size

        Args:
            s3_folder (str): S3 path to folder
            seq_len (int): sequence length
            s3_filename_pattern (Union[Pattern, str], optional): filename pattern. Defaults to None.
            recursive (bool, optional): search recursively. Defaults to True.
            dtype (np.dtype, optional): numpy dtype. Defaults to np.uint16.
            skip_in_stream (bool, optional): skip ahead in stream. Defaults to True.
            num_samples (Optional[int], optional): number of samples. Defaults to None. Only indicative for the number of epoch
        """
        self.s3_folder = s3_folder
        if isinstance(s3_filename_pattern, str):
            s3_filename_pattern = re.compile(s3_filename_pattern)
        self.s3_filename_pattern = s3_filename_pattern
        matched_file_paths = _get_s3_file_list(s3_folder, s3_filename_pattern, recursive)
        if not matched_file_paths:
            raise FileNotFoundError(f'No files matching "{s3_filename_pattern}" found in {s3_folder}')

        self.files = []
        remaining_tokens = max_tokens
        remaining_skip_tokens = skip_tokens or 0
        for path in matched_file_paths:
            file_data = S3TokenizedBytesFileDataset(
                path,
                seq_len,
                dtype=dtype,
                skip_in_stream=skip_in_stream,
                max_tokens=remaining_tokens,
                skip_tokens=remaining_skip_tokens,
            )
            if remaining_skip_tokens:
                remaining_skip_tokens -= len(file_data) * (seq_len + 1)
                if remaining_skip_tokens <= 0:
                    remaining_skip_tokens = 0
                elif remaining_skip_tokens > 0:
                    continue  # We skip this file entirely
            self.files.append(file_data)
            if remaining_tokens:
                remaining_tokens -= len(file_data) * (seq_len + 1)
                if remaining_tokens <= 0:
                    break

        self.lens = np.cumsum([0] + [len(f) for f in self.files]).tolist()

        self.current_file = 0

        self.subset_log = S3FolderDatasetLog(
            dataset_type=self.__class__.__name__,
            s3_folder=s3_folder,
            s3_filename_pattern=str(s3_filename_pattern),
            recursive=recursive,
            seq_len=seq_len,
            dtype=np.dtype(dtype).name,
            skip_in_stream=skip_in_stream,
            num_samples=self.lens[-1] if self.lens else 0,
            num_tokens=self.lens[-1] * (seq_len + 1),
            human_format_num_tokens=human_format(self.lens[-1] * (seq_len + 1)),
            num_epochs=num_samples // self.lens[-1] if num_samples and self.lens else 0,
        )

    def __getitem__(self, item):
        epoch_item = item % len(self)
        # if item >= len(self):
        #     raise IndexError(
        #         f"Index {item} requested for dataset {self.s3_folder} (pattern: {self.s3_filename_pattern}) "
        #         f"but it only has size {len(self)}"
        #     )
        # check if we are in the same file as before
        if not (self.lens[self.current_file] <= epoch_item < self.lens[self.current_file + 1]):
            # figure out current file
            self.current_file = bisect(self.lens, epoch_item) - 1
        # subtract file starting offset
        return self.files[self.current_file][epoch_item - self.lens[self.current_file]]

    def __len__(self):
        return self.lens[-1] if self.lens else 0


def build_dataset(
    dataset_args: Union[TokenizedBytesDatasetFileArgs, TokenizedBytesDatasetFolderArgs],
    seq_length,
    skip_in_stream: bool = True,
    num_samples: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> "Dataset":
    """Build one S3 dataset from a file or a folder on S3

    Args:
        dataset_args (Union[TokenizedBytesDatasetFileArgs, TokenizedBytesDatasetFolderArgs]): dataset config
        seq_length ([type]): sequence length
        skip_in_stream (bool, optional): skip ahead in stream. Defaults to True.
    """
    if isinstance(dataset_args, TokenizedBytesDatasetFileArgs):
        return S3TokenizedBytesFileDataset(
            dataset_args.filepath,
            seq_length,
            max_tokens=max_tokens,
            skip_in_stream=skip_in_stream,
            num_samples=num_samples,
            skip_tokens=dataset_args.skip_tokens,
        )
    elif isinstance(dataset_args, TokenizedBytesDatasetFolderArgs):
        return S3TokenizedBytesFolderDataset(
            dataset_args.folder,
            seq_length,
            dataset_args.filename_pattern,
            skip_in_stream=skip_in_stream,
            max_tokens=max_tokens,
            num_samples=num_samples,
            skip_tokens=dataset_args.skip_tokens,
        )
    else:
        raise ValueError(
            "Each dataset must be of type TokenizedBytesDatasetFileArgs or TokenizedBytesDatasetFolderArgs"
        )


def get_s3_datasets(
    config: TokenizedBytesDatasetArgs,
    sequence_length: int,
    global_batch_size: int,
    train_steps: int,
    dpg: DistributedProcessGroups,
) -> Tuple[DataLoader, TrainDataLog]:
    """Build S3 datasets

    Args:
        config (TokenizedBytesDatasetArgs): dataset config
        sequence_length (int): sequence length
        global_batch_size (int): global batch size
        train_steps (int): number of training steps
        dpg (DistributedProcessGroups): distributed process groups
    """
    log_rank("Building S3 Streamable datasets.", logger=logger, level=logging.INFO, rank=0)
    dataset_max_tokens = config.dataset_max_tokens
    if dataset_max_tokens is None:
        dataset_max_tokens = [None] * len(config.datasets)
    train_num_samples = train_steps * global_batch_size

    datasets = [
        build_dataset(
            dataset, sequence_length, config.skip_in_stream, max_tokens=max_tokens, num_samples=train_num_samples
        )
        for dataset, max_tokens in zip(config.datasets, dataset_max_tokens)
    ]

    if len(datasets) == 1:
        outputs_dataset = datasets[0]
    else:
        if dist.get_rank(dpg.world_pg) == 0:
            try:
                compile_helper()
            except ImportError:
                raise ImportError(
                    "Could not compile megatron dataset C++ helper functions and therefore cannot import helpers python file."
                )
        dist.barrier(dpg.world_pg)
        weights = config.dataset_weights
        if not weights:
            weights = [1] * len(datasets)

        outputs_dataset = BlendableDataset(datasets, weights, train_num_samples, dpg=dpg)

    train_data_log = TrainDataLog(
        train_num_samples=train_num_samples,
        eval_num_samples=None,
        test_num_samples=None,
        global_batch_size=global_batch_size,
        sequence_length=sequence_length,
        total_training_tokens=train_num_samples * sequence_length,
        human_total_train_tokens=human_format(train_num_samples * sequence_length),
        train_subset=outputs_dataset.subset_log,
        eval_subset=None,
        test_subset=None,
    )
    return outputs_dataset, train_data_log


def get_s3_dataloader(
    dataset: Union[S3TokenizedBytesFolderDataset, S3TokenizedBytesFileDataset, BlendableDataset, Dataset],
    sequence_length: int,
    micro_batch_size: int,
    global_batch_size: int,
    cfg: TokenizedBytesDatasetArgs,
    num_workers: int,
    consumed_samples: int,
    num_samples: int,
    dpg: DistributedProcessGroups,
    input_pp_rank: int,
    output_pp_rank: int,
    dataloader_drop_last: bool = True,
    dataloader_pin_memory: bool = True,
) -> DataLoader:
    # Only some rank require to run the dataloader.
    if dist.get_rank(dpg.pp_pg) not in [
        input_pp_rank,
        output_pp_rank,
    ]:
        dataset = EmptyInfiniteDataset(length=len(dataset))

    log_rank(
        f"Building dataloader with consumed samples: {consumed_samples}", logger=logger, level=logging.INFO, rank=0
    )
    # Megatron sampler
    batch_sampler = MegatronPretrainingSampler(
        total_samples=num_samples,
        consumed_samples=consumed_samples,
        micro_batch_size=micro_batch_size,
        data_parallel_rank=dist.get_rank(dpg.dp_pg),
        data_parallel_size=dpg.dp_pg.size(),
        drop_last=dataloader_drop_last,
        global_batch_size=global_batch_size,
        pad_samples_to_global_batch_size=cfg.pad_samples_to_global_batch_size,
    )

    # We use the data collator to put the tensors on the right pipeline parallelism rank
    data_collator = DataCollatorForCLM(
        sequence_length=sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        dpg=dpg,
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=dataloader_pin_memory,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dist.get_rank(dpg.dp_pg)),
    )
