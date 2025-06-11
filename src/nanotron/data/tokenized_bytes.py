import os
import re
import time
from bisect import bisect
from dataclasses import dataclass
from re import Pattern
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from datatrove.utils.dataset import DatatroveFolderDataset
from torch.utils.data import DataLoader, Dataset

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import NanosetDatasetsArgs
from nanotron.data import DataCollatorForCLM, DataCollatorForCLMWithPositionIds, EmptyInfiniteDataset
from nanotron.data.dataloader import get_dataloader_worker_init
from nanotron.data.nemo_dataset import BlendableDataset
from nanotron.data.nemo_dataset.dataset_utils import compile_helper
from nanotron.data.s3_utils import BOTO3_AVAILABLE, _get_s3_file_list, _get_s3_object, _stream_file
from nanotron.data.samplers import MegatronPretrainingSampler
from nanotron.logging import human_format, log_rank
from nanotron.parallel import ParallelContext
from nanotron.utils import main_rank_first

try:
    tb_logger_available = True
except ImportError:
    tb_logger_available = False

logger = logging.get_logger(__name__)


@dataclass
class TBFileDatasetLog:
    dataset_type: str
    file_path: str
    seq_len: int
    dtype: str
    skip_in_stream: bool
    num_samples: int
    num_tokens: int
    human_format_num_tokens: str
    num_epochs: Optional[int]


@dataclass
class TBFolderDatasetLog:
    dataset_type: str
    folder_path: str
    filename_pattern: str
    recursive: bool
    seq_len: int
    dtype: str
    skip_in_stream: bool
    num_samples: int
    num_tokens: int
    human_format_num_tokens: str
    shuffle: Optional[bool]
    seed: Optional[int]
    num_epochs: Optional[int]
    files_order: Optional[list]


@dataclass
class TrainDataLog:
    global_batch_size: int
    sequence_length: int
    total_training_tokens: int
    human_total_train_tokens: str
    train_num_samples: int
    eval_num_samples: int
    test_num_samples: int
    train_subset: Union[TBFileDatasetLog, TBFolderDatasetLog]
    eval_subset: Union[TBFileDatasetLog, TBFolderDatasetLog]
    test_subset: Union[TBFileDatasetLog, TBFolderDatasetLog]


class TokenizedBytesFileDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        seq_len: int,
        dtype: np.dtype = np.uint16,
        skip_in_stream: bool = True,
        num_samples: Optional[int] = None,
        max_tokens: Optional[int] = None,
        skip_tokens: Optional[int] = None,
    ):
        """Streaming dataset for a single TokenizedByte file
        We loop on the dataset if asking for an index larger than the dataset size

        Args:
            file_path (str): path to file on s3 or locally
            seq_len (int): sequence length
            dtype (np.dtype, optional): numpy dtype. Defaults to np.uint16.
            skip_in_stream (bool, optional): skip ahead in stream. Defaults to True.
            num_samples (Optional[int], optional): number of samples. Defaults to None. Only indicative for the number of epoch
        """
        self.file_path = file_path
        self.seq_len = seq_len
        self.dtype = dtype
        self.dtype_size = np.dtype(dtype).itemsize
        self.skip_in_stream = skip_in_stream
        self.skip_tokens = skip_tokens or 0
        # total number of full contexts in this file
        if file_path.startswith("s3://"):
            if not BOTO3_AVAILABLE:
                raise ImportError("boto3 is required: pip install boto3")
            num_tokens = _get_s3_object(file_path).content_length // self.dtype_size - self.skip_tokens
        else:
            num_tokens = os.path.getsize(file_path) // self.dtype_size - self.skip_tokens
        self._len = (min(max_tokens, num_tokens) if max_tokens else num_tokens) // (seq_len + 1)
        self._stream = None
        self._last_item_requested = None

        self.subset_log = TBFileDatasetLog(
            dataset_type=self.__class__.__name__,
            file_path=file_path,
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
        for chunk in _stream_file(self.file_path, chunk_size, index):
            assert len(chunk) == self.dtype_size * (self.seq_len + 1), (
                f"Expected {chunk_size} bytes from file but got " f"{len(chunk)}"
            )
            # careful with type conversions here
            yield torch.as_tensor(np.frombuffer(chunk, self.dtype).astype(np.int64), dtype=torch.int64)

    def __getitem__(self, item):
        # We loop on the dataset if asking for an index larger than the dataset size
        epoch_item = item % len(self)
        # if item >= len(self):
        #     raise IndexError(f"Index {item} requested for file {self.file_path} but it only has size {len(self)}")
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
                    logger.error("Giving up on re-establishing stream.")
                    raise e

                time.sleep(sleep_time)
                self._stream = self._get_new_stream(self._last_item_requested)
                sleep_time *= 2

    def __len__(self):
        return self._len


class OldTokenizedBytesFolderDataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        seq_len: int,
        filename_pattern: Union[Pattern, str] = None,
        recursive: bool = True,
        dtype: np.dtype = np.uint16,
        skip_in_stream: bool = True,
        num_samples: Optional[int] = None,
        max_tokens: Optional[int] = None,
        skip_tokens: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        """Dataset for a folder of TokenizedBytes files
        We loop on the dataset if asking for an index larger than the dataset size

        Args:
            folder_path (str): path to folder on S3 or locally
            seq_len (int): sequence length
            filename_pattern (Union[Pattern, str], optional): filename pattern. Defaults to None.
            recursive (bool, optional): search recursively. Defaults to True.
            dtype (np.dtype, optional): numpy dtype. Defaults to np.uint16.
            skip_in_stream (bool, optional): skip ahead in stream. Defaults to True.
            num_samples (Optional[int], optional): number of samples. Defaults to None. Only indicative for the number of epoch
            shuffle (bool, optional): shuffle the files in the folder. Defaults to False.
            seed (int, optional): seed for shuffling. Defaults to 42.
        """
        self.folder_path = folder_path
        if isinstance(filename_pattern, str):
            filename_pattern = re.compile(filename_pattern)
        self.filename_pattern = filename_pattern
        if folder_path.startswith("s3://"):
            matched_file_paths = _get_s3_file_list(folder_path, filename_pattern, recursive)
        else:
            matched_file_paths = [
                os.path.join(root, file)
                for root, _, files in os.walk(folder_path)
                for file in files
                if filename_pattern.match(os.path.join(root, file))
            ]
        matched_file_paths = sorted(matched_file_paths)
        if not matched_file_paths:
            raise FileNotFoundError(f'No files matching "{filename_pattern}" found in {folder_path}')

        self.files = []
        remaining_tokens = max_tokens
        remaining_skip_tokens = skip_tokens or 0
        for path in matched_file_paths:
            file_data = TokenizedBytesFileDataset(
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

        log_rank(f"Found {len(self.files)} files.", logger=logger, level=logging.INFO, rank=0)
        if shuffle:
            log_rank("Shuffling...", logger=logger, level=logging.INFO, rank=0)
            rand = np.random.default_rng(seed)
            ordering = rand.permutation(range(len(self.files)))
            self.files = [self.files[i] for i in ordering]

        self.lens = np.cumsum([0] + [len(f) for f in self.files]).tolist()

        self.current_file = 0

        self.subset_log = TBFolderDatasetLog(
            dataset_type=self.__class__.__name__,
            folder_path=folder_path,
            filename_pattern=str(filename_pattern),
            recursive=recursive,
            seq_len=seq_len,
            dtype=np.dtype(dtype).name,
            skip_in_stream=skip_in_stream,
            num_samples=self.lens[-1] if self.lens else 0,
            num_tokens=self.lens[-1] * (seq_len + 1),
            human_format_num_tokens=human_format(self.lens[-1] * (seq_len + 1)),
            shuffle=shuffle,
            seed=seed,
            num_epochs=num_samples // self.lens[-1] if num_samples and self.lens else 0,
            files_order=[str(f.file_path) for f in self.files],
        )

    def __getitem__(self, item):
        epoch_item = item % len(self)
        # if item >= len(self):
        #     raise IndexError(
        #         f"Index {item} requested for dataset {self.folder_path} (pattern: {self.filename_pattern}) "
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


class TokenizedBytesFolderDataset(DatatroveFolderDataset):
    def __init__(
        self,
        folder_path: str,
        seq_len: int,
        filename_pattern: str = None,
        recursive: bool = True,
        token_size: int = 2,
        max_tokens: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
        return_positions: bool = False,
        eos_token_id: int | None = None,
        skip_in_stream: bool = True,
        num_samples: Optional[int] = None,
        folder_read_path: Optional[str] = None,
        force_update_cache: bool = os.environ.get("FORCE_UPDATE_CACHE_S3", 0) == "1",
    ):
        log_rank("Using DatatroveFolderDataset", logger=logger, level=logging.INFO, rank=0)
        if return_positions and not eos_token_id:
            log_rank(
                "Using DatatroveFolderDataset with return_positions=True but no eos_token_id provided. It can be slow...",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )

        # Handle S3 paths specially
        matched_files = None
        file_sizes = None
        if folder_path.startswith("s3://"):
            cache_dir = os.path.expanduser("~/.cache/nanotron/s3_cache")
            os.makedirs(cache_dir, exist_ok=True)

            # Create a unique cache key based on the folder path and pattern
            import hashlib

            cache_key = hashlib.md5(f"{folder_path}:{filename_pattern}:{recursive}".encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"{cache_key}.cache")

            with main_rank_first():
                if dist.get_rank() == 0:
                    # Check if we have a valid cache
                    if os.path.exists(cache_file) and not force_update_cache:
                        try:
                            import pickle

                            with open(cache_file, "rb") as f:
                                cached_data = pickle.load(f)
                                matched_files = cached_data["matched_files"]
                                file_sizes = cached_data["file_sizes"]
                            log_rank(
                                "[TokenizedBytesFolderDataset] Using cached S3 file list",
                                logger=logger,
                                level=logging.INFO,
                                rank=0,
                            )
                        except Exception as e:
                            log_rank(
                                f"[TokenizedBytesFolderDataset] Failed to load cache, fetching from S3: {e}",
                                logger=logger,
                                level=logging.WARNING,
                                rank=0,
                            )

                    # If no cache or cache invalid, fetch from S3
                    if matched_files is None:
                        log_rank(
                            "[TokenizedBytesFolderDataset] Fetching file list from S3...",
                            logger=logger,
                            level=logging.INFO,
                            rank=0,
                        )
                        from datatrove.utils.dataset import url_to_fs

                        fs_folder, stripped_folder_path = url_to_fs(folder_path)
                        matched_files = (
                            fs_folder.find(stripped_folder_path, detail=False, maxdepth=1 if not recursive else None)
                            if not filename_pattern
                            else fs_folder.glob(
                                os.path.join(stripped_folder_path, filename_pattern),
                                maxdepth=1 if not recursive else None,
                            )
                        )
                        matched_files = sorted(matched_files)

                        # Get file sizes
                        file_sizes = {}
                        for path in matched_files:
                            file_path = fs_folder.unstrip_protocol(path)
                            fs, file_path = url_to_fs(file_path)
                            file_sizes[file_path] = fs.size(file_path)

                        # Save to cache
                        try:
                            import pickle

                            with open(cache_file, "wb") as f:
                                pickle.dump({"matched_files": matched_files, "file_sizes": file_sizes}, f)
                            log_rank(
                                "[TokenizedBytesFolderDataset] Saved S3 file list to cache",
                                logger=logger,
                                level=logging.INFO,
                                rank=0,
                            )
                        except Exception as e:
                            log_rank(
                                f"[TokenizedBytesFolderDataset] Failed to save cache: {e}",
                                logger=logger,
                                level=logging.WARNING,
                                rank=0,
                            )
            if dist.get_rank() != 0:
                try:
                    import pickle

                    with open(cache_file, "rb") as f:
                        cached_data = pickle.load(f)
                        matched_files = cached_data["matched_files"]
                        file_sizes = cached_data["file_sizes"]
                except Exception as e:
                    raise RuntimeError(f"Failed to read cache file on rank {dist.get_rank()}: {e}")

        super().__init__(
            folder_path=folder_path,
            seq_len=seq_len,
            filename_pattern=filename_pattern,
            recursive=recursive,
            token_size=token_size,
            max_tokens=max_tokens,
            shuffle=shuffle,
            seed=seed,
            return_positions=return_positions,
            eos_token_id=eos_token_id,
            read_path=folder_read_path,
            matched_files=matched_files,
            file_sizes=file_sizes,
        )

        self.subset_log = TBFolderDatasetLog(
            dataset_type=self.__class__.__name__,
            folder_path=folder_path,
            filename_pattern=str(filename_pattern),
            recursive=recursive,
            seq_len=seq_len,
            dtype=np.dtype(np.uint16 if token_size == 2 else np.uint32 if token_size == 4 else np.uint64).name,
            skip_in_stream=skip_in_stream,
            num_samples=self.lens[-1] if self.lens else 0,
            num_tokens=self.lens[-1] * (seq_len + 1),
            human_format_num_tokens=human_format(self.lens[-1] * (seq_len + 1)),
            shuffle=shuffle,
            seed=seed,
            num_epochs=num_samples // self.lens[-1] if num_samples and self.lens else 0,
            files_order=[str(f.file_path) for f in self.files],
        )


def build_dataset(
    dataset_folder: str,
    seq_length: int,
    token_size: int,
    return_positions: bool = False,
    eos_token_id: Optional[int] = None,
    use_old_brrr_dataloader: bool = False,
    skip_in_stream: bool = True,
    num_samples: Optional[int] = None,
    max_tokens: Optional[int] = None,
    skip_tokens: Optional[int] = None,
    shuffle: Optional[bool] = False,
    seed: Optional[int] = 6,
    folder_read_path: Optional[str] = None,
) -> "DatatroveFolderDataset":
    """Build one TokenizedBytes dataset from a file or a folder on S3 or locally

    Args:
        dataset_args (Union[TokenizedBytesDatasetFileArgs, TokenizedBytesDatasetFolderArgs]): dataset config
        seq_length ([type]): sequence length
        skip_in_stream (bool, optional): skip ahead in stream. Defaults to True.
    """
    if use_old_brrr_dataloader:
        log_rank(
            "Using old tokenized bytes folder dataset because skip_in_stream is False",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        if folder_read_path:
            log_rank(
                f"Ignoring folder_read_path={folder_read_path} because use_old_brrr_dataloader is True",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
        return OldTokenizedBytesFolderDataset(
            dataset_folder,
            seq_length,
            filename_pattern=".*\\.ds$",
            skip_in_stream=skip_in_stream,
            max_tokens=max_tokens,
            num_samples=num_samples,
            skip_tokens=skip_tokens,  # # Optional number of tokens to skip at the beginning (We'll only train on the rest)
            shuffle=shuffle,
            seed=seed,
            dtype=np.uint16 if token_size == 2 else np.uint32,
        )

    return TokenizedBytesFolderDataset(
        folder_path=dataset_folder,
        filename_pattern="*.ds",
        seq_len=seq_length,
        recursive=False,
        token_size=token_size,
        max_tokens=max_tokens, # TODO: remove
        shuffle=shuffle,
        return_positions=return_positions,  # if set to True, the position ids are directly read from datatrove
        eos_token_id=eos_token_id,
        seed=seed,
        skip_in_stream=skip_in_stream,
        num_samples=num_samples,
        folder_read_path=folder_read_path,
    )


def get_tb_datasets(
    config: NanosetDatasetsArgs,
    sequence_length: int,
    global_batch_size: int,
    train_steps: int,
    current_iteration: int,
    parallel_context: ParallelContext,
    eos_token_id: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 6,
    consumed_samples: int = 0,
    consumed_tokens_per_dataset_folder: Optional[Dict[str, int]] = None,
    last_stages_consumed_tokens_per_dataset_folder: Optional[Dict[str, int]] = None,
) -> Tuple[DataLoader, TrainDataLog]:
    """Build TokenizedBytes datasets

    Args:
        config (NanosetDatasetsArgs): dataset config
        sequence_length (int): sequence length
        global_batch_size (int): global batch size
        train_steps (int): number of training steps
        parallel_context (ParallelContext): distributed process groups
    """
    log_rank("Building Streamable datasets.", logger=logger, level=logging.INFO, rank=0)
    dataset_max_tokens = config.dataset_max_tokens
    if dataset_max_tokens is None:
        dataset_max_tokens = [None] * len(config.dataset_folder)
    train_num_samples = train_steps * global_batch_size
    last_stages_consumed_samples_per_dataset_folder = {k: v // sequence_length for k, v in last_stages_consumed_tokens_per_dataset_folder.items()}

    datasets = [
        build_dataset(
            dataset_folder,
            sequence_length,
            token_size=config.token_size_in_bytes,
            return_positions=config.return_positions,
            eos_token_id=eos_token_id,
            use_old_brrr_dataloader=config.use_old_brrr_dataloader,
            skip_in_stream=config.skip_in_stream,
            max_tokens=max_tokens,
            num_samples=train_num_samples,
            shuffle=shuffle,
            seed=seed,
            folder_read_path=config.dataset_read_path[i] if config.dataset_read_path else None,
        )
        for i, (dataset_folder, max_tokens) in enumerate(zip(config.dataset_folder, dataset_max_tokens))
    ]

    # in case of dataset_read_path check we have enough files locally for the training
    if config.dataset_read_path:

        weights = config.dataset_weights    
        if not weights:
            weights = [1] * len(datasets)

        # Normalize weights
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # check we have enough files locally for the training
        for i, dataset in enumerate(datasets):
            # warmup datasets
            estimate_current_sample = int(consumed_samples * weights[i]) + last_stages_consumed_samples_per_dataset_folder.get(dataset.folder_path, 0)
            _ = dataset[estimate_current_sample]
            # print which file we're currently reading from
            log_rank(f"Dataset {i} ({dataset.folder_path}) is reading from file {dataset.current_file_path}", logger=logger, level=logging.INFO, rank=0)
            # estimate number of tokens needed for this dataset
            needed_num_samples_dataset = int((train_steps - current_iteration) * global_batch_size * weights[i])
            needed_num_tokens_dataset = needed_num_samples_dataset * sequence_length
            needed_size_tokens_dataset = human_format(needed_num_tokens_dataset * config.token_size_in_bytes)
            log_rank(f"Dataset {i} ({dataset.folder_path}) needs {needed_num_tokens_dataset} tokens (size: {needed_size_tokens_dataset}) for current stage", logger=logger, level=logging.INFO, rank=0)

            # NOTE: let's assume that s3 folder keep the same old files when resuming
            # check that sum of lens of files in dataset is greater than needed_num_samples_dataset (use dataset.lens)
            total_num_samples_dataset = int(train_steps * global_batch_size * weights[i])
            log_rank(f"Dataset {i} ({dataset.folder_path}) on s3 has {len(dataset) * sequence_length} tokens (size: {human_format(len(dataset) * sequence_length * config.token_size_in_bytes)}) and needs {total_num_samples_dataset * sequence_length} tokens (size: {human_format(total_num_samples_dataset * sequence_length * config.token_size_in_bytes)}) for all stages", logger=logger, level=logging.INFO, rank=0)
            assert total_num_samples_dataset <= len(dataset), f"Not enough files on s3 for dataset {i} ({dataset.folder_path})"
            # check that local files exist for the needed_num_samples_dataset
            estimate_end_sample = estimate_current_sample + needed_num_samples_dataset
            for file_idx, file in enumerate(dataset.files):
                # intersection [start_sample, end_sample] with [dataset.lens[file_idx], dataset.lens[file_idx+1]]
                a, b, c, d = estimate_current_sample, estimate_end_sample, dataset.lens[file_idx], dataset.lens[file_idx+1]
                if max(a, c) < min(b, d): # ranges overlap
                    assert os.path.exists(file.file_path), f"Dataset {i} ({dataset.folder_path}) will need file {file.file_path} but it does not exist"
                    log_rank(f"Dataset {i} ({dataset.folder_path}) will need file {file.file_path} from sample {max(a, c)} to {min(b, d)} (offset: {last_stages_consumed_samples_per_dataset_folder.get(dataset.folder_path, 0)})", logger=logger, level=logging.INFO, rank=0)
                else:
                    log_rank(f"Dataset {i} ({dataset.folder_path}) will not need file {file.file_path} to train from sample {estimate_current_sample} to {estimate_end_sample} (offset: {last_stages_consumed_samples_per_dataset_folder.get(dataset.folder_path, 0)})", logger=logger, level=logging.INFO, rank=0)
                    

    if len(datasets) == 1 and False:
        outputs_dataset = datasets[0]
    else:
        if dist.get_rank(parallel_context.world_pg) == 0:
            try:
                compile_helper()
            except ImportError:
                raise ImportError(
                    "Could not compile megatron dataset C++ helper functions and therefore cannot import helpers python file."
                )
        dist.barrier(parallel_context.world_pg)
        weights = config.dataset_weights
        if not weights:
            weights = [1] * len(datasets)

        outputs_dataset = BlendableDataset(
            datasets,
            weights,
            train_num_samples,
            parallel_context=parallel_context,
            seed=seed,
            consumed_tokens_per_dataset_folder=consumed_tokens_per_dataset_folder,
            offsets_in_samples=last_stages_consumed_samples_per_dataset_folder,
        )

    log_rank("Streamable datasets ready.", logger=logger, level=logging.INFO, rank=0)
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


def get_tb_dataloader(
    dataset: Union[DatatroveFolderDataset, BlendableDataset, Dataset],
    sequence_length: int,
    micro_batch_size: int,
    global_batch_size: int,
    cfg: NanosetDatasetsArgs,
    num_workers: int,
    consumed_samples: int,
    num_samples: int,
    parallel_context: ParallelContext,
    input_pp_rank: int,
    output_pp_rank: int,
    dataloader_drop_last: bool = True,
    dataloader_pin_memory: bool = True,
    use_position_ids: bool = False,
    use_doc_masking: bool = False,
) -> DataLoader:
    # Only some rank require to run the dataloader.
    if dist.get_rank(parallel_context.pp_pg) not in [
        input_pp_rank,
        output_pp_rank,
    ]:
        dataset = EmptyInfiniteDataset(length=len(dataset))

    log_rank(
        f"Building dataloader with consumed samples for current datastage: {consumed_samples}", logger=logger, level=logging.INFO, rank=0
    )
    # Megatron sampler
    # batch_sampler = MegatronPretrainingRandomSampler(
    batch_sampler = MegatronPretrainingSampler(
        total_samples=num_samples,
        consumed_samples=consumed_samples,
        micro_batch_size=micro_batch_size,
        data_parallel_rank=dist.get_rank(parallel_context.dp_pg),
        data_parallel_size=parallel_context.dp_pg.size(),
        drop_last=dataloader_drop_last,
        global_batch_size=global_batch_size,
        pad_samples_to_global_batch_size=cfg.pad_samples_to_global_batch_size,
    )

    # We use the data collator to put the tensors on the right pipeline parallelism rank
    if use_position_ids:
        data_collator = DataCollatorForCLMWithPositionIds(
            sequence_length=sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            parallel_context=parallel_context,
            use_doc_masking=use_doc_masking,
        )
    else:
        data_collator = DataCollatorForCLM(
            sequence_length=sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            parallel_context=parallel_context,
        )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=dataloader_pin_memory,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dist.get_rank(parallel_context.dp_pg)),
    )
