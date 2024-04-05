import dataclasses
import warnings
from typing import Dict, Generator, Iterator, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import Config, PretrainDatasetsArgs
from nanotron.logging import log_rank
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.random import set_random_seed
from nanotron.sanity_checks import (
    assert_fail_except_rank_with,
    assert_tensor_synced_across_pg,
)
from nanotron.utils import main_rank_first

try:
    import datasets
    from datasets import (
        Dataset,
        DatasetDict,
        Features,
        Sequence,
        Value,
        concatenate_datasets,
        interleave_datasets,
        load_dataset,
    )
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
    from transformers import __version__ as tf_version
    from transformers.trainer_pt_utils import DistributedSamplerWithLoop
except ImportError:
    warnings.warn("Datasets and/or Transformers not installed, you'll be unable to use the dataloader.")
    hf_hub_version = None
    tf_version = None


logger = logging.get_logger(__name__)


class CombinedDataset(Dataset):
    def __init__(self, datasets: List[Dataset], weights: torch.Tensor, seed: int):
        assert len(datasets) == len(weights)
        assert weights.sum() == 1.0
        assert weights.ndim == 1

        for i in range(len(datasets)):
            assert all(x["dataset_idxs"] == i for x in datasets[i])

        self.datasets = datasets
        self.comebined_dataset = concatenate_datasets(datasets)
        # self.comebined_dataset = interleave_datasets(datasets, weights.tolist(), stopping_strategy="first_exhausted", seed=42)
        # self.comebined_dataset = interleave_datasets(datasets, stopping_strategy="first_exhausted", seed=42)
        # assert 1 == 1

    def __len__(self) -> int:
        return len(self.comebined_dataset)

    def __getitem__(self, batch: Union[int, List[int]]) -> Dict[str, Union[torch.Tensor, np.array]]:
        if isinstance(batch, list) is False:
            batch = [batch]

        assert len(batch) > 0
        if isinstance(batch[0], list):
            # TODO(xrsrke): do a single index, then split the output
            samples = [self.comebined_dataset[idxs] for idxs in batch]
            return self._merge_dicts(samples)

        return self.comebined_dataset[batch]

    def _merge_dicts(self, data):
        merged = {}
        for key in data[0].keys():
            merged[key] = np.concatenate([d[key] for d in data if key in d])
        return merged


import math


class WeightedDistributedSampler(DistributedSampler):
    def __init__(
        self,
        weights: torch.Tensor,
        datasets: List[Dataset],
        batch_size: int,
        num_microbatches: int,
        num_replicas: int,
        rank: int,
        parallel_context: ParallelContext,
        shuffle: bool = False,
        seed: int = 42,
        drop_last: bool = False,
    ):
        super().__init__(datasets, num_replicas=num_replicas, rank=rank, shuffle=shuffle, drop_last=drop_last)

        self.weights = weights
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_microbatches = num_microbatches
        self.parallel_context = parallel_context
        self.total_size = self._calculate_total_size()

        self.lengths = [len(d) for d in self.datasets]
        self.offsets = np.cumsum([0] + self.lengths[:-1])
        self.seed = seed

        # self.global_batch_size = batch_size * dist.get_world_size(parallel_context.dp_pg) * num_microbatches
        self.global_batch_size = batch_size * self.num_replicas * num_microbatches
        # NOTE: Reset the seed of the generator for consistent randomness across epochs
        self.generator = torch.Generator(device="cpu").manual_seed(
            seed * (1 + dist.get_rank(self.parallel_context.dp_pg)) * (1 + dist.get_rank(self.parallel_context.pp_pg))
        )

        self.reset()

    def _calculate_total_size(self):
        total_samples = sum(len(d) for d in self.datasets)
        return math.ceil(total_samples / self.batch_size) * self.batch_size

    def __iter__(self):
        return self

    def _recompute_domain_batch_sizes(self, domain_weights):
        domain_batch_sizes = [round(self.global_batch_size * weight.item()) for weight in domain_weights]

        # NOTE: in some cases, the weight of a domain is too small
        # resulting in a domain with 0 samples per global batch
        # => zero loss for that domain => we no longer update the weights of that domain
        # so we add a sample to that domain
        domain_batch_sizes = [1 if x < 1 else x for x in domain_batch_sizes]

        if sum(domain_batch_sizes) != self.global_batch_size:
            # NOTE: randomly add a sample to round it up
            domain_batch_sizes = self._round_up_domain_batch_sizes(
                domain_batch_sizes,
                target_total_size=self.global_batch_size,
            )

        assert all(x > 0 for x in domain_batch_sizes), "There is a domain with 0 samples per global batch"
        return domain_batch_sizes

    def __next__(self):
        if self.microbatch_idx == 0:
            # NOTE: because we randomly add a sample to round up the domain batch sizes
            # so it's better if we recompute the global batch every time we start a new microbatch
            # so that not bias towards a domain (where that domain gets more samples than the others)
            self.domain_batch_sizes = self._recompute_domain_batch_sizes(
                domain_weights=self.weights,
            )

            self.batch = []
            for domain_index, (idxs, domain_batch_size) in enumerate(
                zip(self.domain_indices, self.domain_batch_sizes)
            ):
                start_idx = self.domain_counters[domain_index]
                end_idx = start_idx + domain_batch_size

                if end_idx > len(idxs):
                    raise StopIteration(f"Domain {domain_index}-th ran out of samples")

                assert self.domain_counters[domain_index] + domain_batch_size == end_idx
                self.domain_counters[domain_index] = end_idx
                global_batch_idxs = idxs[start_idx:end_idx]
                self.batch.extend(global_batch_idxs)

        num_samples_per_dp_rank = self.batch_size * self.num_microbatches
        dp_start_idx = self.rank * num_samples_per_dp_rank
        dp_end_idx = dp_start_idx + num_samples_per_dp_rank

        if dp_end_idx > len(self.batch):
            raise StopIteration(f"[DoReMi] Rank {self.rank} ran out of samples, len(batch)={len(self.batch)}")

        dp_batch = self.batch[dp_start_idx:dp_end_idx]

        microbatch_start_idx = self.microbatch_idx * self.batch_size
        microbatch_end_idx = microbatch_start_idx + self.batch_size

        if microbatch_end_idx > len(dp_batch):
            raise StopIteration(
                f"[DoReMi] Rank {self.rank}'s microbatch {self.microbatch_idx}-th ran out of samples, len(dp_batch)={len(dp_batch)}"
            )

        microbatch_idxs = dp_batch[microbatch_start_idx:microbatch_end_idx]

        if self.microbatch_idx == self.num_microbatches - 1:
            self.microbatch_idx = 0
        else:
            self.microbatch_idx += 1

        return microbatch_idxs

    def _recompute_global_batch(self):
        self.domain_batch_sizes = self._recompute_domain_batch_sizes(
            domain_weights=self.weights,
        )
        for domain_index, (idxs, domain_batch_size) in enumerate(zip(self.domain_indices, self.domain_batch_sizes)):
            start_idx = self.domain_counters[domain_index]
            end_idx = start_idx + domain_batch_size

            if end_idx > len(idxs):
                raise StopIteration(f"Domain {domain_index}-th ran out of samples")

            self.domain_counters[domain_index] = end_idx
            global_batch_idxs = idxs[start_idx:end_idx]
            self.batch.extend(global_batch_idxs)

    def _round_up_domain_batch_sizes(self, domain_batch_sizes: List[int], target_total_size: int) -> List[int]:
        """
        NOTE: Makes sum(domain_batch_sizes) == batch_size
        """
        total_batch_size = sum(domain_batch_sizes)
        while total_batch_size != target_total_size:
            diff = target_total_size - total_batch_size

            # NOTE: Randomly select a domain to increase/decrase a sample
            # to match the target_total_size
            eligible_indices = torch.nonzero(torch.tensor(domain_batch_sizes) > 1).view(-1)
            random_index = torch.randint(
                low=0, high=len(eligible_indices), size=(1,), generator=self.generator, device="cpu"
            ).item()
            selected_domain = eligible_indices[random_index].item()

            if diff > 0:
                domain_batch_sizes[selected_domain] += 1
            elif diff < 0 and domain_batch_sizes[selected_domain] > 0:
                domain_batch_sizes[selected_domain] -= 1

            total_batch_size = sum(domain_batch_sizes)

        return domain_batch_sizes

    def reset(self):
        """Reset the state of the sampler for a new epoch."""
        self.microbatch_idx = 0
        self.domain_counters = [0 for _ in self.datasets]
        self.total_samples_yielded = 0
        self.out_of_samples = False

        domain_indices = []
        for i, dataset in enumerate(self.datasets):
            local_indices = torch.arange(0, len(dataset), device="cpu").tolist()

            # NOTE: align the indices across the combined dataset
            global_indices = local_indices + self.offsets[i]
            domain_indices.append(global_indices)

        self.num_samples_per_global_step = self.batch_size * self.num_microbatches * self.num_replicas
        self.domain_indices = domain_indices
        self.expected_total_samples = sum([len(d) for d in domain_indices])


import abc
from dataclasses import dataclass


@dataclass
class BaseMegatronSampler:
    total_samples: int
    consumed_samples: int
    micro_batch_size: int
    data_parallel_rank: int
    data_parallel_size: int
    global_batch_size: int
    drop_last: bool = True
    pad_samples_to_global_batch_size: Optional[bool] = False

    def __post_init__(self):
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * self.data_parallel_size

        # Sanity checks.
        if self.total_samples <= 0:
            raise RuntimeError("no sample to consume: {}".format(self.total_samples))
        if self.consumed_samples >= self.total_samples:
            raise RuntimeError("no samples left to consume: {}, {}".format(self.consumed_samples, self.total_samples))
        if self.micro_batch_size <= 0:
            raise RuntimeError(f"micro_batch_size size must be greater than 0, but {self.micro_batch_size}")
        if self.data_parallel_size <= 0:
            raise RuntimeError(f"data parallel size must be greater than 0, but {self.data_parallel_size}")
        if self.data_parallel_rank >= self.data_parallel_size:
            raise RuntimeError(
                "data_parallel_rank should be smaller than data size, but {} >= {}".format(
                    self.data_parallel_rank, self.data_parallel_size
                )
            )
        if self.global_batch_size % (self.micro_batch_size * self.data_parallel_size) != 0:
            raise RuntimeError(
                f"`global_batch_size` ({self.global_batch_size}) is not divisible by "
                f"`micro_batch_size ({self.micro_batch_size}) x data_parallel_size "
                f"({self.data_parallel_size})`"
            )
        if self.pad_samples_to_global_batch_size and self.global_batch_size is None:
            raise RuntimeError(
                "`pad_samples_to_global_batch_size` can be `True` only when "
                "`global_batch_size` is set to an integer value"
            )
        log_rank(
            f"Instantiating MegatronPretrainingSampler with total_samples: {self.total_samples} and consumed_samples: {self.consumed_samples}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

    @abc.abstractmethod
    def __iter__(self):
        ...


@dataclass
class MegatronPretrainingSampler(BaseMegatronSampler):
    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __len__(self):
        num_available_samples: int = self.total_samples - self.consumed_samples
        if self.global_batch_size is not None:
            if self.drop_last:
                return num_available_samples // self.global_batch_size
            else:
                return (num_available_samples + self.global_batch_size - 1) // self.global_batch_size
        else:
            if self.drop_last:
                return num_available_samples // self.micro_batch_times_data_parallel_size
            else:
                return (num_available_samples - 1) // self.micro_batch_times_data_parallel_size + 1

    def __iter__(self):
        batch = []
        batch_idx = 0
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                log_rank(
                    f"DLrank {self.data_parallel_rank} batch {batch_idx} {batch[start_idx:end_idx]} self.consumed_samples {self.consumed_samples}",
                    logger=logger,
                    level=logging.DEBUG,
                )
                yield batch[start_idx:end_idx]
                batch = []
                batch_idx += 1

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            if self.pad_samples_to_global_batch_size:
                for i in range(
                    self.data_parallel_rank, self.global_batch_size, self.micro_batch_times_data_parallel_size
                ):
                    indices = [batch[j] for j in range(i, max(len(batch), i + self.micro_batch_size))]
                    num_pad = self.micro_batch_size - len(indices)
                    indices = indices + [-1] * num_pad
                    yield indices
            else:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]


def sanity_check_dataloader(
    dataloader: Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]],
    parallel_context: ParallelContext,
    config: Config,
) -> Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]:
    for batch in dataloader:
        micro_batch = {
            k: v if isinstance(v, TensorPointer) else v.to("cuda", memory_format=torch.contiguous_format)
            for k, v in batch.items()
        }

        if not config.general.ignore_sanity_checks:
            # SANITY CHECK: Check input are not the same across DP
            for key, value in sorted(micro_batch.items(), key=lambda x: x[0]):
                if isinstance(value, TensorPointer):
                    continue

                if "mask" in key:
                    # It's fine if mask is the same across DP
                    continue

                with assert_fail_except_rank_with(AssertionError, rank_exception=0, pg=parallel_context.dp_pg):
                    assert_tensor_synced_across_pg(
                        tensor=value, pg=parallel_context.dp_pg, msg=lambda err: f"{key} {err}"
                    )

            # SANITY CHECK: Check input are synchronized throughout TP
            for key, value in sorted(micro_batch.items(), key=lambda x: x[0]):
                if isinstance(value, TensorPointer):
                    continue
                assert_tensor_synced_across_pg(
                    tensor=value,
                    pg=parallel_context.tp_pg,
                    msg=lambda err: f"{key} are not synchronized throughout TP {err}",
                )

            # SANITY CHECK: Check that input are synchronized throughout PP
            # TODO @thomasw21: That's really hard to test as input gets sharded across the PP, let's assume it works for now.

            # SANITY CHECK: Check that an input only exists on the PP rank responsible for it
            # TODO @nouamanetazi: add this test
        yield micro_batch


# Adapted from h4/src/h4/data/loading.py
def get_datasets(
    hf_dataset_or_datasets: Union[dict, str],
    splits: Optional[Union[List[str], str]] = ["train", "test"],
) -> "DatasetDict":
    """
    Function to load dataset directly from DataArguments.

    Args:
        hf_dataset_or_datasets (Union[dict, str]): dict or string. When all probabilities are 1, we concatenate the datasets instead of sampling from them.
        splits (Optional[List[str]], optional): Section of the dataset to load, defaults to "train", "test"
            Can be one of `train_ift`, `test_rl`, or `..._rm` etc. H4 datasets are divided into 6 subsets for training / testing.

    Returns
        DatasetDict: DatasetDict object containing the dataset of the appropriate section with test + train parts.
    """

    if isinstance(splits, str):
        splits = [splits]

    if isinstance(hf_dataset_or_datasets, dict):
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        raw_datasets = _get_dataset_mix(hf_dataset_or_datasets, splits=splits)
    elif isinstance(hf_dataset_or_datasets, str):
        # e.g. Dataset = "HuggingFaceH4/testing_alpaca_small"
        # Note this returns things other than just train/test, which may not be intended
        raw_datasets = DatasetDict()
        for split in splits:
            raw_datasets[split] = load_dataset(
                hf_dataset_or_datasets,
                split=split,
            )
    else:
        raise ValueError(f"hf_dataset_or_datasets must be a dict or string but is {type(hf_dataset_or_datasets)}")

    return raw_datasets


# Adapted from h4/src/h4/data/loading.py
def _get_dataset_mix(dataset_dict: dict, splits: List[str] = None, seed: int = 42) -> "DatasetDict":
    """
    Helper function to load dataset mix from dict configuration.

    Args:
        dataset_dict (dict): Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], optional): Section of the dataset to load, defaults to "train", "test"
            Can be one of `train_{ift,rm,rl}` and `test_{ift,rm,rl}`. Our datasets are typically divided into 6 subsets for training / testing.
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_test_datasets = []
    fracs = []
    for ds, frac in dataset_dict.items():
        if frac < 0:
            raise ValueError(f"Dataset fraction for dataset {ds} is negative. (= {frac})")

        fracs.append(frac)
        for split in splits:
            if "train" in split:
                raw_train_datasets.append(
                    load_dataset(
                        ds,
                        split=split,
                    )
                )
            elif "test" in split:
                raw_test_datasets.append(
                    load_dataset(
                        ds,
                        split=split,
                    )
                )
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset_idx, (dataset, frac) in enumerate(zip(raw_train_datasets, fracs)):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subset = train_subset.add_column("dataset_idxs", [dataset_idx] * len(train_subset))
            train_subsets.append(train_subset)
        raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=seed)

    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_test_datasets) > 0:
        test_subsets = []
        for idx, dataset in enumerate(raw_test_datasets):
            dataset = dataset.add_column("dataset_idxs", [idx] * len(dataset))
            test_subsets.append(dataset)
        raw_datasets["test"] = concatenate_datasets(raw_test_datasets).shuffle(seed=seed)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_dict} not recognized with split {split}. Check the dataset has been correctly formatted."
        )

    return raw_datasets


def dummy_infinite_data_generator(
    micro_batch_size: int,
    sequence_length: int,
    input_pp_rank: int,
    output_pp_rank: int,
    vocab_size: int,
    seed: int,
    parallel_context: ParallelContext,
):
    def data_generator() -> Generator[Dict[str, Union[torch.Tensor, TensorPointer]], None, None]:
        # Random generator
        generator = torch.Generator(device="cuda")
        # Make sure that TP are synced always
        generator.manual_seed(
            seed * (1 + dist.get_rank(parallel_context.dp_pg)) * (1 + dist.get_rank(parallel_context.pp_pg))
        )

        while True:
            yield {
                "input_ids": torch.randint(
                    0,
                    vocab_size,
                    (micro_batch_size, sequence_length),
                    dtype=torch.long,
                    device="cuda",
                    generator=generator,
                )
                if dist.get_rank(parallel_context.pp_pg) == input_pp_rank
                else TensorPointer(group_rank=input_pp_rank),
                "input_mask": torch.ones(
                    micro_batch_size,
                    sequence_length,
                    dtype=torch.bool,
                    device="cuda",
                )
                if dist.get_rank(parallel_context.pp_pg) == input_pp_rank
                else TensorPointer(group_rank=input_pp_rank),
                "label_ids": torch.randint(
                    0,
                    vocab_size,
                    (micro_batch_size, sequence_length),
                    dtype=torch.long,
                    device="cuda",
                    generator=generator,
                )
                if dist.get_rank(parallel_context.pp_pg) == output_pp_rank
                else TensorPointer(group_rank=output_pp_rank),
                "label_mask": torch.ones(
                    micro_batch_size,
                    sequence_length,
                    dtype=torch.bool,
                    device="cuda",
                )
                if dist.get_rank(parallel_context.pp_pg) == output_pp_rank
                else TensorPointer(group_rank=output_pp_rank),
            }

    return data_generator


# Adapted from https://github.com/huggingface/accelerate/blob/a73898027a211c3f6dc4460351b0ec246aa824aa/src/accelerate/data_loader.py#L781C1-L824C28
class SkipBatchSampler(BatchSampler):
    """
    A `torch.utils.data.BatchSampler` that skips the first `n` batches of another `torch.utils.data.BatchSampler`.
    Note that in case of DDP, we skip batches on each rank, so a total of `skip_batches * parallel_context.dp_pg.size()` batches
    """

    def __init__(self, batch_sampler: BatchSampler, skip_batches: int, dp_size: int):
        self.batch_sampler = batch_sampler
        # In case of DDP, we skip batches on each rank, so a total of `skip_batches * parallel_context.dp_pg.size()` batches
        self.skip_batches = skip_batches // dp_size

    def __iter__(self):
        for index, samples in enumerate(self.batch_sampler):
            if index >= self.skip_batches:
                yield samples

    @property
    def total_length(self):
        return len(self.batch_sampler)

    def __len__(self):
        return len(self.batch_sampler) - self.skip_batches


def set_tensor_pointers(
    input_dict: Dict[str, Union[torch.Tensor, TensorPointer]], group: dist.ProcessGroup, group_rank: int
) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
    """Make sure only the group_rank rank has the data, others have TensorPointers."""
    return {
        k: v if dist.get_rank(group) == group_rank else TensorPointer(group_rank=group_rank)
        for k, v in input_dict.items()
    }


### CAUSAL LANGUAGE MODELING ###
def clm_process(
    raw_dataset: "Dataset",
    tokenizer: "PreTrainedTokenizerBase",
    text_column_name: str,
    dataset_processing_num_proc_per_process: int,
    dataset_overwrite_cache: bool,
    sequence_length: int,
    dataset_idx: int,
):
    """Concatenate all texts from raw_dataset and generate chunks of `sequence_length + 1`, where chunks overlap by a single token."""
    # Adapted from https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/examples/pytorch/language-modeling/run_clm.py#L391-L439

    def group_texts(examples: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        # Concatenate all texts.
        concatenated_examples = {k: np.concatenate(v) for k, v in examples.items()}
        total_length = len(concatenated_examples[next(iter(examples.keys()))])
        # WARNING: We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= sequence_length + 1:
            total_length = ((total_length - 1) // sequence_length) * sequence_length + 1
        # Split by chunks of sequence_length.
        result = {
            k: [
                t[i : i + sequence_length + 1] for i in range(0, total_length - (sequence_length + 1), sequence_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    def _tokenize_and_group_texts(texts: List[str]) -> Dict[str, List[np.ndarray]]:
        tokenized_batch = tokenizer.batch_encode_plus(texts, return_attention_mask=False, return_token_type_ids=False)
        tokenized_batch = {k: [np.array(tokenized_texts) for tokenized_texts in v] for k, v in tokenized_batch.items()}
        return group_texts(tokenized_batch)

    train_dataset = raw_dataset.map(
        _tokenize_and_group_texts,
        input_columns=text_column_name,
        remove_columns=raw_dataset.column_names,
        features=Features({"input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length + 1)}),
        batched=True,
        num_proc=dataset_processing_num_proc_per_process,
        load_from_cache_file=not dataset_overwrite_cache,
        desc=f"Grouping texts in chunks of {sequence_length+1}",
    )
    # TODO(xrsrke): remove this shit
    train_dataset = train_dataset.add_column("dataset_idxs", [dataset_idx] * len(train_dataset))
    return train_dataset


# Adapted from: https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/data/data_collator.py#L607
@dataclasses.dataclass
class DataCollatorForCLM:
    """
    Data collator used for causal language modeling.

    - input_pp_rank: Discards last input id token
    - output_pp_rank: Discards first label id token
    - other pp ranks: Don't have data. Instead, we use `TensorPointer` to point to the rank having the data.
    """

    sequence_length: int
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext

    def __call__(self, examples: List[Dict[str, List[np.ndarray]]]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Process the case when current rank doesn't require data. We return `TensorPointer` that points to ranks having the data.
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        if current_pp_rank not in [
            self.input_pp_rank,
            self.output_pp_rank,
        ]:
            assert all(len(example) == 0 for example in examples)
            return {
                "input_ids": TensorPointer(self.input_pp_rank),
                "input_mask": TensorPointer(self.input_pp_rank),
                "label_ids": TensorPointer(self.output_pp_rank),
                "label_mask": TensorPointer(self.output_pp_rank),
            }

        # Make sure we load only what's necessary, ie we only load a `input_ids` column.
        # assert all(list(example.keys()) == ["input_ids"] for example in examples)
        assert all(list(example.keys()) == ["input_ids", "dataset_idxs"] for example in examples)

        # TODO @nouamanetazi: Is it better to have examples as np.array or torch.Tensor?
        input_ids = np.vstack([examples[i]["input_ids"] for i in range(len(examples))])  # (b, s)
        batch_size, expanded_input_length = input_ids.shape

        result: Dict[str, Union[np.ndarray, TensorPointer]] = {}

        result["input_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["input_mask"] = TensorPointer(group_rank=self.input_pp_rank)
        result["label_ids"] = TensorPointer(group_rank=self.output_pp_rank)
        result["label_mask"] = TensorPointer(group_rank=self.output_pp_rank)
        result["dataset_idxs"] = TensorPointer(group_rank=self.output_pp_rank)

        assert (
            expanded_input_length == self.sequence_length + 1
        ), f"Samples should be of length {self.sequence_length + 1} (seq_len+1), but got {expanded_input_length}"

        # Process inputs: last token is the label
        if current_pp_rank == self.input_pp_rank:
            result["input_ids"] = input_ids[:, :-1]
            result["input_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

        # Process labels: shift them to the left
        if current_pp_rank == self.output_pp_rank:
            result["label_ids"] = input_ids[:, 1:]
            result["label_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

            dataset_idxs = np.vstack([examples[i]["dataset_idxs"] for i in range(len(examples))])  # (b, s)
            result["dataset_idxs"] = dataset_idxs

        if isinstance(result["input_ids"], torch.Tensor) and result["input_ids"].shape[-1] != self.sequence_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {result['input_ids'].shape[-1]}, but should be"
                f" {self.sequence_length}."
            )
        if isinstance(result["label_ids"], torch.Tensor) and result["label_ids"].shape[-1] != self.sequence_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {result['label_ids'].shape[-1]}, but should be"
                f" {self.sequence_length}."
            )

        # Cast np.array to torch.Tensor
        result = {k: v if isinstance(v, TensorPointer) else torch.from_numpy(v) for k, v in result.items()}
        return result


# Adapted from https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L763-L835
def _get_train_sampler(
    dl_ranks_size: int,
    dl_rank: int,
    train_dataset: "Dataset",
    shuffle: bool,
    seed: int,
    use_loop_to_round_batch_size: bool,
    consumed_train_samples: int,
    micro_batch_size: Optional[int] = None,
    drop_last: Optional[bool] = True,
) -> Optional[torch.utils.data.Sampler]:
    """returns sampler that restricts data loading to a subset of the dataset proper to the DP rank"""

    # Build the sampler.
    # TODO @nouamanetazi: Support group_by_length: https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L783-L810

    if use_loop_to_round_batch_size:
        assert micro_batch_size is not None
        # loops at the end back to the beginning of the shuffled samples to make each process have a round multiple of batch_size samples.
        sampler = DistributedSamplerWithLoop(
            train_dataset,
            batch_size=micro_batch_size,
            num_replicas=dl_ranks_size,
            rank=dl_rank,
            seed=seed,
            drop_last=drop_last,
        )
    else:
        sampler = DistributedSampler(
            train_dataset, num_replicas=dl_ranks_size, rank=dl_rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )

    if consumed_train_samples > 0:
        sampler = SkipBatchSampler(sampler, skip_batches=consumed_train_samples, dp_size=dl_ranks_size)

    return sampler


# Adapted from https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L837
def get_train_dataloader(
    train_dataset: "Dataset",
    sequence_length: int,
    parallel_context: ParallelContext,
    input_pp_rank: int,
    output_pp_rank: int,
    micro_batch_size: int,
    consumed_train_samples: int,
    dataloader_num_workers: int,
    seed_worker: int,
    dataloader_drop_last: bool = True,
    dataloader_pin_memory: bool = True,
    use_loop_to_round_batch_size: bool = False,
    # TODO(xrsrke): refactor
    weights: Optional[torch.Tensor] = None,
    num_microbatches: Optional[int] = None,
) -> DataLoader:
    if not isinstance(train_dataset, datasets.Dataset):
        raise ValueError(f"training requires a datasets.Dataset, but got {type(train_dataset)}")

    # Case of ranks requiring data
    if dist.get_rank(parallel_context.pp_pg) in [
        input_pp_rank,
        output_pp_rank,
    ]:
        # for dataset_idx in range(len(train_dataset.comebined_dataset)):
        #     d = train_dataset.comebined_dataset[dataset_idx]
        #     train_dataset.comebined_dataset[dataset_idx] = d.with_format(type="numpy", columns=["input_ids"], output_all_columns=True)
        pass
    # Case of ranks not requiring data. We give them an infinite dummy dataloader
    else:
        # TODO(xrsrke): delete train_dataset from memory

        dataset_length = len(train_dataset)
        # for dataset_idx in range(len(train_dataset.comebined_dataset)):
        #     d = train_dataset.comebined_dataset[dataset_idx]
        #     assert d.column_names == ["input_ids"], (
        #         f"Dataset has to have a single column, with `input_ids` as the column name. "
        #         f"Current dataset: {train_dataset}"
        #     )
        #     dataset_length = len(d)
        #     d = d.remove_columns(column_names="input_ids")
        #     assert (
        #         len(d) == 0
        #     ), f"Dataset has to be empty after removing the `input_ids` column. Current dataset: {d}"
        #     # HACK as if we remove the last column of a train_dataset, it becomes empty and it's number of rows becomes empty.

        train_dataset = EmptyInfiniteDataset(length=dataset_length)
        # No need to spawn a lot of workers, we can just use main
        dataloader_num_workers = 0

    data_collator = DataCollatorForCLM(
        sequence_length=sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        parallel_context=parallel_context,
    )

    # Compute size and rank of dataloader workers
    dp_ranks_size = parallel_context.dp_pg.size()
    dp_rank = parallel_context.dp_pg.rank()

    # TODO @nouamanetazi: Remove unused columns: https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L852
    # TODO @nouamanetazi: Support torch.utils.data.IterableDataset: https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L855-L872

    train_sampler = WeightedDistributedSampler(
        weights=weights,
        datasets=train_dataset.datasets,
        batch_size=micro_batch_size,
        num_microbatches=num_microbatches,
        num_replicas=dp_ranks_size,
        rank=dp_rank,
        seed=42,
        drop_last=True,
        parallel_context=parallel_context,
    )

    # train_sampler = _get_train_sampler(
    #     dl_rank=dp_rank,
    #     dl_ranks_size=dp_ranks_size,
    #     train_dataset=train_dataset,
    #     shuffle=False,
    #     seed=seed_worker,
    #     use_loop_to_round_batch_size=use_loop_to_round_batch_size,
    #     micro_batch_size=micro_batch_size,
    #     drop_last=dataloader_drop_last,
    #     consumed_train_samples=consumed_train_samples,
    # )

    # return DataLoader(
    #     train_dataset,
    #     shuffle=False,
    #     batch_size=micro_batch_size,
    #     batch_sampler=train_sampler,
    #     collate_fn=data_collator,
    #     drop_last=dataloader_drop_last,  # we also drop_last in `clm_process()`
    #     num_workers=dataloader_num_workers,
    #     pin_memory=dataloader_pin_memory,
    #     worker_init_fn=get_dataloader_worker_init(dp_rank=dp_rank),
    #     # TODO @thomasw21: I'm not sure but this doesn't seem to work at all.
    #     # pin_memory_device="cuda",
    # )

    return DataLoader(
        train_dataset,
        # shuffle=False,
        # batch_size=micro_batch_size,
        batch_sampler=train_sampler,
        collate_fn=data_collator,
        # drop_last=dataloader_drop_last,  # we also drop_last in `clm_process()`
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dp_rank),
        # TODO @thomasw21: I'm not sure but this doesn't seem to work at all.
        # pin_memory_device="cuda",
    )


def get_dataloader_worker_init(dp_rank: int):
    """Creates random states for each worker in order to get different state in each workers"""

    def dataloader_worker_init(worker_id):
        # Dataloader is TP/PP synced in random states
        seed = 2 ** (1 + worker_id) * 3 ** (1 + dp_rank) % (2**32)
        set_random_seed(seed)

    return dataloader_worker_init


class EmptyInfiniteDataset:
    """Hack as removing all columns from a datasets.Dataset makes the number of rows 0."""

    def __init__(self, length: int):
        self._length = length

    def __getitem__(self, item) -> Dict:
        if isinstance(item, int):
            return {}
        raise NotImplementedError(f"{item} of type {type(item)} is not supported yet")

    def __len__(self) -> int:
        return self._length


def get_dataloader(trainer: "DistributedTrainer"):
    """Returns a dataloader for training."""

    # First, we need to know which ranks to feed the dataloader to
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    # Case 1: Dummy data generator
    if trainer.config.data.dataset is None:
        log_rank("Using dummy data generator", logger=logger, level=logging.INFO, rank=0)
        dataloader = dummy_infinite_data_generator(
            micro_batch_size=trainer.micro_batch_size,
            sequence_length=trainer.sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            vocab_size=trainer.model_config.vocab_size,
            seed=trainer.config.data.seed,
            parallel_context=trainer.parallel_context,
        )()

    # Case 2: HuggingFace datasets
    elif isinstance(trainer.config.data.dataset, PretrainDatasetsArgs):
        log_rank("Using `datasets` library", logger=logger, level=logging.INFO, rank=0)
        tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
        log_rank(
            f"Loading tokenizer from {tokenizer_path} and transformers/hf_hub versions {tf_version, hf_hub_version}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        # We need to the 1st device to process dataset and cache it, then other devices load from cache
        with main_rank_first(trainer.parallel_context.world_pg):
            # TODO @nouamanetazi: this may timeout before 1st device finishes processing dataset. Can we have a ctxmanager to modify timeout?
            # TODO: generalise to include  for validation/test splits

            # We load the raw dataset
            raw_dataset = get_datasets(
                hf_dataset_or_datasets=trainer.config.data.dataset.hf_dataset_or_datasets,
                splits=trainer.config.data.dataset.hf_dataset_splits,
            )["train"]

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            num_datasets = len(trainer.config.data.dataset.hf_dataset_or_datasets)
            if "dataset_idxs" in raw_dataset.column_names:
                # assert num_datasets > 1, "Multiple datasets are required to use `dataset_idxs` column."

                raw_datasets = []
                for i in range(num_datasets):
                    raw_datasets.append(raw_dataset.filter(lambda x: x["dataset_idxs"] == i))

            # We apply the Causal Language Modeling preprocessing
            tokenized_datasets = []
            for dataset_idx in range(len(raw_datasets)):
                d = raw_datasets[dataset_idx]
                assert all(x["dataset_idxs"] == dataset_idx for x in d)
                d = clm_process(
                    raw_dataset=d,
                    tokenizer=tokenizer,
                    text_column_name=trainer.config.data.dataset.text_column_name,
                    dataset_processing_num_proc_per_process=trainer.config.data.dataset.dataset_processing_num_proc_per_process,
                    dataset_overwrite_cache=trainer.config.data.dataset.dataset_overwrite_cache,
                    sequence_length=trainer.sequence_length,
                    dataset_idx=dataset_idx,
                )
                d = d.with_format(type="numpy", columns=["input_ids"], output_all_columns=True)
                tokenized_datasets.append(d)

            assert all(x["dataset_idxs"] == i for i, d in enumerate(tokenized_datasets) for x in d)
            weights = torch.tensor(list(trainer.config.data.dataset.hf_dataset_or_datasets.values()))
            train_dataset = CombinedDataset(tokenized_datasets, weights, trainer.config.data.seed)

            # We load the processed dataset on the ranks requiring it
            dataloader = get_train_dataloader(
                train_dataset=train_dataset,
                sequence_length=trainer.sequence_length,
                parallel_context=trainer.parallel_context,
                input_pp_rank=input_pp_rank,
                output_pp_rank=output_pp_rank,
                micro_batch_size=trainer.micro_batch_size,
                consumed_train_samples=trainer.consumed_train_samples,
                dataloader_num_workers=trainer.config.data.num_loading_workers,
                seed_worker=trainer.config.data.seed,
                dataloader_drop_last=True,
                weights=weights,
                num_microbatches=trainer.n_micro_batches_per_batch,
            )
            # Check if we have enough samples for train_steps
            total_tokens_dataset = len(dataloader.dataset) * trainer.sequence_length
            num_tokens_needed_for_training = (
                (trainer.config.tokens.train_steps - trainer.start_iteration_step)
                * trainer.global_batch_size
                * trainer.sequence_length
            )
            assert num_tokens_needed_for_training <= total_tokens_dataset, (
                f"Dataset is too small for steps ({total_tokens_dataset} < {num_tokens_needed_for_training}), "
                f"Try train_steps<={len(dataloader.dataset) // trainer.global_batch_size + trainer.start_iteration_step}"
            )
    else:
        raise ValueError(f"Unhandled case of `self.config.data.dataset`. Got: {trainer.config.data.dataset}")

    return dataloader
