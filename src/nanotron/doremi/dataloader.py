import dataclasses
import math
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import PretrainDatasetsArgs
from nanotron.dataloader import SkipBatchSampler, get_dataloader_worker_init
from nanotron.doremi.doremi_context import DoReMiContext
from nanotron.logging import log_rank
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

try:
    from datasets import (
        Dataset,
        DatasetDict,
        Features,
        Sequence,
        Value,
        concatenate_datasets,
        load_dataset,
        load_from_disk,
    )
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
    from transformers import __version__ as tf_version

    # from transformers.trainer_pt_utils import DistributedSamplerWithLoop
except ImportError:
    warnings.warn("Datasets and/or Transformers not installed, you'll be unable to use the dataloader.")


logger = logging.get_logger(__name__)


def get_doremi_datasets(
    hf_dataset: str,
    domain_keys: List[str],
    splits: Optional[Union[List[str], str]] = ["train", "test"],
) -> List[DatasetDict]:
    if isinstance(splits, str):
        splits = [splits]

    raw_datasets = DatasetDict()

    # NOTE: only for the pile splitted
    # DOMAIN_KEYS = [
    #     'Wikipedia (en)',
    #     'ArXiv', 'Github', 'StackExchange', 'DM Mathematics', 'PubMed Abstracts'
    # ]
    # from datasets.features import Sequence, ClassLabel, Value
    # features = Features({
    #     'text': Value("string"),
    #     'meta': {
    #         "pile_set_name": Value("string")
    #     },
    #     "domain": ClassLabel(names=DOMAIN_KEYS)
    # })

    for split in splits:
        raw_datasets[split] = []
        for domain_key in domain_keys:
            d = load_dataset(
                hf_dataset,
                domain_key,
                split=split,
                # TODO: set this in config
                # num_proc=50,
                # download_mode="force_redownload"
                # features=features
            )
            raw_datasets[split].append(d)

    return raw_datasets


def doremi_clm_process(
    domain_idx: int,
    raw_dataset: "Dataset",
    tokenizer: "PreTrainedTokenizerBase",
    text_column_name: str,
    dataset_processing_num_proc_per_process: int,
    dataset_overwrite_cache: bool,
    sequence_length: int,
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
        result["domain_ids"] = [domain_idx] * len(result[next(iter(result.keys()))])
        return result

    def _tokenize_and_group_texts(texts: List[str]) -> Dict[str, List[np.ndarray]]:
        tokenized_batch = tokenizer.batch_encode_plus(texts, return_attention_mask=False, return_token_type_ids=False)
        tokenized_batch = {k: [np.array(tokenized_texts) for tokenized_texts in v] for k, v in tokenized_batch.items()}
        return group_texts(tokenized_batch)

    train_dataset = raw_dataset.map(
        _tokenize_and_group_texts,
        input_columns=text_column_name,
        remove_columns=raw_dataset.column_names,
        features=Features(
            {
                "input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length + 1),
                "domain_ids": Value(dtype="int64"),
            }
        ),
        batched=True,
        num_proc=dataset_processing_num_proc_per_process,
        load_from_cache_file=not dataset_overwrite_cache,
        desc=f"Grouping texts in chunks of {sequence_length+1}",
    )
    return train_dataset


def get_dataloader(
    trainer: DistributedTrainer, domain_keys: List[str], tokenized_datasets: Optional[List[Dataset]] = None
) -> DataLoader:
    """Returns a dataloader for training."""
    assert isinstance(trainer.config.data.dataset, PretrainDatasetsArgs), "Please provide a dataset in the config file"

    if tokenized_datasets is None:
        log_rank("Using `datasets` library", logger=logger, level=logging.INFO, rank=0)

        tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
        log_rank(
            f"Loading tokenizer from {tokenizer_path} and transformers/hf_hub versions {tf_version, hf_hub_version}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        log_rank(
            f"Downloading dataset {trainer.config.data.dataset.hf_dataset_or_datasets}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        raw_datasets = get_doremi_datasets(
            hf_dataset=trainer.config.data.dataset.hf_dataset_or_datasets,
            domain_keys=domain_keys,
            splits=trainer.config.data.dataset.hf_dataset_splits,
        )["train"]

        train_datasets = []
        for domain_idx, raw_dataset in enumerate(raw_datasets):
            train_datasets.append(
                doremi_clm_process(
                    domain_idx=domain_idx,
                    raw_dataset=raw_dataset,
                    tokenizer=tokenizer,
                    text_column_name=trainer.config.data.dataset.text_column_name,
                    dataset_processing_num_proc_per_process=trainer.config.data.dataset.dataset_processing_num_proc_per_process,
                    dataset_overwrite_cache=trainer.config.data.dataset.dataset_overwrite_cache,
                    sequence_length=trainer.sequence_length,
                )
            )
    else:
        train_datasets = []
        for dataset_path in tqdm(tokenized_datasets, desc="Loading tokenized dataset from disk"):
            d = load_from_disk(dataset_path)
            train_datasets.append(d)

    assert 1 == 1

    # NOTE: We load the processed dataset on the ranks requiring it
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)
    doremi_context = trainer.doremi_context
    dataloader = get_doremi_dataloader(
        doremi_context=doremi_context,
        train_datasets=train_datasets,
        ref_model=trainer.ref_model if doremi_context.is_proxy is True else None,
        sequence_length=trainer.sequence_length,
        parallel_context=trainer.parallel_context,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        micro_batch_size=trainer.micro_batch_size,
        num_microbatches=trainer.n_micro_batches_per_batch,
        consumed_train_samples=trainer.consumed_train_samples,
        dataloader_num_workers=trainer.config.data.num_loading_workers,
        seed_worker=trainer.config.data.seed,
        dataloader_drop_last=True,
    )
    # NOTE: we need to call the dataloader to generate reference losses
    # if the model is a proxy model
    dataloader = dataloader() if doremi_context.is_proxy is True else dataloader

    # NOTE: Check if we have enough samples for train_steps
    # batch_size = trainer.micro_batch_size
    # assert (
    #     trainer.config.tokens.train_steps - trainer.start_iteration_step
    # ) * trainer.global_batch_size // trainer.parallel_context.dp_pg.size() < batch_size, (
    #     f"Dataset is too small for steps ({batch_size} < {(trainer.config.tokens.train_steps - trainer.start_iteration_step) * trainer.global_batch_size // trainer.parallel_context.dp_pg.size()}), "
    #     f"Try train_steps<={batch_size * trainer.parallel_context.dp_pg.size() // trainer.global_batch_size + trainer.start_iteration_step}"
    # )
    return dataloader


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
    doremi_context: DoReMiContext

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

        assert all(list(example.keys()) == ["input_ids", "domain_ids"] for example in examples)

        input_ids = np.vstack([examples[i]["input_ids"] for i in range(len(examples))])  # (b, s)
        batch_size, expanded_input_length = input_ids.shape

        result: Dict[str, Union[np.ndarray, TensorPointer]] = {}

        result["input_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["input_mask"] = TensorPointer(group_rank=self.input_pp_rank)
        result["label_ids"] = TensorPointer(group_rank=self.output_pp_rank)
        result["label_mask"] = TensorPointer(group_rank=self.output_pp_rank)

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

            # NOTE: only the last pipeline stage needs domain_idxs for computing DoReMi loss
            # and only the proxy model needs domain_idxs for computing reference loss
            if self.doremi_context.is_proxy is True:
                result["domain_idxs"] = np.vstack([examples[i]["domain_ids"] for i in range(len(examples))])

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


# class DistributedSamplerForDoReMi(DistributedSampler):
#     def __init__(
#         self,
#         datasets: List[Dataset],
#         batch_size: int,
#         num_microbatches: int,
#         shuffle: bool = False,
#         seed: int = 42,
#         doremi_context: Optional[DoReMiContext] = None,
#         parallel_context: Optional[ParallelContext] = None,
#         **kwargs,
#     ):
#         assert len(datasets) == len(
#             doremi_context.domain_weights
#         ), "The number of datasets must equal to the number of domain weights"
#         assert doremi_context is not None
#         assert parallel_context is not None

#         super().__init__(datasets, **kwargs)

#         self.datasets = datasets
#         self.batch_size = batch_size
#         self.num_microbatches = num_microbatches
#         self.shuffle = shuffle
#         self.doremi_context = doremi_context
#         self.parallel_context = parallel_context
#         self.total_size = self._calculate_total_size()

#         self.lengths = [len(d) for d in self.datasets]
#         self.offsets = np.cumsum([0] + self.lengths[:-1])
#         self.seed = seed

#         dp_size = dist.get_world_size(self.parallel_context.dp_pg)
#         self.global_batch_size = batch_size * dp_size * num_microbatches
#         # TODO(xrsrke): make seed be configureable
#         # Reset the seed of the generator for consistent randomness across epochs
#         self.generator = torch.Generator(device="cpu").manual_seed(
#             seed * (1 + dist.get_rank(self.parallel_context.dp_pg)) * (1 + dist.get_rank(self.parallel_context.pp_pg))
#         )

#         self.update_step = 0
#         self.reset()

#     def _calculate_total_size(self):
#         total_samples = sum(len(d) for d in self.datasets)
#         return math.ceil(total_samples / self.batch_size) * self.batch_size

#     def _round_up_if_fractional_part_greater_than_threshold(self, number: float, threshold=0.0000001):
#         import math

#         fractional_part = number - int(number)
#         return math.ceil(number) if fractional_part > threshold else int(number)

#     def __iter__(self):
#         domain_indices = []
#         domain_weights = self.doremi_context.domain_weights
#         print("------------------ \n")
#         dist.barrier()
#         for i, dataset in enumerate(self.datasets):
#             dataset_partition_size = len(dataset) // self.num_replicas
#             # num_samples = self._round_up_if_fractional_part_greater_than_threshold(dataset_partition_size * domain_weights[i].item())
#             num_samples = round(dataset_partition_size * domain_weights[i].item())
#             start_offset_idx = self.rank * num_samples
#             end_offset_idx = start_offset_idx + num_samples

#             # local_indices = torch.randint(
#             #     low=start_offset_idx, high=end_offset_idx, size=(num_samples,), generator=self.generator, device="cpu"
#             # ).tolist()
#             local_indices = torch.arange(start_offset_idx, end_offset_idx, device="cpu").tolist()

#             # NOTE: align the indicies across the combined dataset
#             global_indices = local_indices + self.offsets[i]
#             domain_indices.append(global_indices)

#         # print(f"rank: {self.rank}, domain_indices: {domain_indices} \n")

#         # NOTE: this one is correct
#         # total_domain_idxs = torch.tensor(sum([len(d) for d in domain_indices]), dtype=torch.int, device="cuda")
#         # dist.all_reduce(total_domain_idxs, op=dist.ReduceOp.SUM)
#         # assert 1 == 1

#         # NOTE: in some cases, the weight of a domain is too small
#         # so with a small batch size like 64, the number of samples based on the weight
#         # would be smaller than 1 => no samples from that domain
#         num_samples_per_replicas = self.batch_size * self.num_microbatches
#         # domain_batch_sizes = [self._round_up_if_fractional_part_greater_than_threshold(num_samples_per_replicas * weight.item()) for weight in domain_weights]
#         domain_batch_sizes = [round(num_samples_per_replicas * weight.item()) for weight in domain_weights]
#         if sum(domain_batch_sizes) != num_samples_per_replicas:
#             # NOTE: randomly add a sample to round it up
#             domain_batch_sizes = self._round_up_domain_batch_sizes(
#                 domain_batch_sizes,
#                 target_total_size=num_samples_per_replicas,
#             )

#         # TODO(xrsrke): cache this
#         assert sum(domain_batch_sizes) == num_samples_per_replicas
#         # print(f"rank: {self.rank}, domain_batch_sizes after rounding: {domain_batch_sizes} \n")

#         microbatch_idx = 0
#         dp_size = dist.get_world_size(self.parallel_context.dp_pg)

#         while self.total_samples_yielded < self.total_size:
#             batch = []
#             # NOTE: Flag to indicate if a domain is out of samples
#             out_of_samples = False

#             # sample_per_domain_loggins = []
#             for domain_index, (idxs, domain_batch_size) in enumerate(zip(domain_indices, domain_batch_sizes)):
#                 start_idx = self.domain_counters[domain_index]
#                 end_idx = start_idx + domain_batch_size

#                 # NOTE: a domain run out of samples
#                 if end_idx > len(idxs):
#                     out_of_samples = True
#                     break

#                 # NOTE: if the current microbatch is the last one
#                 # then after yielding the samples, we need to update
#                 # the domain counter
#                 if microbatch_idx == self.num_microbatches - 1:
#                     dist.barrier()
#                     print(f"rank: {self.rank}, domain_index: {domain_index}, microbatch_idx={microbatch_idx}, now update domain counter to {end_idx} \n")
#                     self.domain_counters[domain_index] = end_idx

#                 # NOTE: if the current microbatch is more than
#                 # the number of microbatches, then we need to
#                 # to reset the microbatch index
#                 # if microbatch_idx == self.num_microbatches:
#                 #     dist.barrier()
#                 #     print(f"rank: {self.rank}, domain_index: {domain_index}, microbatch_idx={microbatch_idx}, now reset to 0 \n")
#                 #     microbatch_idx = 0
#                 #     # self.domain_counters[domain_index] = end_idx

#                 dist.barrier()
#                 print(
#                     f"rank: {self.rank}, domain_index: {domain_index}, microbatch_idx: {microbatch_idx}, start_idx={start_idx}, end_idx={end_idx} \n"
#                 )

#                 global_batch_idxs = idxs[start_idx:end_idx]
#                 # sample_per_domain_loggins.append(len(global_batch_idxs))
#                 batch.extend(global_batch_idxs)

#             # NOTE: stop if either one of the domains are
#             # out of sample or the batch is empty
#             if out_of_samples or len(batch) == 0:
#                 break

#             assert len(batch) == self.num_microbatches * self.batch_size

#             microbatch_start_idx = microbatch_idx * self.batch_size
#             microbatch_end_idx = microbatch_start_idx + self.batch_size

#             assert microbatch_end_idx <= len(batch)
#             microbatch_idxs = batch[microbatch_start_idx:microbatch_end_idx]

#             dist.barrier()
#             print(
#                 f"rank: {self.rank}, microbatch_idx: {microbatch_idx}, microbatch_start_idx: {microbatch_start_idx}, microbatch_end_idx: {microbatch_end_idx} \n"
#             )
#             # print(f"rank: {self.rank}, yield microbatch_idxs: {microbatch_idxs} \n")
#             self.total_samples_yielded += len(microbatch_idxs) * dp_size
#             microbatch_idx += 1

#             yield microbatch_idxs

#             if microbatch_idx == self.num_microbatches:
#                 dist.barrier()
#                 print(f"rank: {self.rank}, domain_index: {domain_index}, microbatch_idx={microbatch_idx}, now reset to 0 \n")
#                 microbatch_idx = 0

#             # NOTE: once a microbatch is yielded
#             # that means that same microbatch is yielded
#             # across all dp ranks

#             # if microbatch_idx == self.num_microbatches:
#             #     _logs = {
#             #         f"domain_{self.doremi_context.get_domain_name(i)}": v
#             #         for i, v in enumerate(sample_per_domain_loggins)
#             #     }
#             #     log_rank(
#             #         f"Samples per domain: {_logs}",
#             #         logger=logger,
#             #         level=logging.INFO,
#             #         rank=0,
#             #         group=self.parallel_context.tp_pg,
#             #     )

#             #     microbatch_idx = 0

#     def _round_up_domain_batch_sizes(self, domain_batch_size: List[int], target_total_size: int) -> List[int]:
#         """
#         NOTE: Make sum(domain_batch_sizes) == batch_size
#         """
#         total_batch_size = sum(domain_batch_size)
#         while total_batch_size != target_total_size:
#             diff = target_total_size - total_batch_size
#             # NOTE: Randomly select a domain to increase the batch size
#             selected_domain = torch.randint(
#                 low=0, high=len(domain_batch_size), size=(1,), generator=self.generator, device="cpu"
#             ).item()

#             if diff > 0:
#                 domain_batch_size[selected_domain] += 1
#             elif diff < 0 and domain_batch_size[selected_domain] > 0:
#                 domain_batch_size[selected_domain] -= 1

#             total_batch_size = sum(domain_batch_size)

#         return domain_batch_size

#     def reset(self):
#         """Reset the state of the sampler for a new epoch."""
#         self.domain_counters = [0 for _ in self.datasets]
#         self.total_samples_yielded = 0

#         if self.update_step > 0:
#             self.update_step += 1


# NOTE: #2
# class DistributedSamplerForDoReMi(DistributedSampler):
#     def __init__(
#         self,
#         datasets: List[Dataset],
#         batch_size: int,
#         num_microbatches: int,
#         shuffle: bool = False,
#         seed: int = 42,
#         doremi_context: Optional[DoReMiContext] = None,
#         parallel_context: Optional[ParallelContext] = None,
#         **kwargs,
#     ):
#         assert len(datasets) == len(
#             doremi_context.domain_weights
#         ), "The number of datasets must equal to the number of domain weights"
#         assert doremi_context is not None
#         assert parallel_context is not None

#         super().__init__(datasets, **kwargs)

#         self.datasets = datasets
#         self.batch_size = batch_size
#         self.num_microbatches = num_microbatches
#         self.shuffle = shuffle
#         self.doremi_context = doremi_context
#         self.parallel_context = parallel_context
#         self.total_size = self._calculate_total_size()

#         self.lengths = [len(d) for d in self.datasets]
#         self.offsets = np.cumsum([0] + self.lengths[:-1])
#         self.seed = seed

#         dp_size = dist.get_world_size(self.parallel_context.dp_pg)
#         self.global_batch_size = batch_size * dp_size * num_microbatches
#         # TODO(xrsrke): make seed be configureable
#         # Reset the seed of the generator for consistent randomness across epochs
#         self.generator = torch.Generator(device="cpu").manual_seed(
#             seed * (1 + dist.get_rank(self.parallel_context.dp_pg)) * (1 + dist.get_rank(self.parallel_context.pp_pg))
#         )

#         self.update_step = 0
#         self.reset()

#     def _calculate_total_size(self):
#         total_samples = sum(len(d) for d in self.datasets)
#         return math.ceil(total_samples / self.batch_size) * self.batch_size

#     def _round_up_if_fractional_part_greater_than_threshold(self, number: float, threshold=0.0000001):
#         import math

#         fractional_part = number - int(number)
#         return math.ceil(number) if fractional_part > threshold else int(number)

#     def __iter__(self):
#         domain_indices = []
#         domain_weights = self.doremi_context.domain_weights
#         # print("------------------ \n")
#         # dist.barrier()
#         for i, dataset in enumerate(self.datasets):
#             dataset_partition_size = len(dataset) // self.num_replicas
#             # num_samples = self._round_up_if_fractional_part_greater_than_threshold(dataset_partition_size * domain_weights[i].item())
#             start_offset_idx = self.rank * dataset_partition_size
#             end_offset_idx = start_offset_idx + dataset_partition_size
#             local_indices = torch.arange(start_offset_idx, end_offset_idx, device="cpu").tolist()

#             # NOTE: align the indicies across the combined dataset
#             global_indices = local_indices + self.offsets[i]
#             domain_indices.append(global_indices)

#         # NOTE: in some cases, the weight of a domain is too small
#         # so with a small batch size like 64, the number of samples based on the weight
#         # would be smaller than 1 => no samples from that domain
#         num_samples_per_replicas = self.batch_size * self.num_microbatches
#         domain_batch_sizes = [round(num_samples_per_replicas * weight.item()) for weight in domain_weights]
#         if sum(domain_batch_sizes) != num_samples_per_replicas:
#             # NOTE: randomly add a sample to round it up
#             domain_batch_sizes = self._round_up_domain_batch_sizes(
#                 domain_batch_sizes,
#                 target_total_size=num_samples_per_replicas,
#             )

#         assert all([x > 0 for x in domain_batch_sizes]), "There is a domain with 0 samples per global batch"

#         microbatch_idx = 0
#         out_of_samples = False
#         # dist.get_world_size(self.parallel_context.dp_pg)
#         # dist.barrier()
#         # expected_total_samples = sum(
#         #     [round(len(ds) * weight.item()) for ds, weight in zip(self.datasets, domain_weights)]
#         # )
#         # total_sampels = sum([len(d) for d in domain_indices])
#         expected_total_samples = sum(
#             [round(len(d) * weight.item()) for d, weight in zip(domain_indices, domain_weights)]
#         )

#         while self.total_samples_yielded < expected_total_samples:
#             batch = []
#             # dist.barrier()

#             for domain_index, (idxs, domain_batch_size) in enumerate(zip(domain_indices, domain_batch_sizes)):
#                 start_idx = self.domain_counters[domain_index]
#                 end_idx = start_idx + domain_batch_size
#                 # dist.barrier()

#                 # NOTE: BREAK 1
#                 if end_idx > len(idxs) or start_idx >= len(idxs):
#                     out_of_samples = True
#                     print(f"rank: {self.rank}, break1, end_idx: {end_idx}, start_idx: {start_idx}, len(idxs): {len(idxs)} \
#                         domain_batch_sizes: {domain_batch_sizes}, \
#                         domain_counters: {self.domain_counters}, domain_batch_size: {domain_batch_size} \
#                         microbatch_idx: {microbatch_idx}, domain_index: {domain_index}, total_samples_yielded: {self.total_samples_yielded} \
#                             expected_total_samples: {expected_total_samples} \
#                     ")
#                     break

#                 if microbatch_idx == self.num_microbatches - 1:
#                     # dist.barrier()
#                     # print(
#                     #     f"rank: {self.rank}, domain_index: {domain_index}, microbatch_idx={microbatch_idx}, now update domain counter to {end_idx} \n"
#                     # )
#                     self.domain_counters[domain_index] = end_idx
#                     # dist.barrier()

#                 # NOTE: this contains the idxs portion for num_microbatches
#                 global_batch_idxs = idxs[start_idx:end_idx]

#                 # dist.barrier()
#                 # print(
#                 #     f"rank: {self.rank}, domain_index: {domain_index}, microbatch_idx={microbatch_idx}, global_batch_idxs: {global_batch_idxs} \n"
#                 # )
#                 batch.extend(global_batch_idxs)
#                 # dist.barrier()

#             # NOTE: BREAK2
#             if out_of_samples or len(batch) == 0:
#                 print(f"rank: {self.rank}, break2, end_idx: {end_idx}, start_idx: {start_idx}, len(idxs): {len(idxs)} \
#                     domain_counters: {self.domain_counters}, domain_batch_size: {domain_batch_size} \
#                     domain_batch_sizes: {domain_batch_sizes}, \
#                     microbatch_idx: {microbatch_idx}, domain_index: {domain_index}, total_samples_yielded: {self.total_samples_yielded} \
#                         expected_total_samples: {expected_total_samples} \
#                     out_of_samples: {out_of_samples}, len(batch): {len(batch)} \
#                 ")

#                 break

#             # dist.barrier()
#             assert len(batch) == self.num_microbatches * self.batch_size

#             microbatch_start_idx = microbatch_idx * self.batch_size
#             microbatch_end_idx = microbatch_start_idx + self.batch_size

#             assert microbatch_end_idx <= len(batch)

#             # dist.barrier()
#             # print(
#             #     f"rank: {self.rank}, microbatch_idx: {microbatch_idx}, microbatch_start_idx: {microbatch_start_idx}, microbatch_end_idx: {microbatch_end_idx} \n"
#             # )
#             microbatch_idxs = batch[microbatch_start_idx:microbatch_end_idx]

#             # dist.barrier()
#             if microbatch_idx == self.num_microbatches - 1:
#                 microbatch_idx = 0
#             else:
#                 microbatch_idx += 1

#             # self.total_samples_yielded += len(microbatch_idxs) * dp_size
#             self.total_samples_yielded += len(microbatch_idxs)

#             # dist.barrier()
#             # print(f"rank: {self.rank}, microbatch_idx: {microbatch_idx}, yield microbatch_idxs: {microbatch_idxs} \n")
#             yield microbatch_idxs

#         #     dist.barrier()

#         # dist.barrier()

#     def _round_up_domain_batch_sizes(self, domain_batch_size: List[int], target_total_size: int) -> List[int]:
#         """
#         NOTE: Make sum(domain_batch_sizes) == batch_size
#         """
#         total_batch_size = sum(domain_batch_size)
#         while total_batch_size != target_total_size:
#             diff = target_total_size - total_batch_size
#             # NOTE: Randomly select a domain to increase the batch size
#             selected_domain = torch.randint(
#                 low=0, high=len(domain_batch_size), size=(1,), generator=self.generator, device="cpu"
#             ).item()

#             if diff > 0:
#                 domain_batch_size[selected_domain] += 1
#             elif diff < 0 and domain_batch_size[selected_domain] > 0:
#                 domain_batch_size[selected_domain] -= 1

#             total_batch_size = sum(domain_batch_size)

#         return domain_batch_size

#     def reset(self):
#         """Reset the state of the sampler for a new epoch."""
#         self.domain_counters = [0 for _ in self.datasets]
#         self.total_samples_yielded = 0

#         if self.update_step > 0:
#             self.update_step += 1


class DistributedSamplerForDoReMi(DistributedSampler):
    def __init__(
        self,
        datasets: List[Dataset],
        batch_size: int,
        num_microbatches: int,
        shuffle: bool = False,
        seed: int = 42,
        doremi_context: Optional[DoReMiContext] = None,
        parallel_context: Optional[ParallelContext] = None,
        **kwargs,
    ):
        assert len(datasets) == len(
            doremi_context.domain_weights
        ), "The number of datasets must equal to the number of domain weights"
        assert doremi_context is not None
        assert parallel_context is not None

        super().__init__(datasets, **kwargs)

        self.datasets = datasets
        self.batch_size = batch_size
        self.num_microbatches = num_microbatches
        self.shuffle = shuffle
        self.doremi_context = doremi_context
        self.parallel_context = parallel_context
        self.total_size = self._calculate_total_size()

        self.lengths = [len(d) for d in self.datasets]
        self.offsets = np.cumsum([0] + self.lengths[:-1])
        self.seed = seed

        dp_size = dist.get_world_size(self.parallel_context.dp_pg)
        self.global_batch_size = batch_size * dp_size * num_microbatches
        # TODO(xrsrke): make seed be configureable
        # Reset the seed of the generator for consistent randomness across epochs
        self.generator = torch.Generator(device="cpu").manual_seed(
            seed * (1 + dist.get_rank(self.parallel_context.dp_pg)) * (1 + dist.get_rank(self.parallel_context.pp_pg))
        )

        # self.update_step = 0
        self.reset()
        self.setup()

    def _calculate_total_size(self):
        total_samples = sum(len(d) for d in self.datasets)
        return math.ceil(total_samples / self.batch_size) * self.batch_size

    def _round_up_if_fractional_part_greater_than_threshold(self, number: float, threshold=0.0000001):
        import math

        fractional_part = number - int(number)
        return math.ceil(number) if fractional_part > threshold else int(number)

    # def __iter__(self):
    #     domain_indices = []
    #     domain_weights = self.doremi_context.domain_weights
    #     # print("------------------ \n")
    #     # dist.barrier()
    #     for i, dataset in enumerate(self.datasets):
    #         # dataset_partition_size = len(dataset) // self.num_replicas
    #         # num_samples = self._round_up_if_fractional_part_greater_than_threshold(dataset_partition_size * domain_weights[i].item())
    #         # start_offset_idx = self.rank * dataset_partition_size
    #         # end_offset_idx = start_offset_idx + dataset_partition_size
    #         # local_indices = torch.arange(start_offset_idx, end_offset_idx, device="cpu").tolist()
    #         local_indices = torch.arange(0, len(dataset), device="cpu").tolist()

    #         # NOTE: align the indicies across the combined dataset
    #         global_indices = local_indices + self.offsets[i]
    #         domain_indices.append(global_indices)

    #     # NOTE: in some cases, the weight of a domain is too small
    #     # so with a small batch size like 64, the number of samples based on the weight
    #     # would be smaller than 1 => no samples from that domain
    #     # num_samples_per_replicas = self.batch_size * self.num_microbatches
    #     # domain_batch_sizes = [round(num_samples_per_replicas * weight.item()) for weight in domain_weights]
    #     # if sum(domain_batch_sizes) != num_samples_per_replicas:
    #     #     # NOTE: randomly add a sample to round it up
    #     #     domain_batch_sizes = self._round_up_domain_batch_sizes(
    #     #         domain_batch_sizes,
    #     #         target_total_size=num_samples_per_replicas,
    #     #     )

    #     num_samples_per_global_step = self.batch_size * self.num_microbatches * self.num_replicas
    #     domain_batch_sizes = [round(num_samples_per_global_step * weight.item()) for weight in domain_weights]
    #     if sum(domain_batch_sizes) != num_samples_per_global_step:
    #         # NOTE: randomly add a sample to round it up
    #         domain_batch_sizes = self._round_up_domain_batch_sizes(
    #             domain_batch_sizes,
    #             target_total_size=num_samples_per_global_step,
    #         )

    #     assert all(x > 0 for x in domain_batch_sizes), "There is a domain with 0 samples per global batch"
    #     self.domain_batch_sizes = domain_batch_sizes
    #     self.domain_indices = domain_indices
    #     self.expected_total_samples = sum([len(d) for d in domain_indices])
    #     return self

    def setup(self):
        domain_indices = []
        for i, dataset in enumerate(self.datasets):
            # dataset_partition_size = len(dataset) // self.num_replicas
            # num_samples = self._round_up_if_fractional_part_greater_than_threshold(dataset_partition_size * domain_weights[i].item())
            # start_offset_idx = self.rank * dataset_partition_size
            # end_offset_idx = start_offset_idx + dataset_partition_size
            # local_indices = torch.arange(start_offset_idx, end_offset_idx, device="cpu").tolist()
            local_indices = torch.arange(0, len(dataset), device="cpu").tolist()

            # NOTE: align the indicies across the combined dataset
            global_indices = local_indices + self.offsets[i]
            domain_indices.append(global_indices)

        self.num_samples_per_global_step = self.batch_size * self.num_microbatches * self.num_replicas
        self.domain_indices = domain_indices
        self.expected_total_samples = sum([len(d) for d in domain_indices])

        # print("------------------ \n")
        # dist.barrier()

        # NOTE: in some cases, the weight of a domain is too small
        # so with a small batch size like 64, the number of samples based on the weight
        # would be smaller than 1 => no samples from that domain
        # num_samples_per_replicas = self.batch_size * self.num_microbatches
        # domain_batch_sizes = [round(num_samples_per_replicas * weight.item()) for weight in domain_weights]
        # if sum(domain_batch_sizes) != num_samples_per_replicas:
        #     # NOTE: randomly add a sample to round it up
        #     domain_batch_sizes = self._round_up_domain_batch_sizes(
        #         domain_batch_sizes,
        #         target_total_size=num_samples_per_replicas,
        #     )
        # self._recompute_domain_batch_sizes(
        #     domain_weights=self.doremi_context.domain_weights,
        #     num_samples_per_global_step=self.num_samples_per_global_step,
        # )
        return self

    def __iter__(self):
        return self

    def _recompute_domain_batch_sizes(self, domain_weights, num_samples_per_global_step):
        domain_batch_sizes = [round(num_samples_per_global_step * weight.item()) for weight in domain_weights]
        if sum(domain_batch_sizes) != num_samples_per_global_step:
            # NOTE: randomly add a sample to round it up
            domain_batch_sizes = self._round_up_domain_batch_sizes(
                domain_batch_sizes,
                target_total_size=num_samples_per_global_step,
            )

        # assert all(x > 0 for x in domain_batch_sizes), "There is a domain with 0 samples per global batch"
        return domain_batch_sizes

    def __next__(self):
        # microbatch_idx = 0
        # dist.get_world_size(self.parallel_context.dp_pg)
        # dist.barrier()
        # expected_total_samples = sum(
        #     [round(len(ds) * weight.item()) for ds, weight in zip(self.datasets, domain_weights)]
        # )
        # total_sampels = sum([len(d) for d in domain_indices])
        # expected_total_samples = sum(
        #     [round(len(d) * weight.item()) for d, weight in zip(domain_indices, domain_weights)]
        # )
        # domain_weights = self.doremi_context.domain_weights
        domain_batch_sizes = self._recompute_domain_batch_sizes(
            domain_weights=self.doremi_context.domain_weights,
            num_samples_per_global_step=self.num_samples_per_global_step,
        )

        if self.total_samples_yielded >= self.expected_total_samples:
            raise StopIteration

        batch = []
        for domain_index, (idxs, domain_batch_size) in enumerate(zip(self.domain_indices, domain_batch_sizes)):
            start_idx = self.domain_counters[domain_index]
            end_idx = start_idx + domain_batch_size
            # dist.barrier()

            # NOTE: BREAK 1
            if end_idx > len(idxs):
                # self.out_of_samples = True
                print(
                    f"rank: {self.rank}, break1, end_idx: {end_idx}, start_idx: {start_idx}, len(idxs): {len(idxs)} \
                    domain_batch_sizes: {domain_batch_sizes}, \
                    domain_counters: {self.domain_counters}, domain_batch_size: {domain_batch_size} \
                    microbatch_idx: {self.microbatch_idx}, domain_index: {domain_index}, total_samples_yielded: {self.total_samples_yielded} \
                        expected_total_samples: {self.expected_total_samples} \
                "
                )
                raise StopIteration

            if self.microbatch_idx == self.num_microbatches - 1:
                # dist.barrier()
                # print(
                #     f"rank: {self.rank}, domain_index: {domain_index}, microbatch_idx={microbatch_idx}, now update domain counter to {end_idx} \n"
                # )
                self.domain_counters[domain_index] = end_idx
                # dist.barrier()

            # NOTE: this contains the idxs portion for num_microbatches
            global_batch_idxs = idxs[start_idx:end_idx]

            # dist.barrier()
            # print(
            #     f"rank: {self.rank}, domain_index: {domain_index}, microbatch_idx={microbatch_idx}, global_batch_idxs: {global_batch_idxs} \n"
            # )
            batch.extend(global_batch_idxs)
            # dist.barrier()

        # if len(batch) == 0:
        #     print(
        #         f"rank: {self.rank}, break2, end_idx: {end_idx}, start_idx: {start_idx}, len(idxs): {len(idxs)} \
        #         domain_counters: {self.domain_counters}, domain_batch_size: {domain_batch_size} \
        #         domain_batch_sizes: {self.domain_batch_sizes}, \
        #         microbatch_idx: {self.microbatch_idx}, domain_index: {domain_index}, total_samples_yielded: {self.total_samples_yielded} \
        #             expected_total_samples: {self.expected_total_samples} \
        #         out_of_samples: {self.out_of_samples}, len(batch): {len(batch)} \
        #     "
        #     )

        #     raise StopIteration

        assert len(batch) == self.num_microbatches * self.batch_size * self.num_replicas

        # NOTE: BREAK2
        # if self.out_of_samples or len(batch) == 0:

        # dist.barrier()
        num_samples_per_dp_rank = self.batch_size * self.num_microbatches
        dp_start_idx = self.rank * num_samples_per_dp_rank
        dp_end_idx = dp_start_idx + num_samples_per_dp_rank

        # assert dp_end_idx <= len(batch)

        if dp_end_idx > len(batch):
            raise StopIteration(f"dp_end_idx > len(batch), dp_end_idx: {dp_end_idx}, len(batch): {len(batch)}")

        dp_batch = batch[dp_start_idx:dp_end_idx]

        assert len(dp_batch) == self.num_microbatches * self.batch_size

        microbatch_start_idx = self.microbatch_idx * self.batch_size
        microbatch_end_idx = microbatch_start_idx + self.batch_size

        # assert microbatch_end_idx <= len(dp_batch) -1
        if microbatch_end_idx > len(dp_batch):
            raise StopIteration(
                f"microbatch_end_idx > len(dp_batch) - 1, microbatch_end_idx: {microbatch_end_idx}, len(dp_batch): {len(dp_batch)}"
            )

        # dist.barrier()
        # print(
        #     f"rank: {self.rank}, microbatch_idx: {microbatch_idx}, microbatch_start_idx: {microbatch_start_idx}, microbatch_end_idx: {microbatch_end_idx} \n"
        # )
        microbatch_idxs = dp_batch[microbatch_start_idx:microbatch_end_idx]

        # dist.barrier()
        if self.microbatch_idx == self.num_microbatches - 1:
            self.microbatch_idx = 0
        else:
            self.microbatch_idx += 1

        # self.total_samples_yielded += len(microbatch_idxs) * dp_size
        self.total_samples_yielded += len(microbatch_idxs) * self.num_replicas

        # dist.barrier()
        # print(f"rank: {self.rank}, microbatch_idx: {microbatch_idx}, yield microbatch_idxs: {microbatch_idxs} \n")
        return microbatch_idxs

        # dist.barrier()

    def _round_up_domain_batch_sizes(self, domain_batch_size: List[int], target_total_size: int) -> List[int]:
        """
        NOTE: Make sum(domain_batch_sizes) == batch_size
        """
        total_batch_size = sum(domain_batch_size)
        while total_batch_size != target_total_size:
            diff = target_total_size - total_batch_size
            # NOTE: Randomly select a domain to increase the batch size
            selected_domain = torch.randint(
                low=0, high=len(domain_batch_size), size=(1,), generator=self.generator, device="cpu"
            ).item()

            if diff > 0:
                domain_batch_size[selected_domain] += 1
            elif diff < 0 and domain_batch_size[selected_domain] > 0:
                domain_batch_size[selected_domain] -= 1

            total_batch_size = sum(domain_batch_size)

        return domain_batch_size

    def reset(self):
        """Reset the state of the sampler for a new epoch."""
        self.setup()

        self.microbatch_idx = 0
        self.domain_counters = [0 for _ in self.datasets]
        self.total_samples_yielded = 0
        self.out_of_samples = False

        # if self.update_step > 0:
        #     self.update_step += 1


# Adapted from https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L763-L835
def _get_train_sampler(
    dp_size: int,
    dp_rank: int,
    train_datasets: "Dataset",
    seed: int,
    use_loop_to_round_batch_size: bool,
    consumed_train_samples: int,
    doremi_context: DoReMiContext,
    parallel_context: ParallelContext,
    micro_batch_size: Optional[int] = None,
    num_microbatches: Optional[int] = None,
    drop_last: Optional[bool] = True,
) -> Optional[torch.utils.data.Sampler]:
    """returns sampler that restricts data loading to a subset of the dataset proper to the DP rank"""
    assert num_microbatches is not None

    # Build the sampler.
    # TODO @nouamanetazi: Support group_by_length: https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L783-L810

    if use_loop_to_round_batch_size:
        assert micro_batch_size is not None
        # loops at the end back to the beginning of the shuffled samples to make each process have a round multiple of batch_size samples.
        # sampler = DistributedSamplerWithLoop(
        #     train_datasets,
        #     batch_size=micro_batch_size,
        #     num_replicas=dp_size,
        #     rank=dp_rank,
        #     seed=seed,
        #     drop_last=drop_last,
        # )
        raise NotImplementedError("use_loop_to_round_batch_size is not implemented yet")
    else:
        # sampler = DistributedSampler(train_dataset, num_replicas=dp_size, rank=dp_rank, seed=seed, drop_last=drop_last)
        sampler = DistributedSamplerForDoReMi(
            train_datasets,
            batch_size=micro_batch_size,
            num_microbatches=num_microbatches,
            num_replicas=dp_size,
            rank=dp_rank,
            seed=seed,
            drop_last=drop_last,
            doremi_context=doremi_context,
            parallel_context=parallel_context,
        )

    if consumed_train_samples > 0:
        sampler = SkipBatchSampler(sampler, skip_batches=consumed_train_samples, dp_size=dp_size)

    return sampler


class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.comebined_dataset = concatenate_datasets(datasets)

    def __len__(self):
        return len(self.comebined_dataset)

    def __getitem__(self, batch):
        if isinstance(batch, list) is False:
            batch = [batch]

        assert len(batch) > 0
        if isinstance(batch[0], list):

            def merge_dicts(data):
                merged = {}
                # NOTE: # Assuming all dictionaries have the same keys
                for key in data[0].keys():
                    # NOTE: Concatenating values corresponding to each key
                    merged[key] = np.concatenate([d[key] for d in data if key in d])
                return merged

            # TODO(xrsrke): do a single index, then split the output
            samples = [self.comebined_dataset[idxs] for idxs in batch]
            return merge_dicts(samples)

        return self.comebined_dataset[batch]


# Adapted from https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L837
def get_doremi_dataloader(
    doremi_context: DoReMiContext,
    ref_model: Optional[nn.Module],
    train_datasets: List["Dataset"],
    sequence_length: int,
    parallel_context: ParallelContext,
    input_pp_rank: int,
    output_pp_rank: int,
    num_microbatches: int,
    micro_batch_size: int,
    consumed_train_samples: int,
    dataloader_num_workers: int,
    seed_worker: int,
    dataloader_drop_last: bool = True,
    dataloader_pin_memory: bool = True,
    use_loop_to_round_batch_size: bool = False,
) -> DataLoader:
    # Case of ranks requiring data
    if dist.get_rank(parallel_context.pp_pg) in [
        input_pp_rank,
        output_pp_rank,
    ]:
        train_datasets = [
            d.with_format(type="numpy", columns=["input_ids"], output_all_columns=True) for d in train_datasets
        ]

    # Case of ranks not requiring data. We give them an infinite dummy dataloader
    else:
        # # TODO(xrsrke): recheck this
        # # train_datasets = train_datasets[0]
        # # assert train_dataset.column_names == ["input_ids"], (
        # #     f"Dataset has to have a single column, with `input_ids` as the column name. "
        # #     f"Current dataset: {train_dataset}"
        # # )
        # dataset_length = len(train_datasets[0])
        # train_dataset = train_datasets[0].remove_columns(column_names="input_ids")
        # assert (
        #     len(train_dataset) == 0
        # ), f"Dataset has to be empty after removing the `input_ids` column. Current dataset: {train_dataset}"
        # # HACK as if we remove the last column of a train_dataset, it becomes empty and it's number of rows becomes empty.
        # train_datasets = EmptyInfiniteDataset(length=dataset_length)
        # # No need to spawn a lot of workers, we can just use main
        # dataloader_num_workers = 0
        raise NotImplementedError("This case is not implemented yet")

    data_collator = DataCollatorForCLM(
        sequence_length=sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        parallel_context=parallel_context,
        doremi_context=doremi_context,
    )

    train_sampler = _get_train_sampler(
        dp_size=parallel_context.dp_pg.size(),
        dp_rank=dist.get_rank(parallel_context.dp_pg),
        train_datasets=train_datasets,
        seed=seed_worker,
        use_loop_to_round_batch_size=use_loop_to_round_batch_size,
        micro_batch_size=micro_batch_size,
        num_microbatches=num_microbatches,
        drop_last=dataloader_drop_last,
        consumed_train_samples=consumed_train_samples,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )

    comebined_dataset = CombinedDataset(train_datasets)
    dataloader = DataLoader(
        comebined_dataset,
        batch_size=micro_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        drop_last=dataloader_drop_last,  # we also drop_last in `clm_process()`
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dist.get_rank(parallel_context.dp_pg)),
    )

    def _data_generator():
        dist.barrier()
        for batch in dataloader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            # NOTE: because the inference model don't take `domain_idxs`
            # as input we need to remove it from the batch
            batch_for_inference = {k: v for k, v in batch.items() if k != "domain_idxs"}

            ref_losses = ref_model(**batch_for_inference)["losses"]
            batch["ref_losses"] = ref_losses
            yield batch

    return _data_generator if ref_model is not None else dataloader
