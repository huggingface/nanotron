import dataclasses
import math
import warnings
from typing import Dict, List, Union

import numpy as np
import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.data.dataloader import get_dataloader_worker_init
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .doremi_context import DoReMiContext

try:
    from datasets import Dataset, concatenate_datasets, load_from_disk
except ImportError:
    warnings.warn("Datasets and/or Transformers not installed, you'll be unable to use the dataloader.")


logger = logging.get_logger(__name__)


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
            # TODO(xrsrke): do a single index, then split the output
            samples = [self.comebined_dataset[idxs] for idxs in batch]
            return self._merge_dicts(samples)

        return self.comebined_dataset[batch]

    def _merge_dicts(self, data):
        merged = {}
        for key in data[0].keys():
            merged[key] = np.concatenate([d[key] for d in data if key in d])
        return merged


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


class DistributedSamplerForDoReMi(DistributedSampler):
    def __init__(
        self,
        datasets: List[Dataset],
        batch_size: int,
        num_microbatches: int,
        num_replicas: int,
        rank: int,
        doremi_context: DoReMiContext,
        parallel_context: ParallelContext,
        shuffle: bool = False,
        seed: int = 42,
        drop_last: bool = False,
    ):
        assert len(datasets) == len(
            doremi_context.domain_weights
        ), "The number of datasets must equal to the number of domain weights"

        super().__init__(datasets, num_replicas=num_replicas, rank=rank, shuffle=shuffle, drop_last=drop_last)

        self.datasets = datasets
        self.batch_size = batch_size
        self.num_microbatches = num_microbatches
        self.doremi_context = doremi_context
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
                domain_weights=self.doremi_context.domain_weights,
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
            domain_weights=self.doremi_context.domain_weights,
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


def get_datasets(paths):
    datasets = []
    for path in tqdm(paths, desc="Loading dataset from disk"):
        d = load_from_disk(path)
        datasets.append(d)

    return datasets


def get_dataloader(trainer: DistributedTrainer, datasets) -> DataLoader:
    doremi_context = trainer.doremi_context
    parallel_context = trainer.parallel_context

    datasets = [d.with_format(type="numpy", columns=["input_ids"], output_all_columns=True) for d in datasets]

    # TODO(xrsrke): decouple trainer from dataloader
    # TODO(xrsrke): decouple data collating from data loading
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)
    data_collator = DataCollatorForCLM(
        sequence_length=trainer.sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        parallel_context=parallel_context,
        doremi_context=doremi_context,
    )

    sampler = DistributedSamplerForDoReMi(
        datasets,
        batch_size=trainer.micro_batch_size,
        num_microbatches=trainer.n_micro_batches_per_batch,
        num_replicas=parallel_context.dp_pg.size(),
        rank=dist.get_rank(parallel_context.dp_pg),
        seed=trainer.config.data_stages[0].data.seed,
        drop_last=True,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )

    comebined_dataset = CombinedDataset(datasets)

    dataloader = DataLoader(
        comebined_dataset,
        batch_sampler=sampler,
        collate_fn=data_collator,
        num_workers=trainer.config.data_stages[0].data.num_loading_workers,
        pin_memory=True,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dist.get_rank(parallel_context.dp_pg)),
    )

    def _data_generator(dataloader):
        def inner():
            for batch in dataloader:
                batch = {k: v.to("cuda") for k, v in batch.items()}
                # NOTE: because the inference model don't take `domain_idxs`
                # as input we need to remove it from the batch
                batch_for_inference = {k: v for k, v in batch.items() if k != "domain_idxs"}

                ref_losses = trainer.ref_model(**batch_for_inference)["losses"]
                batch["ref_losses"] = ref_losses
                yield batch

        return inner

    dataloader = _data_generator(dataloader) if doremi_context.is_proxy is True else dataloader

    # NOTE: we need to call the dataloader to generate reference losses
    # if the model is a proxy model
    dataloader = dataloader() if doremi_context.is_proxy is True else dataloader

    return dataloader
