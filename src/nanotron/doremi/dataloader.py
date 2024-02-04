import dataclasses
import math
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.dataloader import get_dataloader_worker_init
from nanotron.doremi.doremi_context import DoReMiContext
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

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

            # NOTE: only the last pipeline stage needs domain_idxs for computing DoReMi loss
            # and only the proxy model needs domain_idxs for computing reference loss
            # if self.doremi_context.is_proxy is True:
            #     result["domain_idxs"] = np.vstack([examples[i]["domain_ids"] for i in range(len(examples))])
            # TODO(xrsrke): use the default one, then add domain_ids, don't duplicate code!
            # result["domain_idxs"] = np.vstack([examples[i]["domain_ids"] for i in range(len(examples))])

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

        self.reset()

        # self.debug_history = []

    def _calculate_total_size(self):
        total_samples = sum(len(d) for d in self.datasets)
        return math.ceil(total_samples / self.batch_size) * self.batch_size

    # def _round_up_if_fractional_part_greater_than_threshold(self, number: float, threshold=0.0000001):
    #     import math

    #     fractional_part = number - int(number)
    #     return math.ceil(number) if fractional_part > threshold else int(number)

    def __iter__(self):
        return self

    def _recompute_domain_batch_sizes(self, domain_weights, num_samples_per_global_step):
        domain_batch_sizes = [round(num_samples_per_global_step * weight.item()) for weight in domain_weights]

        # NOTE: in some cases, the weight of a domain is too small
        # resulting in a domain with 0 samples per global batch
        # => zero loss for that domain => we no longer update the weights of that domain
        # so we add a sample to that domain
        domain_batch_sizes = [1 if x == 0 else x for x in domain_batch_sizes]

        if sum(domain_batch_sizes) != num_samples_per_global_step:
            # NOTE: randomly add a sample to round it up
            domain_batch_sizes = self._round_up_domain_batch_sizes(
                domain_batch_sizes,
                target_total_size=num_samples_per_global_step,
            )

        assert all(x > 0 for x in domain_batch_sizes), "There is a domain with 0 samples per global batch"
        return domain_batch_sizes

    def __next__(self):

        # TODO(xrsrke): if reference training => don't recompute domain batch sizes
        if self.microbatch_idx == 0:
            self.domain_batch_sizes = self._recompute_domain_batch_sizes(
                domain_weights=self.doremi_context.domain_weights,
                num_samples_per_global_step=self.num_samples_per_global_step,
            )

        # if self.total_samples_yielded >= self.expected_total_samples:
        #     raise StopIteration

        batch = []
        for domain_index, (idxs, domain_batch_size) in enumerate(zip(self.domain_indices, self.domain_batch_sizes)):
            start_idx = self.domain_counters[domain_index]
            end_idx = start_idx + domain_batch_size

            # if domain_index == 0:
            #     self.debug_history.append((self.microbatch_idx, domain_index, start_idx, end_idx))

            # NOTE: BREAK 1
            if end_idx > len(idxs):
                print(
                    f"rank: {self.rank}, break1, end_idx: {end_idx}, start_idx: {start_idx}, len(idxs): {len(idxs)} \
                    domain_batch_sizes: {self.domain_batch_sizes}, \
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
                # if domain_index == 0:
                #     assert 1 == 1

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

        # assert_tensor_synced_across_pg(
        #     torch.tensor(batch, device="cuda"), self.parallel_context.dp_pg, msg=lambda err: f"batch are not synced across ranks {err}"
        # )
        # assert_tensor_synced_across_pg(
        #     torch.tensor(batch, device="cuda"), self.parallel_context.tp_pg, msg=lambda err: f"batch are not synced across ranks {err}"
        # )

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
        # self.total_samples_yielded += len(microbatch_idxs) * self.num_replicas

        # assert_tensor_synced_across_pg(
        #     torch.tensor(microbatch_idxs, device="cuda"), self.parallel_context.tp_pg, msg=lambda err: f"batch are not synced across ranks {err}"
        # )

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

            # NOTE: Randomly select a domain to increase/decrase a sample
            # to match the target_total_size
            eligible_indices = torch.nonzero(torch.tensor(domain_batch_size) > 1).view(-1)
            random_index = torch.randint(
                low=0, high=len(eligible_indices), size=(1,), generator=self.generator, device="cpu"
            ).item()
            selected_domain = eligible_indices[random_index].item()

            if diff > 0:
                domain_batch_size[selected_domain] += 1
            elif diff < 0 and domain_batch_size[selected_domain] > 0:
                domain_batch_size[selected_domain] -= 1

            total_batch_size = sum(domain_batch_size)

        return domain_batch_size

    def reset(self):
        """Reset the state of the sampler for a new epoch."""
        self.microbatch_idx = 0
        self.domain_counters = [0 for _ in self.datasets]
        self.total_samples_yielded = 0
        self.out_of_samples = False

        domain_indices = []
        for i, dataset in enumerate(self.datasets):
            local_indices = torch.arange(0, len(dataset), device="cpu").tolist()

            # NOTE: align the indicies across the combined dataset
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
        seed=trainer.config.data.seed,
        drop_last=True,
        doremi_context=doremi_context,
        parallel_context=parallel_context,
    )

    comebined_dataset = CombinedDataset(datasets)

    dataloader = DataLoader(
        comebined_dataset,
        batch_size=trainer.micro_batch_size,
        sampler=sampler,
        collate_fn=data_collator,
        drop_last=True,  # we also drop_last in `clm_process()`
        num_workers=trainer.config.data.num_loading_workers,
        pin_memory=True,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dist.get_rank(parallel_context.dp_pg)),
    )

    def _data_generator(dataloader):
        # dist.barrier()
        def inner():
            for batch in dataloader:
                # TODO(xrskre): remove this, use sanity_check
                batch = {k: v.to("cuda") for k, v in batch.items()}
                # NOTE: because the inference model don't take `domain_idxs`
                # as input we need to remove it from the batch
                batch_for_inference = {k: v for k, v in batch.items() if k != "domain_idxs"}

                ref_losses = trainer.ref_model(**batch_for_inference)["losses"]
                batch["ref_losses"] = ref_losses
                yield batch

        return inner

    # TODO(xrsrke): refactor out data_generator
    dataloader = _data_generator(dataloader) if doremi_context.is_proxy is True else dataloader

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
