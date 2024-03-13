from nanotron import logging
from nanotron.logging import log_rank

import nanotron.distributed as dist
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from nanotron.dataloader import SkipBatchSampler, EmptyInfiniteDataset, get_dataloader_worker_init # QUESTION: Move them here?

from dataclasses import dataclass
import torch
import numpy as np
from typing import Dict, List, Union, Optional
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel import ParallelContext

logger = logging.get_logger(__name__)

def build_nanoset_dataloader(
        dataset,
        sequence_length: int,
        parallel_context: ParallelContext,
        input_pp_rank: int,
        output_pp_rank: int,
        micro_batch_size: int,
        dataloader_num_workers: int,
        seed_worker: int,
        consumed_train_samples: int = 0,
        dataloader_drop_last: bool = True,
        dataloader_pin_memory: bool = True,
        use_loop_to_round_batch_size: bool = False,
) -> DataLoader:

    # Case of ranks not requiring data. We give them a dummy dataset, then the collator will do his job
    if not dist.get_rank(parallel_context.pp_pg) in [
        input_pp_rank,
        output_pp_rank,
    ]:
        dataset_length = len(dataset)
        dataset = EmptyInfiniteDataset(length=dataset_length)
        # No need to spawn a lot of workers, we can just use main
        dataloader_num_workers = 0
    
    data_collator = NanosetDataCollatorForCLM(
        sequence_length=sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        parallel_context=parallel_context,
    )

    # Compute size and rank of dataloader workers
    dp_ranks_size = parallel_context.dp_pg.size()
    dp_rank = parallel_context.dp_pg.rank()
    
    sampler = get_sampler(
        dataset=dataset,
        dl_ranks_size=dp_ranks_size,
        dl_rank=dp_rank,
        seed=seed_worker,
        use_loop_to_round_batch_size=use_loop_to_round_batch_size,
        micro_batch_size=micro_batch_size,
        drop_last=dataloader_drop_last,
        consumed_train_samples=consumed_train_samples,
    )

    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        collate_fn=data_collator,
        drop_last=dataloader_drop_last,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dp_rank),
        # TODO @thomasw21: I'm not sure but this doesn't seem to work at all.
        # pin_memory_device="cuda",
    )

@dataclass
class NanosetDataCollatorForCLM:
    """
    Data collator used for causal language modeling with Nanoset.

    - sequence_length: Sequence legth of the model
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
            # assert all(len(example) == 0 for example in examples)
            
            return {
                "input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "input_mask": TensorPointer(group_rank=self.input_pp_rank),
                "label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "label_mask": TensorPointer(group_rank=self.output_pp_rank),
            }

        # NOTE: Nanoset datasets key is "text"
        assert all(list(example.keys()) == ["text"] for example in examples)

        # TODO @nouamanetazi: Is it better to have examples as np.array or torch.Tensor?
        input_ids = np.vstack([examples[i]["text"] for i in range(len(examples))])  # (b, s)
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

def get_sampler(
    dl_ranks_size: int,
    dl_rank: int,
    dataset,
    seed: int,
    consumed_train_samples: int,
    drop_last: Optional[bool] = True,
) -> Optional[Sampler]:
    """returns sampler that restricts data loading to a subset of the dataset proper to the DP rank"""
    
    # NOTE: Removed DistributedSamplerWithLoop to support valid and test samplers. 
    
    sampler = DistributedSampler(
            dataset, num_replicas=dl_ranks_size, rank=dl_rank, seed=seed, drop_last=drop_last
    )

    if consumed_train_samples > 0:
        sampler = SkipBatchSampler(sampler, skip_batches=consumed_train_samples, dp_size=dl_ranks_size)

    return sampler
