from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import torch
from nanotron import distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer


def normalize(weights: List[float]) -> List[np.array]:
    """
    Normalize elements of a list

    Args:
        weights (List[float]): The weights

    Returns:
        List[numpy.array]: The normalized weights
    """
    w = np.array(weights, dtype=np.float64)
    w_sum = np.sum(w)
    w = w / w_sum
    return w


def count_dataset_indexes(dataset_idx: np.ndarray, n_datasets: int):
    counts = []

    for dataset in range(n_datasets):
        counts.append(np.count_nonzero(dataset_idx == dataset))

    return counts


# TODO Find a more elegant way
# We could compute position ids after tokenizing each sample but we will still miss the last length of the padding tokens
def build_position_ids(lengths, sequence_length) -> np.array:
    lengths.append((sequence_length - sum(lengths)))  # Append length of the padding tokens
    position_ids = [list(range(length)) for length in lengths]  # Create position ids list
    return np.array([x for xs in position_ids for x in xs], dtype=np.int32)  # Flatten list of position ids


# TODO delete, just 4 switching the remove cross-attention setting
def build_position_ids_dummy(lengths, sequence_length) -> np.array:
    return np.array(list(range(sequence_length)), dtype=np.int32)  # TODO numpy arange


# TODO delete, just 4 switching the training only on completitions setting. This will be in the __iter__ method instead of a function
def build_labels_completions_only(input_ids, is_completitions):
    labels = np.where(
        is_completitions, input_ids, -100
    )  # Mask tokens that don't belong to the completitions by the Assistant
    return np.array(labels[1:], dtype=np.int32)


# TODO delete, just 4 switching the training only on completitions setting
def build_labels(input_ids, is_completitions):
    return np.array(input_ids[1:], dtype=np.int32)


@dataclass
class DataCollatorForChatDataset:  # TODO Find a better name
    """
    Data collator used with Chat Dataset.

    - sequence_length: Sequence length of each sample in the batch
    - input_pp_rank: Discards last input id token
    - output_pp_rank: Discards first label id token
    - other pp ranks: Don't have data. Instead, we use `TensorPointer` to point to the rank having the data.
    """

    sequence_length: int
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Process the case when current rank doesn't require data. We return `TensorPointer` that points to ranks having the data.

        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        if current_pp_rank not in [
            self.input_pp_rank,
            self.output_pp_rank,
        ]:
            assert all(len(example) == 0 for example in examples)
            return {
                "input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "input_mask": TensorPointer(group_rank=self.input_pp_rank),
                "label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "label_mask": TensorPointer(group_rank=self.output_pp_rank),
            }

        # TODO clean this, as we are flatting the batch there is no necessity for vstack but we need the batch dimension too
        input_ids = np.vstack([examples[i]["input_ids"] for i in range(len(examples))])  # (b, s)
        label_ids = np.vstack([examples[i]["label_ids"] for i in range(len(examples))])  # (b, s)
        position_ids = np.vstack([examples[i]["position_ids"] for i in range(len(examples))])  # (b, s)

        result: Dict[str, Union[np.ndarray, TensorPointer]] = {}

        result["input_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["input_mask"] = TensorPointer(group_rank=self.input_pp_rank)
        result["label_ids"] = TensorPointer(group_rank=self.output_pp_rank)
        result["label_mask"] = TensorPointer(group_rank=self.output_pp_rank)

        # Process inputs
        if current_pp_rank == self.input_pp_rank:
            result["input_ids"] = input_ids
            result["input_mask"] = np.ones((1, self.sequence_length), dtype=np.bool_)
            result["position_ids"] = position_ids

        # Process labels: shift them to the left
        if current_pp_rank == self.output_pp_rank:
            result["label_ids"] = label_ids
            result["label_mask"] = np.ones((1, self.sequence_length), dtype=np.bool_)

        # Cast np.array to torch.Tensor
        result = {k: v if isinstance(v, TensorPointer) else torch.from_numpy(v) for k, v in result.items()}
        return result
