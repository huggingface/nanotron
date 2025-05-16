import dataclasses
from typing import Dict, List, Union

import numpy as np
import torch

from nanotron import distributed as dist
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer


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
    # torch vs numpy
    use_numpy: bool = True

    @torch.profiler.record_function("DataCollatorForCLM.__call__")
    def __call__(self, examples: List[Dict[str, List[np.ndarray]]]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:

        vstack = np.vstack if self.use_numpy else torch.vstack
        ones = np.ones if self.use_numpy else torch.ones
        bool_dtype = np.bool_ if self.use_numpy else torch.bool

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

        # TODO @nouamanetazi: Is it better to have examples as np.array or torch.Tensor?
        input_ids = vstack([examples[i]["input_ids"] for i in range(len(examples))])  # (b, s)
        batch_size, expanded_input_length = input_ids.shape

        result: Dict[str, Union[np.ndarray, torch.LongTensor, TensorPointer]] = {}

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
            result["input_mask"] = ones((batch_size, self.sequence_length), dtype=bool_dtype)

            # Context Parallelism: Each CP rank gets a slice of the input_ids and input_mask
            cp_rank, cp_size = dist.get_rank(self.parallel_context.cp_pg), self.parallel_context.context_parallel_size
            local_slice = slice(
                cp_rank * self.sequence_length // cp_size, (cp_rank + 1) * self.sequence_length // cp_size
            )
            result["input_ids"] = result["input_ids"][:, local_slice]  # (b, s/cp_size)
            result["input_mask"] = result["input_mask"][:, local_slice]  # (b, s/cp_size)

        # Process labels: shift them to the left
        if current_pp_rank == self.output_pp_rank:
            result["label_ids"] = input_ids[:, 1:]

            # Create label mask based on position_ids
            if "positions" in examples[0]:
                # Get position_ids for the labels (shifted right by 1 to align with label_ids)
                position_ids = np.vstack([examples[i]["positions"] for i in range(len(examples))])
                position_ids = position_ids[:, 1:]  # Shift right to align with labels

                # Create mask: True for all tokens except the one before position_id == 0
                result["label_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

                # Find where position_ids is 0
                zeros = position_ids == 0
                # Mask the current token where we found zeros (since labels are already shifted right)
                result["label_mask"] &= ~zeros
            else:
                # Default: all tokens are used for loss
                result["label_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

            # Context Parallelism: Each CP rank gets a slice of the label_ids and label_mask
            cp_rank, cp_size = dist.get_rank(self.parallel_context.cp_pg), self.parallel_context.context_parallel_size
            local_slice = slice(
                cp_rank * self.sequence_length // cp_size, (cp_rank + 1) * self.sequence_length // cp_size
            )
            result["label_ids"] = result["label_ids"][:, local_slice]  # (b, s/cp_size)
            result["label_mask"] = result["label_mask"][:, local_slice]  # (b, s/cp_size)

        if (
            not isinstance(result["input_ids"], TensorPointer)
            and result["input_ids"].shape[-1] != self.sequence_length // cp_size
        ):
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {result['input_ids'].shape[-1]}, but should be"
                f" {self.sequence_length // cp_size}."
            )
        if (
            not isinstance(result["label_ids"], TensorPointer)
            and result["label_ids"].shape[-1] != self.sequence_length // cp_size
        ):
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {result['label_ids'].shape[-1]}, but should be"
                f" {self.sequence_length // cp_size}."
            )

        # # Maybe cast np.array to torch.Tensor
        # result = {
        #     k: v if isinstance(v, TensorPointer) else (torch.from_numpy(v).contiguous() if self.use_numpy else v)
        #     for k, v in result.items()
        # }  # TODO: @nouamane in case of memory issues, try keeping numpy here.
        # # assert contiguous
        # for k, v in result.items():
        #     if not isinstance(v, TensorPointer):
        #         assert v.is_contiguous(), f"{k} is not contiguous"
        #         assert not v.is_cuda, f"{k} is in cuda. Bad for pinning memory"
        return result


@dataclasses.dataclass
class DataCollatorForCLMWithPositionIds:
    """
    Data collator used for causal language modeling with position IDs.

    - input_pp_rank: Discards last input id token
    - output_pp_rank: Discards first label id token
    - other pp ranks: Don't have data. Instead, we use `TensorPointer` to point to the rank having the data.
    """

    sequence_length: int
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext
    use_doc_masking: bool = True
    cp_return_global_position_ids: bool = True

    def __call__(self, examples: List[Dict[str, List[np.ndarray]]]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Process the case when current rank doesn't require data
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        if current_pp_rank not in [self.input_pp_rank, self.output_pp_rank]:
            assert all(len(example) == 0 for example in examples)
            return {
                "input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "positions": TensorPointer(group_rank=self.input_pp_rank),
                "label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "label_mask": TensorPointer(group_rank=self.output_pp_rank),
            }

        # input_ids[0,:20]
        # array([  198,    50,    30, 12532,  3589,   198,    51,    30, 30618,
        #         198,    52,    30,  8279, 11274,   198, 21350,    42,   340,
        #         0,  1780])
        # position_ids[0,:20]
        # array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        #        17, 18,  0])
        # result["label_ids"][0,:20]
        # array([   50,    30, 12532,  3589,   198,    51,    30, 30618,   198,
        #         52,    30,  8279, 11274,   198, 21350,    42,   340,     0,
        #         1780,   314])
        # -> label_id for 0 is 1780 -> need to mask 1780
        # result["label_mask"][0,:20]
        # array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
        #         True,  True,  True,  True,  True,  True,  True,  True,  True,
        #     False,  True])

        # document starts with first token, and last token is eos_token (0)
        # label_mask should be 1 for all tokens except the last one

        # Stack input_ids
        input_ids = np.vstack([examples[i]["input_ids"] for i in range(len(examples))])  # (b, s)
        batch_size, expanded_input_length = input_ids.shape

        result: Dict[str, Union[np.ndarray, TensorPointer]] = {}

        # Initialize all fields as TensorPointers
        result["input_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["position_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["label_ids"] = TensorPointer(group_rank=self.output_pp_rank)
        result["label_mask"] = TensorPointer(group_rank=self.output_pp_rank)

        assert expanded_input_length == self.sequence_length + 1, (
            f"Samples should be of length {self.sequence_length + 1} (seq_len+1), " f"but got {expanded_input_length}"
        )

        # Process inputs
        if current_pp_rank == self.input_pp_rank:
            result["input_ids"] = input_ids[:, :-1]

            if "positions" in examples[0] and self.use_doc_masking:
                # Use provided position_ids if available
                position_ids = np.vstack([examples[i]["positions"] for i in range(len(examples))])
                # Simply drop the last position ID for each example
                result["positions"] = position_ids[:, :-1]
            else:
                # Default: sequential position ids
                result["positions"] = np.arange(self.sequence_length)[None, :].repeat(batch_size, axis=0)

            # Context Parallelism: Each CP rank gets a slice of the input_ids and position_ids
            cp_rank, cp_size = dist.get_rank(self.parallel_context.cp_pg), self.parallel_context.context_parallel_size
            local_slice = slice(
                cp_rank * self.sequence_length // cp_size, (cp_rank + 1) * self.sequence_length // cp_size
            )
            result["input_ids"] = result["input_ids"][:, local_slice]  # (b, s/cp_size)
            if not self.cp_return_global_position_ids:
                result["positions"] = result["positions"][:, local_slice]  # (b, s/cp_size)
            result["position_ids"] = result.pop("positions")

        # Process labels
        if current_pp_rank == self.output_pp_rank:
            result["label_ids"] = input_ids[:, 1:]

            # Create label mask based on position_ids
            if "positions" in examples[0] and self.use_doc_masking:
                # Get position_ids for the labels (shifted right by 1 to align with label_ids)
                position_ids = np.vstack([examples[i]["positions"] for i in range(len(examples))])
                position_ids = position_ids[:, 1:]  # Shift right to align with labels

                # Create mask: True for all tokens except the one before position_id == 0
                result["label_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

                # Find where position_ids is 0
                zeros = position_ids == 0
                # Mask the current token where we found zeros (since labels are already shifted right)
                result["label_mask"] &= ~zeros
            else:
                # Default: all tokens are used for loss
                result["label_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

            # Context Parallelism: Each CP rank gets a slice of the label_ids and label_mask
            local_slice = slice(
                cp_rank * self.sequence_length // cp_size, (cp_rank + 1) * self.sequence_length // cp_size
            )
            result["label_ids"] = result["label_ids"][:, local_slice]  # (b, s/cp_size)
            result["label_mask"] = result["label_mask"][:, local_slice]  # (b, s/cp_size)

        # Validate shapes
        if (
            isinstance(result["input_ids"], torch.Tensor)
            and result["input_ids"].shape[-1] != self.sequence_length // cp_size
        ):
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. Length is {result['input_ids'].shape[-1]}, but should be"
                f" {self.sequence_length // cp_size}."
            )
        if (
            isinstance(result["label_ids"], torch.Tensor)
            and result["label_ids"].shape[-1] != result["input_ids"].shape[-1]
        ):
            raise ValueError(
                f"`label_ids` are incorrectly preprocessed. Length is {result['label_ids'].shape[-1]}, but should be"
                f" {result['input_ids'].shape[-1]}."
            )

        # # Cast np.array to torch.Tensor
        # result = {
        #     k: v if isinstance(v, TensorPointer) else torch.from_numpy(v).contiguous() for k, v in result.items()
        # }

        # # assert contiguous
        # for k, v in result.items():
        #     if not isinstance(v, TensorPointer):
        #         assert v.is_contiguous(), f"{k} is not contiguous"
        #         assert not v.is_cuda, f"{k} is in cuda. Bad for pinning memory"

        return result
