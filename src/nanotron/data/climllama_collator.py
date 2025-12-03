"""Data collator for ClimLlama model.

This module implements the DataCollatorForClimLlama which handles batching
of climate-specific positional information alongside standard token sequences.
"""

import dataclasses
from typing import Dict, List, Union

import numpy as np
import torch

from nanotron import distributed as dist
from nanotron.models.climllama import CLIMLLAMA_SPATIAL_TEMPORAL_FEATURES
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer


@dataclasses.dataclass
class DataCollatorForClimLlama:
    """Data collator for ClimLlama that handles climate-specific position information.

    This collator extends the standard CLM collator with support for:
    - Variable index (var_idx)
    - Resolution index (res_idx)
    - Lead time index (leadtime_idx)
    - Spatial-temporal features (CLIMLLAMA_SPATIAL_TEMPORAL_FEATURES-D continuous features)

    The collator handles pipeline parallelism by returning TensorPointers for
    ranks that don't need the data.

    Attributes:
        sequence_length: Expected sequence length for the model
        input_pp_rank: Pipeline parallel rank that receives input data
        output_pp_rank: Pipeline parallel rank that receives output/labels
        parallel_context: Parallel context for distributed training
        use_doc_masking: Whether to use document masking for loss computation
        cp_return_global_position_ids: Whether to return global position IDs for CP
    """

    sequence_length: int
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext
    use_doc_masking: bool = True
    cp_return_global_position_ids: bool = True

    def __call__(
        self, examples: List[Dict[str, np.ndarray]]
    ) -> Dict[str, Union[np.ndarray, TensorPointer]]:
        """Collate examples into a batch with all position information.

        Args:
            examples: List of dicts with keys:
                - input_ids: Token IDs [seq_len + 1]
                - positions: Position IDs for document masking [seq_len + 1]
                - var_idx: Variable indices [seq_len + 1]
                - res_idx: Resolution indices [seq_len + 1]
                - leadtime_idx: Lead time indices [seq_len + 1]
                - spatial_temporal_features: [seq_len + 1, CLIMLLAMA_SPATIAL_TEMPORAL_FEATURES]

        Returns:
            Dict with batched tensors or TensorPointers:
                - input_ids: [batch, seq_len]
                - position_ids: [batch, seq_len] or global [batch, seq_len * cp_size]
                - var_idx: [batch, seq_len]
                - res_idx: [batch, seq_len]
                - leadtime_idx: [batch, seq_len]
                - spatial_temporal_features: [batch, seq_len, CLIMLLAMA_SPATIAL_TEMPORAL_FEATURES]
                - label_ids: [batch, seq_len]
                - label_mask: [batch, seq_len]
        """
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        cp_rank = dist.get_rank(self.parallel_context.cp_pg)
        cp_size = self.parallel_context.context_parallel_size

        # Process the case when current rank doesn't require data
        if current_pp_rank not in [self.input_pp_rank, self.output_pp_rank]:
            assert all(len(example) == 0 for example in examples)
            return {
                "input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "position_ids": TensorPointer(group_rank=self.input_pp_rank),
                "var_idx": TensorPointer(group_rank=self.input_pp_rank),
                "res_idx": TensorPointer(group_rank=self.input_pp_rank),
                "leadtime_idx": TensorPointer(group_rank=self.input_pp_rank),
                "spatial_temporal_features": TensorPointer(group_rank=self.input_pp_rank),
                "label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "label_mask": TensorPointer(group_rank=self.output_pp_rank),
            }

        batch_size = len(examples)

        # Stack input_ids to get shape and validate
        input_ids = np.vstack([examples[i]["input_ids"] for i in range(batch_size)])
        expanded_input_length = input_ids.shape[1]

        assert expanded_input_length == self.sequence_length + 1, (
            f"Samples should be of length {self.sequence_length + 1} (seq_len+1), "
            f"but got {expanded_input_length}"
        )

        # Initialize result dict with TensorPointers for non-participating ranks
        result: Dict[str, Union[np.ndarray, TensorPointer]] = {
            "input_ids": TensorPointer(group_rank=self.input_pp_rank),
            "position_ids": TensorPointer(group_rank=self.input_pp_rank),
            "var_idx": TensorPointer(group_rank=self.input_pp_rank),
            "res_idx": TensorPointer(group_rank=self.input_pp_rank),
            "leadtime_idx": TensorPointer(group_rank=self.input_pp_rank),
            "spatial_temporal_features": TensorPointer(group_rank=self.input_pp_rank),
            "label_ids": TensorPointer(group_rank=self.output_pp_rank),
            "label_mask": TensorPointer(group_rank=self.output_pp_rank),
        }

        # Process inputs (first PP rank)
        if current_pp_rank == self.input_pp_rank:
            # Stack all position arrays
            var_idx = np.vstack([examples[i]["var_idx"] for i in range(batch_size)])
            res_idx = np.vstack([examples[i]["res_idx"] for i in range(batch_size)])
            leadtime_idx = np.vstack([examples[i]["leadtime_idx"] for i in range(batch_size)])
            spatial_temporal_features = np.stack(
                [examples[i]["spatial_temporal_features"] for i in range(batch_size)]
            )

            # Get position_ids for document masking
            if "positions" in examples[0] and self.use_doc_masking:
                position_ids = np.vstack([examples[i]["positions"] for i in range(batch_size)])
            else:
                # Default: sequential position ids
                position_ids = np.arange(self.sequence_length + 1)[None, :].repeat(
                    batch_size, axis=0
                )

            # Drop last token for input (next-token prediction)
            result["input_ids"] = input_ids[:, :-1]
            result["var_idx"] = var_idx[:, :-1]
            result["res_idx"] = res_idx[:, :-1]
            result["leadtime_idx"] = leadtime_idx[:, :-1]
            result["spatial_temporal_features"] = spatial_temporal_features[:, :-1, :]

            # Handle position_ids for context parallelism
            if self.cp_return_global_position_ids:
                # Return full position_ids for all CP ranks (needed for cu_seqlens computation)
                result["position_ids"] = position_ids[:, :-1]
            else:
                result["position_ids"] = position_ids[:, :-1]

            # Context Parallelism: Each CP rank gets a slice of the inputs
            local_slice = slice(
                cp_rank * self.sequence_length // cp_size,
                (cp_rank + 1) * self.sequence_length // cp_size,
            )
            result["input_ids"] = result["input_ids"][:, local_slice]
            result["var_idx"] = result["var_idx"][:, local_slice]
            result["res_idx"] = result["res_idx"][:, local_slice]
            result["leadtime_idx"] = result["leadtime_idx"][:, local_slice]
            result["spatial_temporal_features"] = result["spatial_temporal_features"][
                :, local_slice, :
            ]
            if not self.cp_return_global_position_ids:
                result["position_ids"] = result["position_ids"][:, local_slice]

        # Process labels (last PP rank)
        if current_pp_rank == self.output_pp_rank:
            # Create labels (shifted by 1 for next-token prediction)
            result["label_ids"] = input_ids[:, 1:]

            # Create label mask based on position_ids
            if "positions" in examples[0] and self.use_doc_masking:
                position_ids = np.vstack([examples[i]["positions"] for i in range(batch_size)])
                shifted_positions = position_ids[:, 1:]  # Align with labels

                # Mask where position_ids resets to 0 (document boundary)
                result["label_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)
                zeros = shifted_positions == 0
                result["label_mask"] &= ~zeros
            else:
                # Default: all tokens are used for loss
                result["label_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

            # Context Parallelism: Each CP rank gets a slice of the labels
            local_slice = slice(
                cp_rank * self.sequence_length // cp_size,
                (cp_rank + 1) * self.sequence_length // cp_size,
            )
            result["label_ids"] = result["label_ids"][:, local_slice]
            result["label_mask"] = result["label_mask"][:, local_slice]

        return result


@dataclasses.dataclass
class DataCollatorForClimLlamaWithInputMask:
    """Data collator for ClimLlama with explicit input_mask support.

    This variant of the collator includes input_mask in the output, which may
    be needed for certain attention implementations or masking strategies.

    The input_mask is a boolean tensor indicating which positions are valid
    (True) vs padded (False).
    """

    sequence_length: int
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext
    use_doc_masking: bool = True
    cp_return_global_position_ids: bool = True

    def __call__(
        self, examples: List[Dict[str, np.ndarray]]
    ) -> Dict[str, Union[np.ndarray, TensorPointer]]:
        """Collate examples with input_mask included.

        Returns same as DataCollatorForClimLlama plus:
            - input_mask: [batch, seq_len] boolean mask
        """
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        cp_rank = dist.get_rank(self.parallel_context.cp_pg)
        cp_size = self.parallel_context.context_parallel_size

        if current_pp_rank not in [self.input_pp_rank, self.output_pp_rank]:
            assert all(len(example) == 0 for example in examples)
            return {
                "input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "input_mask": TensorPointer(group_rank=self.input_pp_rank),
                "position_ids": TensorPointer(group_rank=self.input_pp_rank),
                "var_idx": TensorPointer(group_rank=self.input_pp_rank),
                "res_idx": TensorPointer(group_rank=self.input_pp_rank),
                "leadtime_idx": TensorPointer(group_rank=self.input_pp_rank),
                "spatial_temporal_features": TensorPointer(group_rank=self.input_pp_rank),
                "label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "label_mask": TensorPointer(group_rank=self.output_pp_rank),
            }

        batch_size = len(examples)
        input_ids = np.vstack([examples[i]["input_ids"] for i in range(batch_size)])
        expanded_input_length = input_ids.shape[1]

        assert expanded_input_length == self.sequence_length + 1, (
            f"Samples should be of length {self.sequence_length + 1} (seq_len+1), "
            f"but got {expanded_input_length}"
        )

        result: Dict[str, Union[np.ndarray, TensorPointer]] = {
            "input_ids": TensorPointer(group_rank=self.input_pp_rank),
            "input_mask": TensorPointer(group_rank=self.input_pp_rank),
            "position_ids": TensorPointer(group_rank=self.input_pp_rank),
            "var_idx": TensorPointer(group_rank=self.input_pp_rank),
            "res_idx": TensorPointer(group_rank=self.input_pp_rank),
            "leadtime_idx": TensorPointer(group_rank=self.input_pp_rank),
            "spatial_temporal_features": TensorPointer(group_rank=self.input_pp_rank),
            "label_ids": TensorPointer(group_rank=self.output_pp_rank),
            "label_mask": TensorPointer(group_rank=self.output_pp_rank),
        }

        if current_pp_rank == self.input_pp_rank:
            var_idx = np.vstack([examples[i]["var_idx"] for i in range(batch_size)])
            res_idx = np.vstack([examples[i]["res_idx"] for i in range(batch_size)])
            leadtime_idx = np.vstack([examples[i]["leadtime_idx"] for i in range(batch_size)])
            spatial_temporal_features = np.stack(
                [examples[i]["spatial_temporal_features"] for i in range(batch_size)]
            )

            if "positions" in examples[0] and self.use_doc_masking:
                position_ids = np.vstack([examples[i]["positions"] for i in range(batch_size)])
            else:
                position_ids = np.arange(self.sequence_length + 1)[None, :].repeat(
                    batch_size, axis=0
                )

            # Drop last token for input
            result["input_ids"] = input_ids[:, :-1]
            result["var_idx"] = var_idx[:, :-1]
            result["res_idx"] = res_idx[:, :-1]
            result["leadtime_idx"] = leadtime_idx[:, :-1]
            result["spatial_temporal_features"] = spatial_temporal_features[:, :-1, :]

            if self.cp_return_global_position_ids:
                result["position_ids"] = position_ids[:, :-1]
            else:
                result["position_ids"] = position_ids[:, :-1]

            # Create input_mask (all True by default, can be modified for padding)
            result["input_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

            # Context Parallelism slicing
            local_slice = slice(
                cp_rank * self.sequence_length // cp_size,
                (cp_rank + 1) * self.sequence_length // cp_size,
            )
            result["input_ids"] = result["input_ids"][:, local_slice]
            result["input_mask"] = result["input_mask"][:, local_slice]
            result["var_idx"] = result["var_idx"][:, local_slice]
            result["res_idx"] = result["res_idx"][:, local_slice]
            result["leadtime_idx"] = result["leadtime_idx"][:, local_slice]
            result["spatial_temporal_features"] = result["spatial_temporal_features"][
                :, local_slice, :
            ]
            if not self.cp_return_global_position_ids:
                result["position_ids"] = result["position_ids"][:, local_slice]

        if current_pp_rank == self.output_pp_rank:
            result["label_ids"] = input_ids[:, 1:]

            if "positions" in examples[0] and self.use_doc_masking:
                position_ids = np.vstack([examples[i]["positions"] for i in range(batch_size)])
                shifted_positions = position_ids[:, 1:]
                result["label_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)
                zeros = shifted_positions == 0
                result["label_mask"] &= ~zeros
            else:
                result["label_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

            local_slice = slice(
                cp_rank * self.sequence_length // cp_size,
                (cp_rank + 1) * self.sequence_length // cp_size,
            )
            result["label_ids"] = result["label_ids"][:, local_slice]
            result["label_mask"] = result["label_mask"][:, local_slice]

        return result
