from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence

import torch

from nanotron import distributed as dist


def all_gather_batches(in_tensor: torch.Tensor, in_split: Sequence[int], group: dist.ProcessGroup) -> torch.Tensor:
    # All gather along first dimension, allow un-equal splits
    out_tensor = torch.empty((sum(in_split),) + in_tensor.shape[1:], dtype=in_tensor.dtype, device=in_tensor.device)
    out_split_list = list(torch.split(out_tensor, in_split, dim=0))
    dist.all_gather(out_split_list, in_tensor, group=group)
    return out_tensor


class SamplerType(Enum):
    TOP_P = auto()
    TOP_K = auto()
    GREEDY = auto()
    BASIC = auto()


class Sampler:
    def __call__(self, sharded_logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class TopPSampler(Sampler):
    pg: dist.ProcessGroup
    p: float = 0.9
    temperature: float = 1.0
    filter_value: float = 0.0
    min_tokens_to_keep: int = 1

    def __call__(self, sharded_logits: torch.Tensor) -> torch.Tensor:
        batch_size, vocab_per_shard = sharded_logits.shape

        # Split max_values/max_indices into a list of tensors along batch
        # We have: [min_shard_batch_size + 1] * nb_shard_containing_extra_one + [min_shard_batch_size] * (self.pg.size() - nb_shard_containing_extra_one)
        min_shard_batch_size = batch_size // self.pg.size()
        nb_shard_containing_extra_one = batch_size % self.pg.size()
        in_split = tuple(
            min_shard_batch_size + 1 if rank < nb_shard_containing_extra_one else min_shard_batch_size
            for rank in range(self.pg.size())
        )

        # out_split should be all equal to be able to concat at last dimension
        out_split = (in_split[dist.get_rank(self.pg)],) * self.pg.size()
        total_out_size = in_split[dist.get_rank(self.pg)] * self.pg.size()

        # Prepare tensors for all-to-all operation
        # Gather logits from all vocab shards but shard on batch, tp_rank first
        sharded_logits_out = torch.empty(
            (total_out_size, vocab_per_shard),
            dtype=sharded_logits.dtype,
            device=sharded_logits.device,
        )  # [pg_size * sharded_batch_size, vocab_per_shard]

        local_sharded_logits_in = list(torch.split(sharded_logits, in_split, dim=0))
        local_sharded_logits_out = list(torch.split(sharded_logits_out, out_split, dim=0))

        dist.all_to_all(local_sharded_logits_out, local_sharded_logits_in, group=self.pg)

        logits = torch.cat(local_sharded_logits_out, dim=-1)  # [sharded_batch_size, vocab_size]

        probs = torch.softmax(logits.to(dtype=torch.float) / self.temperature, dim=-1)  # [batch_size, vocab_size]
        # Sort the probs and their corresponding indices in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=False, dim=-1)
        # Calculate the cumulative sum of the sorted probs
        # the bfloat16 type is not accurate enough for the cumulative sum
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1, dtype=torch.float)  # [batch_size, vocab_size]
        # Find the smallest set of indices for which the cumulative probability mass exceeds p
        sorted_indices_to_remove = cumulative_probs <= (1 - self.p)
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0
        # Construct the probability mask for original indices
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        filter_probs = probs.masked_fill(indices_to_remove, self.filter_value)
        sampled_indices = torch.multinomial(filter_probs, num_samples=1)

        # All gather the new decoder input ids along batch dimension
        gathered_new_decoder_input_ids = all_gather_batches(sampled_indices, in_split, group=self.pg)

        return gathered_new_decoder_input_ids


@dataclass
class GreedySampler(Sampler):
    pg: dist.ProcessGroup

    def __call__(self, sharded_logits: torch.Tensor) -> torch.Tensor:
        batch_size, vocab_per_shard = sharded_logits.shape

        # Find local max logit and its index
        # Note that max is deterministic, and always takes the first one.
        max_values, max_indices = sharded_logits.max(dim=-1, keepdim=True)  # [batch_size, 1]

        # Add offset to the max indices
        # TODO: We're assuming that TensorColumnLinear shards in a specific manner, i.e. rank 0 gets the first.
        # It might require us to expose something from TensorColumnLinear.
        max_indices = max_indices + (dist.get_rank(self.pg) * vocab_per_shard)

        # Split max_values/max_indices into a list of tensors along batch
        # We have: [min_shard_batch_size + 1] * nb_shard_containing_extra_one + [min_shard_batch_size] * (self.pg.size() - nb_shard_containing_extra_one)
        min_shard_batch_size = batch_size // self.pg.size()
        nb_shard_containing_extra_one = batch_size % self.pg.size()
        in_split = tuple(
            min_shard_batch_size + 1 if rank < nb_shard_containing_extra_one else min_shard_batch_size
            for rank in range(self.pg.size())
        )

        # out_split should be all equal to be able to concat at last dimension
        out_split = (in_split[dist.get_rank(self.pg)],) * self.pg.size()
        total_out_size = in_split[dist.get_rank(self.pg)] * self.pg.size()

        # Prepare tensors for all-to-all operation
        # Gather max logits and their indices from all shards, tp_rank first
        max_values_out_mat = torch.empty(
            (total_out_size, 1),
            dtype=max_values.dtype,
            device=max_values.device,
        )
        max_indices_out_mat = torch.empty(
            (total_out_size, 1),
            dtype=max_indices.dtype,
            device=max_indices.device,
        )

        local_max_values_in = list(torch.split(max_values, in_split, dim=0))
        local_max_indices_in = list(torch.split(max_indices, in_split, dim=0))
        local_max_values_out = list(torch.split(max_values_out_mat, out_split, dim=0))
        local_max_indices_out = list(torch.split(max_indices_out_mat, out_split, dim=0))

        dist.all_to_all(local_max_values_out, local_max_values_in, group=self.pg)
        dist.all_to_all(local_max_indices_out, local_max_indices_in, group=self.pg)

        # Concat assumes that the primary dimension is the same across all shards
        sharded_max_values = torch.cat(local_max_values_out, dim=-1)  # [sharded_batch_size, num_shards]
        sharded_max_indices = torch.cat(local_max_indices_out, dim=-1)  # [sharded_batch_size, num_shards]

        # Find global max logit across all shards
        # Note that max is deterministic, and always takes the first one.
        # [sharded_batch_size, 1]
        _global_max_values, global_max_indices = sharded_max_values.max(dim=-1, keepdim=True)

        # Select the corresponding token index from the offsetted gathered indices
        sharded_selected_tokens = sharded_max_indices.gather(1, global_max_indices)

        # All gather the new decoder input ids along batch dimension
        gathered_new_decoder_input_ids = all_gather_batches(sharded_selected_tokens, in_split, group=self.pg)

        return gathered_new_decoder_input_ids


@dataclass
class TopKSampler(Sampler):
    pg: dist.ProcessGroup
    k: int = 50
    temperature: float = 1.0

    def __call__(self, sharded_logits: torch.Tensor) -> torch.Tensor:
        batch_size, vocab_per_shard = sharded_logits.shape

        # Find local top-k logits and their indices
        local_top_k_values, local_top_k_indices = torch.topk(sharded_logits, self.k, dim=-1)

        # Add offset to the indices
        local_top_k_indices = local_top_k_indices + (dist.get_rank(self.pg) * vocab_per_shard)

        # Split local_top_k_values into a list of tensors along batch
        # We have: [min_shard_batch_size + 1] * nb_shard_containing_extra_one + [min_shard_batch_size] * (self.pg.size() - nb_shard_containing_extra_one)
        min_shard_batch_size = batch_size // self.pg.size()
        nb_shard_containing_extra_one = batch_size % self.pg.size()
        in_split = tuple(
            min_shard_batch_size + 1 if rank < nb_shard_containing_extra_one else min_shard_batch_size
            for rank in range(self.pg.size())
        )

        # out_split should be all equal to be able to concat at last dimension
        out_split = (in_split[dist.get_rank(self.pg)],) * self.pg.size()
        total_out_size = in_split[dist.get_rank(self.pg)] * self.pg.size()

        # The last shard could be smaller than shard_batch_size
        local_top_k_values_in = list(torch.split(local_top_k_values, in_split, dim=0))
        local_tok_k_indices_in = list(torch.split(local_top_k_indices, in_split, dim=0))
        # Prepare tensors for all-to-all operation
        # Gather top-k logits and their indices from all shards, tp_rank first
        top_k_values_out_mat = torch.empty(
            (total_out_size,) + local_top_k_values.shape[1:],
            dtype=local_top_k_values.dtype,
            device=local_top_k_values.device,
        )
        top_k_indices_out_mat = torch.empty(
            (total_out_size,) + local_top_k_indices.shape[1:],
            dtype=local_top_k_indices.dtype,
            device=local_top_k_indices.device,
        )
        local_top_k_values_out = list(torch.split(top_k_values_out_mat, out_split, dim=0))
        local_top_k_indices_out = list(torch.split(top_k_indices_out_mat, out_split, dim=0))

        dist.all_to_all(local_top_k_values_out, local_top_k_values_in, group=self.pg)
        dist.all_to_all(local_top_k_indices_out, local_tok_k_indices_in, group=self.pg)

        # Concat assumes that the primary dimension is the same across all shards
        sharded_local_top_k_values = torch.cat(local_top_k_values_out, dim=-1)  # [sharded_batch_size, k * num_shards]
        sharded_local_top_k_indices = torch.cat(
            local_top_k_indices_out, dim=-1
        )  # [sharded_batch_size, k * num_shards]

        # Select global top-k from the gathered top-k, now the top-k is across all vocab, batch_size is sharded
        sharded_top_k_values, sharded_top_k_indices = torch.topk(
            sharded_local_top_k_values, self.k, dim=-1
        )  # [sharded_batch_size, k]

        # Select corresponding indices from the gathered indices
        sharded_top_k_indices = sharded_local_top_k_indices.gather(
            -1, sharded_top_k_indices
        )  # [sharded_batch_size, k]

        # Apply temperature and compute softmax probabilities
        probs = torch.softmax(sharded_top_k_values.to(dtype=torch.float) / self.temperature, dim=-1)

        # Sample from the probabilities
        sampled_indices = torch.multinomial(probs, num_samples=1)  # [sharded_batch_size]

        # Select the corresponding token index from the global top-k indices
        new_decoder_input_ids = sharded_top_k_indices.gather(-1, sampled_indices)  # [sharded_batch_size]

        # All gather the new decoder input ids along batch dimension
        gathered_new_decoder_input_ids = all_gather_batches(new_decoder_input_ids, in_split, group=self.pg)

        return gathered_new_decoder_input_ids


@dataclass
class BasicSampler(Sampler):
    """Basic sampler that samples from the full vocab according to the logits."""

    pg: dist.ProcessGroup

    def __call__(self, sharded_logits: torch.Tensor) -> torch.Tensor:
        # We will cross batch and vocab shards to sample from the full vocab and a part of the batch
        # (right now logits are sharded on vocab and batch, so we need to do all-to-all)
        batch_size, vocab_per_shard = sharded_logits.shape

        # Split max_values/max_indices into a list of tensors along batch
        # We have: [min_shard_batch_size + 1] * nb_shard_containing_extra_one + [min_shard_batch_size] * (self.pg.size() - nb_shard_containing_extra_one)
        min_shard_batch_size = batch_size // self.pg.size()
        nb_shard_containing_extra_one = batch_size % self.pg.size()
        in_split = tuple(
            min_shard_batch_size + 1 if rank < nb_shard_containing_extra_one else min_shard_batch_size
            for rank in range(self.pg.size())
        )

        # out_split should be all equal to be able to concat at last dimension
        out_split = (in_split[dist.get_rank(self.pg)],) * self.pg.size()
        total_out_size = in_split[dist.get_rank(self.pg)] * self.pg.size()

        # Prepare tensors for all-to-all operation
        # Gather logits from all vocab shards but shard on batch, tp_rank first
        sharded_logits_out = torch.empty(
            (total_out_size, vocab_per_shard),
            dtype=sharded_logits.dtype,
            device=sharded_logits.device,
        )  # [pg_size * sharded_batch_size, vocab_per_shard]

        local_sharded_logits_in = list(torch.split(sharded_logits, in_split, dim=0))
        local_sharded_logits_out = list(torch.split(sharded_logits_out, out_split, dim=0))

        dist.all_to_all(local_sharded_logits_out, local_sharded_logits_in, group=self.pg)

        logits = torch.cat(local_sharded_logits_out, dim=-1)  # [sharded_batch_size, vocab_size]

        probs = torch.softmax(logits.to(dtype=torch.float), dim=-1)  # [batch_size, vocab_size]

        # Sample from the probabilities
        sampled_indices = torch.multinomial(probs, num_samples=1)

        # All gather the new decoder input ids along batch dimension
        gathered_new_decoder_input_ids = all_gather_batches(sampled_indices, in_split, group=self.pg)

        return gathered_new_decoder_input_ids
