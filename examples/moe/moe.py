""" LlaMa model with MoEs"""
import warnings
from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from config_llamoe import LlaMoEConfig

try:
    import megablocks.ops as ops
    from megablocks.layers.all_to_all import all_to_all
except ImportError:
    warnings.warn("Please install megablocks to use MoEs: `pip install megablock`")

from nanotron import distributed as dist
from nanotron.config import ParallelismArgs
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)
from torch import nn


def inclusive_cumsum(x, dim):
    scalar = ops.inclusive_cumsum(x, dim)
    return scalar.view(1) if not len(scalar.size()) else scalar


class MLP(nn.Module):
    def __init__(
        self,
        config: LlaMoEConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
    ):
        super().__init__()

        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        self.expert_pg_size = parallel_config.expert_parallel_size
        self.experts_per_rank = config.moe_num_experts // min(self.expert_pg_size, config.moe_num_experts)

        self.w1 = TensorParallelColumnLinear(
            config.hidden_size,
            config.intermediate_size * self.experts_per_rank,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
        )

        self.w2 = TensorParallelRowLinear(
            config.intermediate_size * self.experts_per_rank,
            config.hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication and tp_mode is TensorParallelLinearMode.REDUCE_SCATTER,
        )
        # TODO @nouamane: jit
        self.act = partial(F.gelu, approximate="tanh")

    def forward(self, hidden_states):  # [seq_length, batch_size, hidden_dim]
        merged_states = self.w1(hidden_states)
        hidden_states = self.w2(self.act(merged_states))
        return {"hidden_states": hidden_states}


# Adapted from megablocks.layers.mlp.ParallelDroplessMLP
class ParallelDroplessMLP(torch.nn.Module):
    def __init__(
        self,
        config: LlaMoEConfig,
        use_bias: bool,
        expert_parallel_group: dist.ProcessGroup,
        tp_pg: dist.ProcessGroup,
        parallel_config: Optional[ParallelismArgs],
    ):
        super().__init__()
        self.config = config
        self.use_bias = use_bias

        self.expert_pg_size = expert_parallel_group.size()
        self.expert_parallel_group = expert_parallel_group

        self.hidden_sharding_degree = self.expert_pg_size // min(self.expert_pg_size, self.config.moe_num_experts)
        self.experts_per_rank = self.config.moe_num_experts // min(self.expert_pg_size, self.config.moe_num_experts)

        self.num_experts = config.moe_num_experts
        self.num_experts_per_tok = self.config.num_experts_per_tok

        # Calculate the number of bits needed to represent the expert indices
        # so that we can pass it to radix sort.
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)

        if use_bias:
            self.bias = torch.nn.Parameter(torch.empty(config.hidden_size))  # TODO: init

        # Select the forward function for the operating mode.
        self.forward_fn = self.parallel_forward_once if self.expert_pg_size > 1 else self.forward_once

        self.blocking = 128
        self.mlp = MLP(config=config, parallel_config=parallel_config, tp_pg=tp_pg)

    def expert_capacity(self, tokens):
        tokens_per_expert = self.num_experts_per_tok * tokens * self.expert_pg_size / self.num_experts
        return int(self.config.moe_capacity_factor * tokens_per_expert)

    def indices_and_bins(self, top_expert):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        top_expert = top_expert.int()
        bin_ids, indices = ops.sort(top_expert, self.sort_end_bit)
        tokens_per_expert = ops.histogram(top_expert, self.num_experts)

        # Calculate the bin bounds for the sorted tokens.
        bins = inclusive_cumsum(tokens_per_expert, 0)
        return indices, bin_ids, bins, tokens_per_expert

    def indices_and_padded_bins(self, top_experts):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        bin_ids, indices = ops.sort(top_experts, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        tokens_per_expert = ops.histogram(top_experts, self.num_experts)

        # Round the token counts up to the block size used in
        # the matrix muliplications. Caculate the starting
        # position of each bin.
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, self.blocking)
        padded_bins = inclusive_cumsum(padded_tokens_per_expert, 0)

        # Calculate the bin bounds for the sorted tokens.
        bins = inclusive_cumsum(tokens_per_expert, 0)
        return indices, bin_ids, bins, padded_bins, tokens_per_expert

    def forward_once(self, x, expert_weights, top_experts):  # TODO: sparse
        with torch.no_grad():
            (
                indices,
                bin_ids,
                bins,
                padded_bins,
                tokens_per_expert,
            ) = self.indices_and_padded_bins(top_experts)

        # Route the tokens for MoE computation.
        x = x.view(-1, x.shape[-1])
        x = ops.padded_gather(x, indices, bin_ids, bins, padded_bins, self.num_experts_per_tok)

        x = self.mlp(x)["hidden_states"]

        # Un-route the data for the MoE output.
        x = ops.padded_scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            padded_bins,
            self.num_experts_per_tok,
            self.config.quantize_scatter_num_bits,
        )
        return x, tokens_per_expert

    def parallel_forward_once(self, x, expert_weights, top_experts):
        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = self.indices_and_bins(top_experts)
            repeated_tokens_per_expert = ops.repeat(tokens_per_expert, (self.hidden_sharding_degree,))
            parallel_tokens_per_expert = torch.empty_like(repeated_tokens_per_expert)
            tpe_handle = torch.distributed.all_to_all_single(
                parallel_tokens_per_expert,
                repeated_tokens_per_expert,
                group=self.expert_parallel_group,
                async_op=True,
            )

        x = x.view(-1, x.shape[-1])
        x = ops.gather(x, indices, bin_ids, bins, self.num_experts_per_tok)

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with torch.no_grad():
            tpe_handle.wait()

            # Reshape to [expert_pg_size, num_experts_per_rank].
            repeated_tokens_per_expert = repeated_tokens_per_expert.view(self.expert_pg_size, self.experts_per_rank)
            parallel_tokens_per_expert = parallel_tokens_per_expert.view(self.expert_pg_size, self.experts_per_rank)

            send_counts = repeated_tokens_per_expert.cpu().sum(dim=-1)
            parallel_tokens_per_expert_cpu = parallel_tokens_per_expert.cpu()
            recv_counts = parallel_tokens_per_expert_cpu.sum(dim=-1)

            # Convert the send/recv counts to lists.
            send_counts = send_counts.tolist()
            recv_counts = recv_counts.tolist()
            tokens_received = sum(recv_counts)

        x = ops.repeat(x, (self.hidden_sharding_degree, 1))

        # Start the cross-device permutation asynchronously so we can
        # overlap communication with computation.
        parallel_x, parallel_x_handle = all_to_all(
            x, recv_counts, send_counts, self.expert_parallel_group, async_op=True
        )

        with torch.no_grad():
            replicate_bins = inclusive_cumsum(parallel_tokens_per_expert.flatten(), 0)

            # Construct the expert indices for the permuted tokens.
            parallel_top_expert = torch.remainder(
                torch.arange(
                    self.num_experts * self.hidden_sharding_degree,
                    dtype=torch.int32,
                    device=indices.device,
                ),
                self.experts_per_rank,
            )
            parallel_top_expert = ops.replicate(
                parallel_top_expert.unsqueeze(dim=0), replicate_bins, tokens_received
            ).flatten()

            parallel_bin_ids, parallel_indices = ops.sort(parallel_top_expert, self.sort_end_bit)

            # Calculate the bins boundaries from the token counts.
            parallel_tokens_per_expert = parallel_tokens_per_expert.sum(dim=0, dtype=torch.int)
            parallel_bins = inclusive_cumsum(parallel_tokens_per_expert, 0)

            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            tokens, hs = x.size()
            expert_capacity = self.expert_capacity(tokens)
            if expert_capacity == 0:
                expert_capacity = torch.max(parallel_tokens_per_expert).item()

        # Locally permute the tokens and perform the expert computation.
        # Block to make sure that the cross-device permutation is complete.
        parallel_x_handle.wait()
        parallel_x = self.permute_and_compute(
            parallel_x,
            parallel_tokens_per_expert,
            parallel_indices,
            parallel_bin_ids,
            None,  # expert_weights
            parallel_bins,
            expert_capacity,
            num_experts_per_tok=self.num_experts_per_tok,
        )

        # Un-permute the tokens across the devices.
        x, _ = all_to_all(parallel_x, send_counts, recv_counts, self.expert_parallel_group)

        # Reduce along the hidden sharding to get the final outputs.
        shape = (self.hidden_sharding_degree, -1, self.config.hidden_size)
        x = ops.sum(x.view(shape), dim=0)

        # Un-permute locally to setup for the next series of operations.
        x = ops.scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            self.num_experts_per_tok,
        )
        return x, tokens_per_expert.flatten()

    def forward(self, x, scores, expert_weights, top_experts):
        in_shape = x.size()

        # Compute the experts.
        expert_weights = expert_weights.flatten()
        top_experts = top_experts.flatten()
        x, tokens_per_expert = self.forward_fn(x, expert_weights, top_experts)

        x = x.view(in_shape)
        if self.use_bias:
            return x + self.bias
        return x

    def permute_and_compute(
        self,
        x,
        tokens_per_expert,
        indices,
        bin_ids,
        expert_weights,
        bins,
        expert_capactiy,
        num_experts_per_tok,
    ):
        # Round the token counts up to the block size used in the matrix
        # multiplication. Calculate the starting position of each bin.
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, self.blocking)
        padded_bins = inclusive_cumsum(padded_tokens_per_expert, 0)

        # Route the tokens for MoE computation.
        x = x.view(-1, x.shape[-1])
        x = ops.padded_gather(x, indices, bin_ids, bins, padded_bins, num_experts_per_tok)

        # Perform the expert computation.
        x = self.mlp(x)["hidden_states"]

        # Un-route the data for the MoE output.
        return ops.padded_scatter(x, indices, bin_ids, expert_weights, bins, padded_bins, num_experts_per_tok)


# Adapted from megablocks.layers.router.LearnedRouter
class LearnedRouter(torch.nn.Module):
    def __init__(self, config: LlaMoEConfig):
        super().__init__()
        self.layer = torch.nn.Linear(config.hidden_size, config.moe_num_experts, bias=False)
        # TODO: initialization
        self.config = config

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.view(-1, self.config.hidden_size)
        router_logits = self.layer(x)  # (batch * sequence_length, n_experts)
        scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)  # TODO: fuse?

        if self.config.num_experts_per_tok == 1:
            expert_weights, expert_indices = scores.max(dim=-1, keepdim=True)
        else:
            expert_weights, expert_indices = torch.topk(scores, self.config.num_experts_per_tok, dim=-1)

        return scores, expert_weights, expert_indices.int()


class dMoE(torch.nn.Module):
    def __init__(
        self,
        config: LlaMoEConfig,
        expert_parallel_group: dist.ProcessGroup,
        tp_pg: dist.ProcessGroup,
        parallel_config: Optional[ParallelismArgs],
    ):
        super(dMoE, self).__init__()

        # Token router.
        self.gate = LearnedRouter(config)

        # Expert computation helper.
        self.experts = ParallelDroplessMLP(
            config,
            use_bias=False,
            expert_parallel_group=expert_parallel_group,
            tp_pg=tp_pg,
            parallel_config=parallel_config,
        )

    def forward(self, x: torch.Tensor):
        # Compute the expert scores and assignments.
        # TODO: support sequence parallelism
        scores, expert_weights, top_experts = self.gate(x)

        # Compute the experts.
        return self.experts(x, scores, expert_weights, top_experts)
