from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import CheckpointFunction

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import ParallelismArgs
from nanotron.config.models_config import Qwen2Config
from nanotron.models.base import ignore_init_on_device_and_dtype
from nanotron.nn._moe_kernel import _get_dispatched_routing_indices
from nanotron.nn.activations import ACT2FN
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import (
    all_to_all,
    differentiable_all_gather,
)

logger = logging.get_logger(__name__)


try:
    import grouped_gemm.ops as ops
except ImportError:
    raise RuntimeError(
        "Grouped GEMM is not available. Please run `pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@main` (takes less than 5 minutes)"
    )


def is_expert_param(name: str) -> bool:
    from nanotron.constants import EXPERT_PARAM_NAMES

    return any(param in name for param in EXPERT_PARAM_NAMES)


def permute(x: torch.Tensor, routing_indices: torch.Tensor):
    permuted_x, inverse_permute_mapping = ops.permute(x.to(torch.float32), routing_indices)
    permuted_x = permuted_x.to(x.dtype)
    return permuted_x, inverse_permute_mapping


def unpermute(x: torch.Tensor, inverse_mapping: torch.Tensor, routing_weights: torch.Tensor):
    comebined_x = ops.unpermute(x.to(torch.float32), inverse_mapping, routing_weights)
    return comebined_x.to(x.dtype)


@dataclass
class MoELogging:
    """
    num_local_tokens: List[torch.Tensor]: The number of tokens per local expert per layer
    """

    num_local_tokens: List[torch.Tensor]


class ScaleGradient(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x: torch.Tensor, scale: float):
        ctx.scale = scale
        return x

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad: torch.Tensor):
        return grad * ctx.scale, None


class AllToAllDispatcher(nn.Module):
    def __init__(self, num_local_experts: int, num_experts: int, ep_pg: dist.ProcessGroup):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.num_experts = num_experts
        self.expert_parallel_size = dist.get_world_size(ep_pg)
        self.ep_pg = ep_pg

        self.input_split_sizes = None
        self.output_split_sizes = None

        self._use_haojun_permute = True

    def _haojun_permute_topk(self, hidden_states, routing_indices):
        """
        hidden_states: [num_tokens, hidden_dim]
        routing_indices: [num_tokens, topk]
        num_experts: total number of experts
        Returns:
            permuted: [num_tokens * topk, hidden_dim]
            expert_counts: [num_experts], number of tokens assigned to each expert
            permute_metadata: metadata for unpermute
        """
        num_tokens, hidden_dim = hidden_states.shape
        topk = routing_indices.shape[-1]

        # Expand hidden_states to match topk, shape: [num_tokens, topk, hidden_dim]
        expanded_states = hidden_states.unsqueeze(1).expand(-1, topk, -1)

        # Flatten the batch: [num_tokens * topk, hidden_dim]
        flat_states = expanded_states.reshape(-1, hidden_dim)
        flat_indices = routing_indices.reshape(-1)  # [num_tokens * topk]

        # Sort by expert (so tokens are grouped by expert index)
        sorted_expert_indices, sort_order = flat_indices.sort()
        permuted = flat_states[sort_order]  # [num_tokens * topk, hidden_dim]

        # Count tokens per expert
        num_tokens_per_expert = torch.bincount(sorted_expert_indices, minlength=self.num_experts)

        return permuted, (sort_order, flat_indices.shape[0]), num_tokens_per_expert

    def _haojun_unpermute_topk(self, permuted, sort_order, total_elements, routing_weights):
        """
        permuted: [num_tokens * topk, hidden_dim], output from experts
        sort_order: indices used to sort the tokens in permute
        total_elements: num_tokens * topk
        routing_weights: [num_tokens, topk], used to scale expert outputs before aggregation
        Returns:
            output: [num_tokens, hidden_dim], weighted sum over topk expert outputs
        """
        device = permuted.device
        hidden_dim = permuted.size(-1)
        num_tokens, topk = routing_weights.shape

        # Restore original order
        unsort_order = torch.empty_like(sort_order)
        unsort_order[sort_order] = torch.arange(total_elements, device=device)

        # Restore the original [num_tokens * topk, hidden_dim] order
        unpermuted = permuted[unsort_order]

        # Reshape to [num_tokens, topk, hidden_dim]
        unpermuted = unpermuted.view(num_tokens, topk, hidden_dim)

        # Apply routing weights
        routing_weights = routing_weights.to(permuted.dtype).unsqueeze(-1)  # [num_tokens, topk, 1]
        weighted_output = unpermuted * routing_weights

        # Aggregate over topk experts: sum over topk axis
        output = weighted_output.sum(dim=1)  # [num_tokens, hidden_dim]

        return output

    def permute(
        self,
        hidden_states: torch.Tensor,
        routing_indices: torch.Tensor,
    ):
        """
        Dispatches tokens to their selected experts.
        In a full implementation, this would handle the actual token routing logic
        including communication between devices.

        + local_routing_indices: is the initial routing indices for the local experts's tokens
        + dispatched_routing_indices: is the routing indices for the dispatched tokens corresponding to the local experts

        + inverse_permute_mapping: is the inverse of the permute mapping
        + inverse_expert_sorting_index: is the inverse of the expert sorting index

        outputs:
        + num_local_dispatched_tokens_per_expert: we return it in cpu so don't have to move to cpu for grouped_gemm.gmm
        """

        def calculate_output_split_sizes_for_rank(all_input_split_sizes, rank):
            """
            Calculate output_split_sizes for a specific rank based on input_split_sizes from all ranks.

            Args:
                all_input_split_sizes: List of lists where all_input_split_sizes[i] is the input_split_sizes for rank i
                rank: The rank to calculate output_split_sizes for

            Returns:
                List containing the output_split_sizes for the specified rank
            """
            world_size = len(all_input_split_sizes)
            output_split_sizes = []

            # For each possible sender rank
            for sender_rank in range(world_size):
                # Get how much data sender_rank is sending to our rank
                size = all_input_split_sizes[sender_rank][rank]
                output_split_sizes.append(size.item())

            return output_split_sizes

        with torch.autograd.profiler.record_function("AllToAllDispatcher.permute.all_to_all.pre"):
            # NOTE: start from expert 0 to expert n
            # NOTE: because the routing indices is global,
            # but each expert device has a set of local experts
            # so we need to align the routing indices to the local experts index
            ep_rank = dist.get_rank(self.ep_pg)
            num_tokens_per_expert = torch.bincount(
                routing_indices.flatten(), minlength=self.num_experts
            )  # [num_local_experts]
            global_routing_indices = differentiable_all_gather(routing_indices, group=self.ep_pg)

            if self._use_haojun_permute:
                hidden_states, inverse_permute_mapping, _ = self._haojun_permute_topk(hidden_states, routing_indices)
            else:
                hidden_states, inverse_permute_mapping = permute(hidden_states, routing_indices)

            # NOTE: this part is all-to-all token dispatching
            if self.expert_parallel_size > 1:
                # NOTE: Reshape num_local_tokens_per_expert to [ep_size, num_local_experts]
                # TODO: .view or .reshape? check which one is faster
                # NOTE: this is incorrect in the case of imbalance
                num_tokens_per_expert_device = num_tokens_per_expert.reshape(
                    self.expert_parallel_size, self.num_local_experts
                )
                # NOTE: input_size_splits has a shape = [expert_parallel_size]
                # where each value represent the number of tokens that we send from this device
                # to [i]th device in the input_size_splits
                # TODO: double check cpu-gpu sync
                input_split_sizes = num_tokens_per_expert_device.sum(dim=1)
                list_input_split_sizes = [
                    torch.empty_like(input_split_sizes) for _ in range(self.expert_parallel_size)
                ]
                dist.all_gather(list_input_split_sizes, input_split_sizes, group=self.ep_pg)

                # NOTE: we can compute how many tokens this divide to receive from [i]th device globally
                # NOTE: create a tensor corresponding to dist.get_rank(self.ep_pg)
                # TODO: double check cpu-gpu sync
                input_split_sizes = input_split_sizes.tolist()
                output_split_sizes = calculate_output_split_sizes_for_rank(
                    list_input_split_sizes, dist.get_rank(self.ep_pg)
                )
            else:
                input_split_sizes, output_split_sizes = None, None

        with torch.autograd.profiler.record_function("AllToAllDispatcher.permute.all_to_all"):
            dispatched_hidden_states = all_to_all(
                hidden_states,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=self.ep_pg,
            )

            self.input_split_sizes = input_split_sizes
            self.output_split_sizes = output_split_sizes

        with torch.autograd.profiler.record_function("AllToAllDispatcher.permute.expert_index_sorting"):
            # NOTE: a list of rotuing indices corresponding to the dispatched inputs
            # we shouldn't sort the indices before permutation,
            # but we keep the same expert value for each dispatched token,
            # then the permutation function will handle the sorting and replicating for topk
            dispatched_routing_indices = _get_dispatched_routing_indices(
                global_routing_indices, self.expert_parallel_size, num_experts=self.num_experts
            )[ep_rank]
            # NOTE: we prefer to keep num_local_dispatched_tokens_per_expert on cpu,
            # so we don't have to move it again for grouped_gemm
            dispatched_routing_indices_cpu = dispatched_routing_indices.cpu()

            # NOTE: it should be the number of dispatched tokens per expert
            # because we will use this for local grouped_gemm
            # NOTE: the local_routing_indices has a global expert index,
            # so we need to subtract the number of local experts to get the local expert index
            # dispatched_routing_indices = dispatched_routing_indices.cpu()
            num_local_dispatched_tokens_per_expert = torch.bincount(
                dispatched_routing_indices_cpu - ep_rank * self.num_local_experts, minlength=self.num_local_experts
            )

            # NOTE: torch.bincount requires the indices to be int32
            # otherwise it raises: "RuntimeError: "bincount_cuda" not implemented for 'BFloat16'"
            # NOTE: if dispatched_routing_indices only has a single value,
            # then the shape of expert_sort_indices is a single scalar, but we want it to be a 1d tensor
            # for the sorted_and_dispatched_hidden_states to has shape [num_tokens, d_model]
            expert_sort_indices = torch.argsort(dispatched_routing_indices.squeeze(-1), stable=True)
            expert_sort_indices = expert_sort_indices.view(-1)

            sorted_and_dispatched_hidden_states = dispatched_hidden_states[expert_sort_indices]

        return (
            sorted_and_dispatched_hidden_states,
            inverse_permute_mapping,
            expert_sort_indices,
            num_local_dispatched_tokens_per_expert,
        )

    def unpermute(self, expert_outputs, inverse_permute_mapping, routing_weights, expert_sort_indices):
        """
        Combines outputs from different experts back to the original tensor layout.
        """

        # NOTE: the expert_outputs here is sorted by the expert index
        # so we need to unsort it back to the dispatching order
        inverse_expert_sort_indices = torch.argsort(expert_sort_indices, stable=True)
        # NOTE: expert_outputs is on cuda, inverse_expert_sort_indices is on cpu, how to remove a cuda sync point here?
        expert_outputs = expert_outputs.index_select(0, inverse_expert_sort_indices)

        undispatched_expert_outputs = all_to_all(
            expert_outputs,
            output_split_sizes=self.input_split_sizes,
            input_split_sizes=self.output_split_sizes,
            group=self.ep_pg,
        )

        # NOTE: merging the expert output combination and un-permuting them back into a single operation
        if self._use_haojun_permute:
            comebined_expert_outputs = self._haojun_unpermute_topk(
                undispatched_expert_outputs, *inverse_permute_mapping, routing_weights
            )
        else:
            comebined_expert_outputs = unpermute(undispatched_expert_outputs, inverse_permute_mapping, routing_weights)
        return comebined_expert_outputs


class Router(nn.Module):
    def __init__(self, config: Qwen2Config, parallel_config: Optional[ParallelismArgs], layer_idx: int):
        super().__init__()
        self.config = config
        self.parallel_config = parallel_config
        self.layer_idx = layer_idx

        self.num_experts = config.moe_config.num_experts
        self.num_experts_per_token = config.moe_config.top_k

        # float32 routing weights
        # NOTE: qwen keep the routing weights in float32
        # https://github.com/huggingface/transformers/blob/27a25bee4fcb865e8799ba026f1ea4455f2cca98/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L608
        with ignore_init_on_device_and_dtype():
            self.weight = nn.Parameter(
                torch.randn(self.num_experts, config.hidden_size, dtype=torch.float32, device="cuda")
            )
        assert self.weight.dtype == torch.float32

    def gating(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for all experts (no softmax)."""
        # NOTE: qwen keep the routing logits in float32
        # https://github.com/huggingface/transformers/blob/27a25bee4fcb865e8799ba026f1ea4455f2cca98/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L613
        return F.linear(x.to(torch.float32), self.weight, bias=None)

    def routing(self, logits: torch.Tensor):
        """Top-k softmax-normalized routing weights and indices."""
        routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        routing_weights, routing_indices = torch.topk(routing_weights, k=self.num_experts_per_token, dim=-1)
        routing_indices = routing_indices.to(torch.int32)  # NOTE: ops.permute requires indices to be int32
        return routing_weights, routing_indices

    def forward(self, x: torch.Tensor):
        logits = self.gating(x)
        return self.routing(logits)


class GroupedMLP(nn.Module):
    def __init__(self, config: Qwen2Config, parallel_config: Optional[ParallelismArgs], ep_pg: dist.ProcessGroup):
        super().__init__()
        moe_config = config.moe_config

        num_local_experts = moe_config.num_experts // parallel_config.expert_parallel_size
        self.expert_parallel_size = parallel_config.expert_parallel_size
        self.num_local_experts = torch.tensor(num_local_experts, dtype=torch.int32, device="cuda")
        self.ep_pg = ep_pg
        self.merged_gate_up_proj = nn.Parameter(
            torch.randn(num_local_experts, moe_config.moe_hidden_size, 2 * moe_config.moe_intermediate_size)
        )
        self.merged_down_proj = nn.Parameter(
            torch.randn(num_local_experts, moe_config.moe_intermediate_size, moe_config.moe_hidden_size)
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_local_tokens_per_expert: torch.Tensor,
    ):
        """
        assume hidden_states is permuted

        grouped_gemm's notes:
        ops.gemm expect the inputs to have the following criteria:
        + expect a, b are in bfloat16
        + expect num_tokens_per_expert is a on cpu
        """

        # NOTE: if no tokens are assigned to this expert device, then we just return the hidden states
        if torch.count_nonzero(num_local_tokens_per_expert) == 0:
            # NOTE: this divide don't receive any tokens
            return {"hidden_states": hidden_states}

        merged_states = ops.gmm(hidden_states, self.merged_gate_up_proj, num_local_tokens_per_expert, trans_b=False)
        gate_states, up_states = torch.split(merged_states, merged_states.shape[-1] // 2, dim=-1)
        hidden_states = self.act(gate_states) * up_states
        hidden_states = ops.gmm(hidden_states, self.merged_down_proj, num_local_tokens_per_expert, trans_b=False)

        return {"hidden_states": hidden_states}


class Qwen2MoEMLPLayer(nn.Module):
    """Mixture of experts Layer for Qwen2 models."""

    def __init__(
        self,
        config: Qwen2Config,
        parallel_config: Optional[ParallelismArgs],
        parallel_context: ParallelContext,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        moe_config = config.moe_config
        self.hidden_size = moe_config.moe_hidden_size
        self.intermediate_size = moe_config.moe_intermediate_size

        # MoE specific configurations
        num_experts = config.moe_config.num_experts  # Total number of experts
        num_local_experts = config.moe_config.num_experts // parallel_config.expert_parallel_size  # Experts per device
        self.num_experts_per_token = config.moe_config.top_k  # Number of experts used per token (top-k)
        self.expert_parallel_size = parallel_config.expert_parallel_size
        self.num_local_experts = num_local_experts  # Experts per device

        # Get TP mode configuration

        # Router for selecting experts
        self.router = Router(config, parallel_config, layer_idx)
        self.token_dispatcher = AllToAllDispatcher(num_local_experts, num_experts, parallel_context.ep_pg)
        self.token_dispatcher._use_haojun_permute = config.moe_config.use_haojun_permute

        # Enable shared experts if configured
        self.enable_shared_expert = config.moe_config.enable_shared_expert
        if self.enable_shared_expert:
            from nanotron.models.qwen import Qwen2MLP

            self.shared_expert = Qwen2MLP(
                config=config,
                parallel_config=parallel_config,
                tp_pg=parallel_context.tp_pg,
                hidden_size=moe_config.shared_expert_hidden_size,
                intermediate_size=moe_config.shared_expert_intermediate_size,
            )
            # TODO: duplicte the shared expert gate
            self.shared_expert_gate = nn.Linear(
                self.hidden_size,
                1,
                bias=False,
            )  # TODO: ensure shared_expert_gate is tied across TP

        self.experts = GroupedMLP(config, parallel_config, ep_pg=parallel_context.ep_pg)
        # Whether to recompute MoE layer during backward pass for memory efficiency
        self.recompute_layer = parallel_config.recompute_layer
        self.ep_pg = parallel_context.ep_pg
        self.layer_idx = layer_idx

    def _compute_expert_outputs(self, hidden_states, routing_weights, routing_indices):
        (
            dispatched_inputs,
            inverse_permute_mapping,
            expert_sort_indices,
            num_local_tokens_per_expert,
        ) = self.token_dispatcher.permute(hidden_states, routing_indices)

        expert_outputs = self.experts(dispatched_inputs, num_local_tokens_per_expert)
        output = self.token_dispatcher.unpermute(
            expert_outputs["hidden_states"], inverse_permute_mapping, routing_weights, expert_sort_indices
        )
        return output, num_local_tokens_per_expert

    def _core_forward(self, hidden_states, moe_logging: Optional[MoELogging]):
        """Core forward logic for MoE layer."""
        # Get top-k routing weights and indices
        routing_weights, routing_indices = self.router(hidden_states)  # [num_tokens, num_experts_per_token]

        output, num_local_tokens_per_expert = self._compute_expert_outputs(
            hidden_states, routing_weights, routing_indices
        )

        if self.enable_shared_expert:
            shared_expert_output = self.shared_expert(hidden_states=hidden_states)["hidden_states"]
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
            output = output + shared_gate * shared_expert_output

        if moe_logging is not None:
            moe_logging[self.layer_idx, :] = num_local_tokens_per_expert

        return {"hidden_states": output}

    def _checkpointed_forward(self, hidden_states):
        """Apply gradient checkpointing to save memory during training."""
        return CheckpointFunction.apply(self._core_forward, True, hidden_states)

    def forward(self, hidden_states, moe_logging: Optional[MoELogging] = None):
        """Forward pass for the MoE layer."""
        if self.recompute_layer and self.training:
            outputs = self._checkpointed_forward(hidden_states, moe_logging)
        else:
            outputs = self._core_forward(hidden_states, moe_logging)

        return outputs
