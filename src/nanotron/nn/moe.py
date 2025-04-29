from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import CheckpointFunction

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import ParallelismArgs
from nanotron.config.models_config import Qwen2Config
from nanotron.models.base import ignore_init_on_device_and_dtype
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


def permute(x: torch.Tensor, routing_indices: torch.Tensor):
    permuted_x, inverse_permute_mapping = ops.permute(x.to(torch.float32), routing_indices)
    permuted_x = permuted_x.to(x.dtype)
    return permuted_x, inverse_permute_mapping


def unpermute(x: torch.Tensor, inverse_mapping: torch.Tensor, routing_weights: torch.Tensor):
    return ops.unpermute(x, inverse_mapping, routing_weights)


class AllToAllDispatcher(nn.Module):
    def __init__(self, num_local_experts: int, num_experts: int, ep_pg: dist.ProcessGroup):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.num_experts = num_experts
        self.expert_parallel_size = dist.get_world_size(ep_pg)
        self.ep_pg = ep_pg

        self.input_split_sizes = None
        self.output_split_sizes = None

    def permute(
        self,
        hidden_states: torch.Tensor,
        routing_indices: torch.Tensor,
        logs,
    ):
        """
        Dispatches tokens to their selected experts.
        In a full implementation, this would handle the actual token routing logic
        including communication between devices.

        + local_routing_indices: is the initial routing indices for the local experts's tokens
        + dispatched_routing_indices: is the routing indices for the dispatched tokens corresponding to the local experts

        + inverse_permute_mapping: is the inverse of the permute mapping
        + inverse_expert_sorting_index: is the inverse of the expert sorting index
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

        # def get_dispatched_routing_indices(global_routing_indices, num_ranks, num_experts_per_rank):
        #     """
        #     Compute the routing indices corresponding to the dispatched hidden states on each rank.

        #     Args:
        #         global_routing_indices (torch.Tensor): The global expert indices for each token, shape [num_tokens, 1].
        #         num_ranks (int): Number of devices (ranks).
        #         num_experts_per_rank (int): Number of experts per rank.

        #     Returns:
        #         list of torch.Tensor: A list where each element is the routing indices for the dispatched tokens on that rank.
        #     """
        #     num_tokens = global_routing_indices.size(0)
        #     tokens_per_rank = num_tokens // num_ranks

        #     dispatched_routing = []

        #     for rank in range(num_ranks):
        #         # Determine the experts managed by this rank
        #         expert_start = rank * num_experts_per_rank
        #         expert_end = expert_start + num_experts_per_rank
        #         experts = set(range(expert_start, expert_end))

        #         rank_tokens = []
        #         # Iterate over each sending rank
        #         for sending_rank in range(num_ranks):
        #             # Get the tokens belonging to the sending_rank's local data
        #             sending_start = sending_rank * tokens_per_rank
        #             sending_end = sending_start + tokens_per_rank
        #             sending_tokens = list(range(sending_start, sending_end))

        #             # Filter tokens that are routed to current rank's experts
        #             filtered = []
        #             for token_idx in sending_tokens:
        #                 expert_idx = global_routing_indices[token_idx].item()
        #                 if expert_idx in experts:
        #                     filtered.append(token_idx)

        #             # Sort the filtered tokens by their expert index
        #             filtered.sort(key=lambda x: global_routing_indices[x].item())

        #             # Collect these tokens' global routing indices
        #             rank_tokens.extend(filtered)

        #         # Get the routing indices for these tokens
        #         routing_indices = [global_routing_indices[token_idx].item() for token_idx in rank_tokens]
        #         dispatched_routing.append(torch.tensor(routing_indices, device=global_routing_indices.device))

        #     return dispatched_routing

        import torch

        def get_dispatched_routing_indices(global_routing_indices, num_ranks, num_experts_per_rank):
            """
            Generated by qwen3: https://chat.qwen.ai/c/7921580f-1197-4170-80f4-5bf40e3f3b86
            Compute the routing indices corresponding to the dispatched hidden states on each rank,
            supporting Top-K routing (i.e., multiple expert indices per token).

            Args:
                global_routing_indices (torch.Tensor): Shape [num_tokens, top_k],
                                                    where each row contains the indices of the experts to route to.
                num_ranks (int): Number of devices (ranks).
                num_experts_per_rank (int): Number of experts per rank.

            Returns:
                List[torch.Tensor]: One tensor per rank, containing the expert indices of the dispatched tokens.
            """
            num_tokens = global_routing_indices.size(0)
            global_routing_indices.size(1)
            tokens_per_rank = num_tokens // num_ranks

            dispatched_routing = []

            for rank in range(num_ranks):
                # Define the range of experts managed by this rank
                expert_start = rank * num_experts_per_rank
                expert_end = expert_start + num_experts_per_rank
                experts = set(range(expert_start, expert_end))

                routing_indices_for_rank = []

                # Process tokens from all sending ranks
                for sending_rank in range(num_ranks):
                    sending_start = sending_rank * tokens_per_rank
                    sending_end = sending_start + tokens_per_rank
                    sending_tokens = list(range(sending_start, sending_end))

                    contributions = []
                    for token_idx in sending_tokens:
                        expert_indices = global_routing_indices[token_idx].tolist()
                        for expert_idx in expert_indices:
                            if expert_idx in experts:
                                contributions.append(expert_idx)

                    # Sort contributions from this sending_rank by expert index
                    contributions.sort()
                    routing_indices_for_rank.extend(contributions)

                # Convert to tensor
                dispatched_routing.append(
                    torch.tensor(routing_indices_for_rank, dtype=torch.long, device=global_routing_indices.device)
                )

            return dispatched_routing

        # NOTE: start from expert 0 to expert n
        # NOTE: because the routing indices is global,
        # but each expert device has a set of local experts
        # so we need to align the routing indices to the local experts index
        ep_rank = dist.get_rank(self.ep_pg)
        # num_tokens_per_expert = torch.bincount(
        #     routing_indices.flatten() - ep_rank * self.num_local_experts, minlength=self.num_local_experts
        # ).cuda()  # [num_local_experts]
        # num_tokens_per_expert = torch.bincount(
        #     routing_indices.flatten() - ep_rank * self.num_local_experts, minlength=self.num_experts
        # ).cuda()  # [num_local_experts]
        num_tokens_per_expert = torch.bincount(
            routing_indices.flatten(), minlength=self.num_experts
        ).cuda()  # [num_local_experts]

        global_routing_indices = differentiable_all_gather(routing_indices, group=self.ep_pg)

        hidden_states, inverse_permute_mapping = permute(hidden_states, routing_indices)

        # log_rank(f"[Qwen2MoELayer.forward.ep_pg.before_all_gather]", logger=logger, level=logging.INFO)
        # NOTE: this part is all-to-all token dispatching
        if self.expert_parallel_size > 1:
            # NOTE: input_size_splits has a shape = [expert_parallel_size]
            # where each value represent the number of tokens that we send from this device
            # to [i]th device in the input_size_splits
            # NOTE: Reshape num_local_tokens_per_expert to [ep_size, num_local_experts]
            # TODO: .view or .reshape? check which one is faster

            # NOTE: this is incorrect in the case of imbalance
            num_tokens_per_expert_device = num_tokens_per_expert.reshape(
                self.expert_parallel_size, self.num_local_experts
            )
            # NOTE: we can compute how many tokens this divide to send to [i]th device locally
            # TODO: double check cpu-gpu sync
            input_split_sizes = num_tokens_per_expert_device.sum(dim=1)
            # log_rank(f"[Qwen2MoELayer.forward.ep_pg.before_all_gather.input_split_sizes={input_split_sizes}]", logger=logger, level=logging.INFO)

            # list_num_tokens_per_expert_device = [torch.empty_like(num_tokens_per_expert_device) for _ in range(self.expert_parallel_size)]
            # dist.all_gather(list_num_tokens_per_expert_device, num_tokens_per_expert_device, group=self.ep_pg)

            list_input_split_sizes = [torch.zeros_like(input_split_sizes) for _ in range(self.expert_parallel_size)]
            dist.all_gather(list_input_split_sizes, input_split_sizes, group=self.ep_pg)

            # NOTE: we can compute how many tokens this divide to receive from [i]th device globally
            # NOTE: create a tensor corresponding to dist.get_rank(self.ep_pg)
            # TODO: double check cpu-gpu sync
            # output_split_sizes = [xs[dist.get_rank(self.ep_pg)].item() for xs in list_input_split_sizes]
            input_split_sizes = input_split_sizes.tolist()
            output_split_sizes = calculate_output_split_sizes_for_rank(
                list_input_split_sizes, dist.get_rank(self.ep_pg)
            )
        else:
            input_split_sizes, output_split_sizes = None, None
        # log_rank(f"[Qwen2MoELayer.forward.ep_pg.after_all_gather]", logger=logger, level=logging.INFO)

        # log_rank(f"[Qwen2MoELayer.forward.before_all_to_all]", logger=logger, level=logging.INFO)

        if self.expert_parallel_size > 1:
            list_output_split_sizes = [
                torch.empty_like(torch.tensor(output_split_sizes, device="cuda"))
                for _ in range(self.expert_parallel_size)
            ]
            dist.all_gather(list_output_split_sizes, torch.tensor(output_split_sizes, device="cuda"), group=self.ep_pg)

        # NOTE: we merge this [sorting + token duplication for topk>1] = permute before all-to-all
        # sort_indices = torch.argsort(routing_indices.squeeze(-1), stable=True)
        # sorted_hidden_states = hidden_states[sort_indices]

        dispatched_hidden_states = all_to_all(
            hidden_states,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=self.ep_pg,
        )

        dist.barrier()
        self.input_split_sizes = input_split_sizes
        self.output_split_sizes = output_split_sizes
        # log_rank(f"[Qwen2MoELayer.forward.after_all_to_all]", logger=logger, level=logging.INFO)

        # a list of rotuing indices corresponding to the dispatched inputs
        ep_rank = dist.get_rank(self.ep_pg)
        # we shouldn't sort the indices before permutation,
        # but we keep the same expert value for each dispatched token,
        # then the permutation function will handle the sorting and replicating for topk

        # NOTE: this part is reordering for the grouped_gemm
        # dispatched_inputs, inverse_permute_mapping = ops.permute(global_hidden_states, routing_indices)
        # the local_routing_indices shouldn't be sportee
        dispatched_routing_indices = get_dispatched_routing_indices(
            global_routing_indices, self.expert_parallel_size, self.num_local_experts
        )
        dispatched_routing_indices = dispatched_routing_indices[ep_rank]
        # NOTE: torch.bincount requires the indices to be int32
        # otherwise it raises: "RuntimeError: "bincount_cuda" not implemented for 'BFloat16'"
        # local_routing_indices = local_routing_indices.to(hidden_states.dtype)

        # NOTE: replace with an expert sorting index if it's faster kernel because
        # here we don't need to duplicate the expert tokens
        # dispatched_global_inputs, inverse_expert_sorting_index = permute(
        #     dispatched_hidden_states, dispatched_routing_indices
        # )
        expert_sort_indices = torch.argsort(dispatched_routing_indices.squeeze(-1), stable=True)
        sorted_and_dispatched_hidden_states = dispatched_hidden_states[expert_sort_indices]
        # dispatched_global_inputs = dispatched_global_inputs.to(dispatched_hidden_states.dtype)

        # NOTE: it should be the number of dispatched tokens per expert
        # because we will use this for local grouped_gemm
        # NOTE: the local_routing_indices has a global expert index,
        # so we need to subtract the number of local experts to get the local expert index

        dispatched_routing_indices = dispatched_routing_indices.to(torch.int32)

        num_local_dispatched_tokens_per_expert = torch.bincount(
            dispatched_routing_indices - ep_rank * self.num_local_experts, minlength=self.num_local_experts
        )
        num_local_dispatched_tokens_per_expert = num_local_dispatched_tokens_per_expert
        # return dispatched_global_inputs, inverse_permute_mapping, inverse_expert_sorting_index, num_local_dispatched_tokens_per_expert
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
        # so we need to unsort it back to the original order
        inverse_expert_sort_indices = torch.argsort(expert_sort_indices, stable=True)
        expert_outputs = expert_outputs.index_select(0, inverse_expert_sort_indices)

        # expert_outputs = ops.unpermute(expert_outputs, inverse_mapping, routing_weights)
        # NOTE: recompute the routing weights of the dispatched inputs
        # routing_weights = torch.ones_like(inverse_mapping).unsqueeze(-1)
        # NOTE: we do a combination of the expert outputs, and rearrange them back for all-to-all's comm pattern as well
        # TODO: replace this one with a sorting only kernel if possible
        # permuted_expert_outputs = unpermute(expert_outputs, inverse_expert_sorting_index, routing_weights)
        # permuted_expert_outputs = permuted_expert_outputs.to(expert_outputs.dtype)

        if self.expert_parallel_size > 1:
            list_input_split_sizes = [
                torch.zeros_like(torch.tensor(self.input_split_sizes, device="cuda"))
                for _ in range(self.expert_parallel_size)
            ]
            dist.all_gather(
                list_input_split_sizes, torch.tensor(self.input_split_sizes, device="cuda"), group=self.ep_pg
            )

            list_output_split_sizes = [
                torch.zeros_like(torch.tensor(self.output_split_sizes, device="cuda"))
                for _ in range(self.expert_parallel_size)
            ]
            dist.all_gather(
                list_output_split_sizes, torch.tensor(self.output_split_sizes, device="cuda"), group=self.ep_pg
            )

        all_to_all(
            expert_outputs,
            output_split_sizes=self.input_split_sizes,
            input_split_sizes=self.output_split_sizes,
            group=self.ep_pg,
        )

        dist.barrier(group=self.ep_pg)
        assert 1 == 1
        # NOTE: this part is un-reordering for the grouped_gemm
        # hidden_states = ops.unpermute(expert_outputs, inverse_mapping, routing_weights)
        # NOTE: undo the expert index sorting for all-to-all back to the original order
        # inverse_indices = torch.argsort(sort_indices, stable=True)
        # return dispatched_outputs.index_select(0, inverse_indices)


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

        num_local_experts = config.moe_config.num_experts // parallel_config.expert_parallel_size
        self.expert_parallel_size = parallel_config.expert_parallel_size
        self.num_local_experts = num_local_experts
        self.ep_pg = ep_pg
        self.merged_gate_up_proj = nn.Parameter(
            torch.randn(num_local_experts, config.hidden_size, 2 * config.moe_config.moe_intermediate_size)
        )
        self.merged_down_proj = nn.Parameter(
            torch.randn(num_local_experts, config.moe_config.moe_intermediate_size, config.hidden_size)
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ):
        """
        assume hidden_states is permuted

        grouped_gemm's notes:
        ops.gemm expect the inputs to have the following criteria:
        + expect a, b are in bfloat16
        + expect num_tokens_per_expert is a on cpu
        """
        # NOTE: ops.gemm requires "batch_sizes" (aka: num_tokens_per_expert here) to be on cpu
        num_tokens_per_expert = num_tokens_per_expert.to("cpu")

        # NOTE: refactor, should be the same line
        if self.expert_parallel_size == 1:
            ep_rank = dist.get_rank(self.ep_pg)
            num_local_tokens_per_expert = num_tokens_per_expert.view(
                self.expert_parallel_size, self.num_local_experts
            )[ep_rank]
        else:
            num_local_tokens_per_expert = num_tokens_per_expert

        if torch.count_nonzero(num_local_tokens_per_expert) == 0:
            # NOTE: this divide don't receive any tokens
            return {"hidden_states": hidden_states}

        merged_states = ops.gmm(hidden_states, self.merged_gate_up_proj, num_local_tokens_per_expert, trans_b=False)
        gate_states, up_states = torch.split(merged_states, merged_states.shape[-1] // 2, dim=-1)
        hidden_states = self.act(gate_states) * up_states
        hidden_states = ops.gmm(hidden_states, self.merged_down_proj, num_local_tokens_per_expert, trans_b=False)

        return {"hidden_states": hidden_states}


class Qwen2MoELayer(nn.Module):
    """Mixture of experts Layer for Qwen2 models."""

    def __init__(
        self,
        config: Qwen2Config,
        parallel_config: Optional[ParallelismArgs],
        parallel_context: ParallelContext,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

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

        # Enable shared experts if configured
        self.enable_shared_expert = config.moe_config.enable_shared_expert
        if self.enable_shared_expert:
            from nanotron.models.qwen import Qwen2MLP

            self.shared_expert = Qwen2MLP(
                config=config,
                parallel_config=parallel_config,
                tp_pg=parallel_context.tp_pg,
                intermediate_size=config.moe_config.shared_expert_intermediate_size,
            )
            # TODO: duplicte the shared expert gate
            self.shared_expert_gate = nn.Linear(
                self.hidden_size,
                1,
                bias=False,
            )  # TODO: ensure shared_expert_gate is tied across TP

        # Create the expert MLPs
        # TODO: merge the ep process group to non-ep in parallel_context initialization
        # this is hacky
        # if not hasattr(parallel_context, "ep_pg"):
        #     ep_pg = parallel_context.tp_pg
        # else:
        #     ep_pg = parallel_context.ep_pg

        self.experts = GroupedMLP(config, parallel_config, ep_pg=parallel_context.ep_pg)
        # Whether to recompute MoE layer during backward pass for memory efficiency
        self.recompute_layer = parallel_config.recompute_layer
        self.ep_pg = parallel_context.ep_pg

    def _compute_expert_outputs(self, hidden_states, routing_weights, routing_indices, logs):
        assert 1 == 1
        (
            dispatched_inputs,
            inverse_permute_mapping,
            sort_indices,
            num_tokens_per_expert,
        ) = self.token_dispatcher.permute(hidden_states, routing_indices, logs)

        logs["dispatched_inputs"] = dispatched_inputs
        logs["inverse_permute_mapping"] = inverse_permute_mapping
        logs["num_tokens_per_expert"] = num_tokens_per_expert
        logs["input_split_sizes"] = self.token_dispatcher.input_split_sizes
        logs["output_split_sizes"] = self.token_dispatcher.output_split_sizes

        # log_rank(f"[Qwen2MoELayer.forward.before_experts]", logger=logger, level=logging.INFO)
        expert_outputs = self.experts(dispatched_inputs, num_tokens_per_expert)
        # log_rank(f"[Qwen2MoELayer.forward.after_experts]", logger=logger, level=logging.INFO)

        logs["expert_outputs"] = expert_outputs
        # log_rank(f"[Qwen2MoELayer.forward.before_combine_expert_outputs]", logger=logger, level=logging.INFO)
        output = self.token_dispatcher.unpermute(
            expert_outputs["hidden_states"], inverse_permute_mapping, routing_weights, sort_indices
        )

        logs["output_after_combine_expert_outputs"] = output
        return output

    def _core_forward(self, hidden_states):
        """Core forward logic for MoE layer."""
        # Get top-k routing weights and indices
        logs = {}
        routing_weights, routing_indices = self.router(hidden_states)  # [num_tokens, num_experts_per_token]
        logs["input"] = hidden_states
        logs["routing_weights"] = routing_weights
        logs["routing_indices"] = routing_indices

        output = self._compute_expert_outputs(hidden_states, routing_weights, routing_indices, logs)

        # log_rank(f"[Qwen2MoELayer.forward.after_combine_expert_outputs]", logger=logger, level=logging.INFO)
        # Add shared expert contribution if enabled
        if self.enable_shared_expert:
            shared_expert_output = self.shared_expert(hidden_states=hidden_states)["hidden_states"]

            logs["shared_expert_output"] = shared_expert_output
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
            logs["shared_gate_after_sigmoid"] = shared_gate
            output = output + shared_gate * shared_expert_output

            logs["output_after_shared_expert"] = output

        # return output, logs
        return output

    def _checkpointed_forward(self, hidden_states):
        """Apply gradient checkpointing to save memory during training."""
        return CheckpointFunction.apply(self._core_forward, True, hidden_states)

    def forward(self, hidden_states):
        """Forward pass for the MoE layer."""
        if self.recompute_layer and self.training:
            hidden_states = self._checkpointed_forward(hidden_states)
        else:
            hidden_states = self._core_forward(hidden_states)

        return {"hidden_states": hidden_states}
