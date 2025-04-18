from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import CheckpointFunction

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import ParallelismArgs
from nanotron.config.models_config import Qwen2Config
from nanotron.nn.activations import ACT2FN
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)

logger = logging.get_logger(__name__)


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
    ) -> None:
        super().__init__()

        # Get TP mode and communication settings
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        # Define gate_up_proj as a merged layer for gate and up projections
        gate_up_contiguous_chunks = (
            config.intermediate_size,  # shape of gate_linear
            config.intermediate_size,  # shape of up_linear
        )
        self.gate_up_proj = TensorParallelColumnLinear(
            config.hidden_size,
            2 * config.intermediate_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,  # Qwen2 doesn't use bias for gate_up_proj
            async_communication=tp_linear_async_communication,
            contiguous_chunks=gate_up_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )

        # Define down projection
        self.down_proj = TensorParallelRowLinear(
            config.intermediate_size,
            config.hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,  # Qwen2 doesn't use bias for down_proj
            async_communication=tp_linear_async_communication,
        )

        # Define activation function (silu followed by multiplication)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        # Apply gate_up_proj to get gate and up projections
        merged_states = self.gate_up_proj(hidden_states)

        # Apply activation function (SiLU and Mul)
        gate_states, up_states = torch.split(merged_states, merged_states.shape[-1] // 2, dim=-1)
        hidden_states = self.act(gate_states) * up_states

        # Apply down projection
        hidden_states = self.down_proj(hidden_states)

        return {"hidden_states": hidden_states}


class Qwen2MoELayer(nn.Module):
    """Mixture of experts Layer for Qwen2 models."""

    def __init__(
        self,
        config: Qwen2Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # MoE specific configurations
        self.num_experts = config.moe_config.num_experts  # Total number of experts
        self.num_experts_per_token = config.moe_config.top_k  # Number of experts used per token (top-k)
        self.expert_parallel_size = getattr(parallel_config, "expert_parallel_size", 1)
        self.num_local_experts = self.num_experts // self.expert_parallel_size  # Experts per device

        # Get TP mode configuration
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        # Router for selecting experts
        self.router = TensorParallelColumnLinear(
            self.hidden_size,
            self.num_experts,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
        )

        # Enable shared experts if configured
        self.enable_shared_expert = getattr(config.moe_config, "enable_shared_expert", False)
        if self.enable_shared_expert:
            self.shared_expert = Qwen2MLP(
                config=config,
                parallel_config=parallel_config,
                tp_pg=tp_pg,
            )
            self.shared_expert_gate = TensorParallelColumnLinear(
                self.hidden_size,
                1,
                pg=tp_pg,
                mode=tp_mode,
                bias=False,
                async_communication=tp_linear_async_communication,
            )

        # Create the expert MLPs
        self.experts = nn.ModuleList(
            [
                Qwen2MLP(
                    config=config,
                    parallel_config=parallel_config,
                    tp_pg=tp_pg,
                )
                for _ in range(self.num_local_experts)
            ]
        )

        # Whether to recompute MoE layer during backward pass for memory efficiency
        self.recompute_layer = getattr(parallel_config, "recompute_layer", False)

        # Token dispatcher type - determines communication pattern
        self.token_dispatcher_type = getattr(config.moe_config, "token_dispatcher_type", "alltoall")
        # For more sophisticated implementations, we would add token dispatcher logic here

    def _compute_router_probabilities(self, hidden_states):
        """Compute routing probabilities for each token to each expert."""
        router_logits = self.router(hidden_states)  # [batch_size*seq_length, num_experts]

        # Get the top-k experts per token
        routing_weights, routing_indices = torch.topk(router_logits, k=self.num_experts_per_token, dim=-1)

        # Apply softmax on the top-k values
        routing_weights = F.softmax(routing_weights, dim=-1)

        return routing_weights, routing_indices

    def _dispatch_tokens(self, hidden_states, routing_weights, routing_indices):
        """
        Dispatches tokens to their selected experts.
        In a full implementation, this would handle the actual token routing logic
        including communication between devices.
        """
        # Simplified implementation - in a complete version this would handle
        # all-to-all or all-gather communications for distributed experts

        hidden_states.shape[0]
        dispatched_inputs = []
        expert_counts = []

        # For each expert, gather the tokens assigned to it
        for expert_idx in range(self.num_local_experts):
            # Find tokens that have this expert in their top-k
            expert_mask = (routing_indices == expert_idx).any(dim=-1)
            tokens_for_expert = hidden_states[expert_mask]

            # Get the routing weights for this expert
            expert_positions = (routing_indices == expert_idx).nonzero(as_tuple=True)
            token_positions, k_positions = expert_positions
            expert_weights = routing_weights[token_positions, k_positions].unsqueeze(-1)

            # Scale inputs by routing weights
            scaled_inputs = tokens_for_expert * expert_weights

            dispatched_inputs.append(scaled_inputs)
            expert_counts.append(len(tokens_for_expert))

        return dispatched_inputs, expert_counts

    def _combine_expert_outputs(self, expert_outputs, routing_indices, original_shape):
        """
        Combines outputs from different experts back to the original tensor layout.
        """
        # Initialize output tensor with zeros
        combined_output = torch.zeros(original_shape, device=expert_outputs[0].device)

        for expert_idx, expert_output in enumerate(expert_outputs):
            if expert_output.shape[0] == 0:  # Skip if no tokens were routed to this expert
                continue

            # Find positions where this expert was in the top-k
            expert_mask = (routing_indices == expert_idx).any(dim=-1)
            combined_output[expert_mask] += expert_output

        return combined_output

    def _core_forward(self, hidden_states):
        """Core forward logic for MoE layer."""
        # Get router probabilities
        routing_weights, routing_indices = self._compute_router_probabilities(hidden_states)

        # Dispatch tokens to experts
        dispatched_inputs, expert_counts = self._dispatch_tokens(hidden_states, routing_weights, routing_indices)

        # Process tokens with their assigned experts
        expert_outputs = []
        for expert_idx, (inputs, count) in enumerate(zip(dispatched_inputs, expert_counts)):
            if count == 0:  # Skip computation if no tokens assigned
                expert_outputs.append(torch.tensor([], device=hidden_states.device))
                continue

            # Forward through the expert
            output = self.experts[expert_idx](hidden_states=inputs)["hidden_states"]
            expert_outputs.append(output)

        # Combine expert outputs
        output = self._combine_expert_outputs(expert_outputs, routing_indices, hidden_states.shape)

        # Add shared expert contribution if enabled
        if self.enable_shared_expert:
            shared_expert_output = self.shared_expert(hidden_states=hidden_states)["hidden_states"]
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
            output = output + shared_gate * shared_expert_output

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
