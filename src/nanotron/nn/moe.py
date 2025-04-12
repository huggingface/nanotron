# coding=utf-8
# Copyright 2018 HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LLaMa model."""

from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import CheckpointFunction
from torchtyping import TensorType

from nanotron import logging
from nanotron.config import Llama4Config, ParallelismArgs, Qwen2Config
from nanotron.models.llama import (
    MLP,
)
from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import differentiable_all_gather
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelLinearMode,
)

logger = logging.get_logger(__name__)


class MLPMoE(nn.Module):
    """Mixture of experts Layer for MLP models."""

    def __init__(
        self,
        config: Union[Qwen2Config, Llama4Config],
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        assert (
            parallel_config.tp_mode == TensorParallelLinearMode.ALL_REDUCE
        ), "MoE layer must be ALL_REDUCE because otherwise we need to all2all the tokens twice to route them"

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # MoE specific configurations
        self.num_experts = config.moe_config.num_experts
        self.num_experts_per_token = config.moe_config.top_k

        # Router for selecting experts
        # do tied linear for duplicated router
        self.router = TensorParallelColumnLinear(
            self.hidden_size,
            self.num_experts,
            pg=tp_pg,
            mode=TensorParallelLinearMode.ALL_REDUCE,
            bias=False,
            async_communication=False,
        )

        # Enable shared experts if configured
        if config.moe_config.enable_shared_expert:
            self.shared_expert = MLP(
                config=config,
                parallel_config=parallel_config,
                tp_pg=tp_pg,
            )

        # Create the expert MLPs
        # TODO: double check if all-to-all needed for
        # the first mlp of moe
        self.experts = nn.ModuleList(
            [
                MLP(
                    config=config,
                    parallel_config=parallel_config,
                    tp_pg=tp_pg,
                )
                for _ in range(self.num_experts)
            ]
        )

        self.recompute_layer = parallel_config.moe_layer_recompute
        self.tp_pg = tp_pg

    def _compute_router_probabilities(
        self, hidden_states: TensorType["bs*seq_len", "hidden_size"]
    ) -> Tuple[TensorType["bs*seq_len", "num_experts"], TensorType["bs*seq_len", "num_experts"]]:
        """Compute routing probabilities for each token to each expert."""
        from einops import rearrange

        # todo: .shape[0] is bs
        # seq_len = hidden_states.shape[0]
        # bs = hidden_states.shape[1]

        # assert_tensor_equal_across_processes(hidden_states, self.tp_pg)

        parallel_router_logits = self.router(hidden_states)  # [batch_size*seq_length/2, num_experts/2]
        router_logits = differentiable_all_gather(
            parallel_router_logits, dim=-1, group=self.tp_pg
        )  # [batch_size*seq_length/2, num_experts]

        # router_logits = self.router(hidden_states)  # [batch_size*seq_length/2, num_experts]

        # assert router_logits.shape == (seq_len, bs, self.num_experts)
        rotuing_router_logits, routing_indices = torch.topk(router_logits, k=self.num_experts_per_token, dim=-1)
        rotuing_router_logits = rearrange(rotuing_router_logits, "... 1 -> ...")
        routing_indices = rearrange(routing_indices, "... 1 -> ...")

        routing_scores = F.sigmoid(rotuing_router_logits)

        return routing_scores, routing_indices  # b, s/2

    def _dispatch_tokens(
        self,
        hidden_states: TensorType["bs*seq_len"],
        routing_weights: TensorType["seq_len", "bs"],
        routing_indices: TensorType["seq_len", "bs"],
    ):
        """
        Dispatches tokens to their selected experts.
        In a full implementation, this would handle the actual token routing logic
        including communication between devices.

        routing_weights: where a value specifies the weight of the expert
        routing_indices: where a value specifies the index of the expert
        """
        from einops import einsum, rearrange

        # hidden_states: b, s/2, h | b, s, h
        # routing_weights: b, s/2 | b, s

        # TODO: convert this to a tensor then do padding?
        dispatched_inputs = []  # [n_experts, num_tokens_for_expert, d_model]
        num_tokens_for_experts = torch.empty(
            self.num_experts, device=routing_weights.device, dtype=routing_weights.dtype
        )  # [n_experts]

        # NOTE: recheck if seq_len is sharded
        total_tokens = routing_weights.numel()
        # total_tokens = routing_weights.numel() * self.tp_pg.size()
        router_loss = torch.tensor(0.0, device=routing_weights.device, dtype=routing_weights.dtype)
        frac_of_tokens_routed_to_expert_list = []

        # routing_weights = rearrange(routing_weights, "seq_len bs -> (seq_len bs)")
        for expert_idx in range(self.num_experts):
            # NOTE: Find tokens that have this expert in their top-k
            # expert_mask.shape = [seq_len, bs]
            # true for all tokens that have this expert in their top-k
            expert_mask = routing_indices == expert_idx  # b, s
            num_tokens_for_expert = expert_mask.sum()
            # INFO: in case hidden_states was sharded here, wed need to reroute the tokens to the correct device
            # in case of EP, all-to-all
            tokens_for_expert = hidden_states[expert_mask]  # b, s, h

            # Get the routing weights for this expert
            idx_of_tokens_for_expert = rearrange(torch.nonzero(expert_mask.view(-1)), "... 1 -> ...")  # todo
            routing_weights_for_expert = routing_weights[idx_of_tokens_for_expert]

            # TODO: double check and remove if possible
            if num_tokens_for_expert == 1:
                routing_weights_for_expert = routing_weights_for_expert.unsqueeze(0)

            # NOTE: no scaling
            scaled_inputs = einsum(
                tokens_for_expert, routing_weights_for_expert, "n_tokens d_model, n_tokens -> n_tokens d_model"
            )
            dispatched_inputs.append(scaled_inputs)
            num_tokens_for_experts[expert_idx] = num_tokens_for_expert

            # TODO: separate router calculation from dispatching tokens
            # todo: make sure it is correct for reduce_scatter
            frac_of_tokens_routed_to_expert = num_tokens_for_expert / total_tokens
            frac_of_router_prob_routed_to_expert = routing_weights_for_expert.sum() / total_tokens
            expert_router_loss = frac_of_tokens_routed_to_expert * frac_of_router_prob_routed_to_expert
            router_loss += expert_router_loss
            frac_of_tokens_routed_to_expert_list.append(frac_of_tokens_routed_to_expert)

            # log_rank(f"Expert {expert_idx} has {len(tokens_for_expert)} tokens", logger=logger, level=logging.INFO, rank=0)

        router_aux_loss_coef = 0.001
        router_loss = router_loss * router_aux_loss_coef * self.num_experts
        # ExpertContext.get_instance().push_aux_loss(router_loss)
        return dispatched_inputs, num_tokens_for_experts, router_loss

    def _combine_expert_outputs(self, expert_outputs, routing_indices, original_shape):
        """
        Combines outputs from different experts back to the original tensor layout.
        """
        # todo add shapes
        # Initialize output tensor with zeros
        combined_output = torch.empty(original_shape, device=expert_outputs[0].device, dtype=expert_outputs[0].dtype)

        for expert_idx, expert_output in enumerate(expert_outputs):
            if expert_output.shape[0] == 0:  # Skip if no tokens were routed to this expert
                continue

            # Find positions where this expert was in the top-k
            expert_mask = routing_indices == expert_idx
            combined_output[expert_mask] = expert_output

        return combined_output  # b,s,h

    def _core_forward(self, hidden_states: TensorType["bs*seq_len", "hidden_size"]):
        """Core forward logic for MoE layer."""
        # b, s/2, h
        routing_weights, routing_indices = self._compute_router_probabilities(hidden_states)
        # b, s

        # Dispatch tokens to experts
        dispatched_inputs, num_tokens_for_experts, router_loss = self._dispatch_tokens(
            hidden_states, routing_weights, routing_indices
        )

        expert_outputs = []
        for expert_idx, (inputs, num_tokens_for_expert) in enumerate(zip(dispatched_inputs, num_tokens_for_experts)):
            if num_tokens_for_expert == 0:  # Skip computation if no tokens assigned
                expert_outputs.append(torch.tensor([], device=hidden_states.device, dtype=hidden_states.dtype))
                # todo: just use None instead of tensor([], device=hidden_states.device, dtype=hidden_states.dtype)
                continue

            # Forward through the expert
            expert_output = self.experts[expert_idx](hidden_states=inputs)["hidden_states"]
            expert_outputs.append(expert_output)

        output = self._combine_expert_outputs(expert_outputs, routing_indices, hidden_states.shape)

        # TODO: configurable
        # Add shared expert contribution if enabled
        shared_expert_output = self.shared_expert(hidden_states=hidden_states)["hidden_states"]
        output = output + shared_expert_output

        return output, router_loss

    def _checkpointed_forward(self, hidden_states):
        """Apply gradient checkpointing to save memory during training."""
        return CheckpointFunction.apply(self._core_forward, True, hidden_states)

    def forward(self, hidden_states):
        """Forward pass for the MoE layer."""
        if self.recompute_layer and self.training:
            hidden_states, router_loss = self._checkpointed_forward(hidden_states)
        else:
            hidden_states, router_loss = self._core_forward(hidden_states)

        return {"hidden_states": hidden_states, "router_loss": router_loss}
