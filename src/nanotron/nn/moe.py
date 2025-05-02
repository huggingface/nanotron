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
from nanotron.nn.load_balancing_loss import MoEAuxLossAutoScaler, switch_aux_loss

logger = logging.get_logger(__name__)


try:
    import grouped_gemm.ops as ops
except ImportError:
    raise RuntimeError(
        "Grouped GEMM is not available. Please run `pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@main` (takes less than 5 minutes)"
    )

# Try to debug the topk version
topk_version = True


class Router(nn.Module):
    def __init__(
        self, config: Qwen2Config, parallel_config: Optional[ParallelismArgs], tp_pg: dist.ProcessGroup, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.parallel_config = parallel_config
        self.tp_pg = tp_pg
        self.layer_idx = layer_idx

        self.num_experts = config.moe_config.num_experts
        self.num_experts_per_token = config.moe_config.top_k

        self.aux_loss_coeff = 1e2  # TODO: add to config
        self.load_balancing_type = None  # 'aux_loss' #TODO: add to config
        self.sequence_partition_group = None  # TODO: tp_cp group

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
        if self.load_balancing_type is None:
            routing_weights, routing_indices = self.top_k_softmax(logits)
        elif self.load_balancing_type == "aux_loss":
            routing_weights, routing_indices = self.apply_aux_loss(logits)
        else:
            raise ValueError(f"Invalid load balancing type: {self.load_balancing_type}")
        return routing_weights, routing_indices

    def make_print_param_grad_hook(self, param_name):
        def hook(grad):
            print(f"[Layer {self.layer_idx}] Gradient for parameter '{param_name}':")
            print(f"Mean: {grad.mean().item():.6f}, Std: {grad.std().item():.6f}")
            print(f"Min/Max: {grad.min().item():.6f} / {grad.max().item():.6f}\n")

        return hook

    def forward(self, x: torch.Tensor):
        logits = self.gating(x)

        routing_weights, routing_indices = self.routing(logits)

        # # debug
        # x.register_hook(self.make_print_param_grad_hook("hidden_states"))
        # logits.register_hook(self.make_print_param_grad_hook("logits"))
        # routing_weights.register_hook(self.make_print_param_grad_hook("routing_weights"))

        return routing_weights, routing_indices

    def top_k_softmax(self, logits: torch.Tensor):
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        routing_weights, routing_indices = torch.topk(
            probs, k=self.num_experts_per_token, dim=-1
        )  # [num_tokens, num_experts_per_token]

        # fail to converge
        if topk_version:
            return routing_weights, routing_indices

        # converge
        topk_masked_gates = torch.zeros_like(logits).scatter(
            1, routing_indices, routing_weights
        )  # [num_tokens, num_experts]
        topk_map = torch.zeros_like(logits).int().scatter(1, routing_indices, 1)  # [num_tokens, num_experts]
        return topk_masked_gates, topk_map

        # routing_indices = routing_indices.to(torch.int32)  # NOTE: ops.permute requires indices to be int32

    def apply_aux_loss(self, logits: torch.Tensor):
        # doesn't converge
        # probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        # routing_weights, routing_indices = torch.topk(probs, k=self.num_experts_per_token, dim=-1)
        # tokens_per_expert = torch.bincount(routing_indices.flatten(), minlength=self.num_experts)
        # aux_loss = switch_aux_loss(probs, tokens_per_expert, self.aux_loss_coeff, self.num_experts_per_token, self.sequence_partition_group)
        # probs = MoEAuxLossAutoScaler.apply(probs, aux_loss)
        # return routing_weights, routing_indices

        # converge
        probs, routing_map = self.top_k_softmax(logits)
        tokens_per_expert = routing_map.sum(dim=0)
        aux_loss = switch_aux_loss(
            probs, tokens_per_expert, self.aux_loss_coeff, self.num_experts_per_token, self.sequence_partition_group
        )
        probs = MoEAuxLossAutoScaler.apply(probs, aux_loss)

        # # debug print
        # if self.layer_idx == 0:
        #     print(f"Aux loss value: {aux_loss.item()}")  # See the actual value
        return probs, routing_map


class GroupedMLP(nn.Module):
    def __init__(self, config: Qwen2Config, parallel_config: Optional[ParallelismArgs]):
        super().__init__()

        num_local_experts = config.moe_config.num_experts // parallel_config.expert_parallel_size
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
        merged_states = ops.gmm(hidden_states, self.merged_gate_up_proj, num_tokens_per_expert, trans_b=False)
        gate_states, up_states = torch.split(merged_states, merged_states.shape[-1] // 2, dim=-1)
        hidden_states = self.act(gate_states) * up_states
        hidden_states = ops.gmm(hidden_states, self.merged_down_proj, num_tokens_per_expert, trans_b=False)

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
        self.num_local_experts = (
            config.moe_config.num_experts // parallel_config.expert_parallel_size
        )  # Experts per device
        self.num_experts_per_token = config.moe_config.top_k  # Number of experts used per token (top-k)
        self.expert_parallel_size = parallel_config.expert_parallel_size
        self.num_local_experts = self.num_experts // self.expert_parallel_size  # Experts per device

        # Get TP mode configuration

        # Router for selecting experts
        self.router = Router(config, parallel_config, tp_pg, layer_idx)

        # Enable shared experts if configured
        self.enable_shared_expert = config.moe_config.enable_shared_expert
        if self.enable_shared_expert:
            from nanotron.models.qwen import Qwen2MLP

            self.shared_expert = Qwen2MLP(
                config=config,
                parallel_config=parallel_config,
                tp_pg=tp_pg,
                intermediate_size=config.moe_config.shared_expert_intermediate_size,
            )
            # TODO: duplicte the shared expert gate
            self.shared_expert_gate = nn.Linear(
                self.hidden_size,
                1,
                bias=False,
            )  # TODO: ensure shared_expert_gate is tied across TP

        # Create the expert MLPs
        self.experts = GroupedMLP(config, parallel_config)
        # Whether to recompute MoE layer during backward pass for memory efficiency
        self.recompute_layer = parallel_config.recompute_layer

    def _dispatch_tokens(
        self,
        hidden_states: torch.Tensor,
        routing_indices: torch.Tensor,
    ):
        """
        Dispatches tokens to their selected experts.
        In a full implementation, this would handle the actual token routing logic
        including communication between devices.
        """
        # NOTE: start from expert 0 to expert n
        num_tokens_per_expert = torch.bincount(
            routing_indices.flatten(), minlength=self.num_local_experts
        )  # [num_local_experts]
        dispatched_inputs, inverse_permute_mapping = ops.permute(hidden_states, routing_indices)
        return dispatched_inputs, inverse_permute_mapping, num_tokens_per_expert

    def _combine_expert_outputs(self, expert_outputs, inverse_mapping, routing_weights):
        """
        Combines outputs from different experts back to the original tensor layout.
        """
        hidden_states = ops.unpermute(expert_outputs, inverse_mapping, routing_weights)
        return hidden_states

    def permute(self, hidden_states, probs, routing_map):
        num_tokens, _ = hidden_states.shape
        num_experts = routing_map.shape[1]
        routing_map = routing_map.bool().T.contiguous()  # [num_experts, num_tokens]
        token_indices = (
            torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
        )  # [num_experts, num_tokens]
        sorted_indices = token_indices.masked_select(routing_map)  # [num_tokens*top_k]
        permuted_hidden_states = hidden_states.index_select(0, sorted_indices)  # [num_tokens*top_k, hidden_size]
        permuted_probs = probs.T.masked_select(routing_map)  # [num_tokens*top_k, num_experts]
        return permuted_hidden_states, permuted_probs, sorted_indices

    def unpermute(self, permuted_hidden_states, permuted_probs, sorted_indices, hidden_state_shape):
        input_dtype = permuted_hidden_states.dtype
        permuted_hidden_states = permuted_hidden_states * permuted_probs.unsqueeze(-1)
        unpermuted_hidden_states = torch.zeros(
            hidden_state_shape, dtype=permuted_hidden_states.dtype, device=permuted_hidden_states.device
        )
        unpermuted_hidden_states.scatter_add_(
            0, sorted_indices.unsqueeze(1).expand(-1, unpermuted_hidden_states.shape[1]), permuted_hidden_states
        )
        return unpermuted_hidden_states.to(input_dtype)

    def _core_forward(self, hidden_states):
        """Core forward logic for MoE layer."""
        # Get top-k routing weights and indices
        routing_weights, routing_indices = self.router(hidden_states)  # [num_tokens, num_experts_per_token]

        # Dispatch tokens to experts
        if topk_version:
            dispatched_inputs, inverse_permute_mapping, num_tokens_per_expert = self._dispatch_tokens(
                hidden_states, routing_indices
            )

            expert_outputs = self.experts(dispatched_inputs, num_tokens_per_expert)

            output = self._combine_expert_outputs(
                expert_outputs["hidden_states"], inverse_permute_mapping, routing_weights
            )

        else:
            num_tokens_per_expert = routing_indices.sum(dim=0).long()  # [num_local_experts]
            # permute
            permuted_hidden_states, permuted_probs, sorted_indices = self.permute(
                hidden_states, routing_weights, routing_indices
            )

            # Matrix multiplication
            output = self.experts(permuted_hidden_states, num_tokens_per_expert)

            # unpermute
            output = self.unpermute(output["hidden_states"], permuted_probs, sorted_indices, hidden_states.shape)

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
