from typing import Dict, List, Optional, Tuple, Union

import torch
from flash_attn.modules.mha import flash_attn_varlen_kvpacked_func
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import CheckpointFunction

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import Config, ParallelismArgs
from nanotron.config.models_config import Qwen2Config, RandomInit, SpectralMupInit
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
from nanotron.nn.activations import ACT2FN
from nanotron.nn.attention import ALL_ATTENTION_FUNCTIONS, get_attention_mask
from nanotron.nn.layer_norm import LlamaRMSNorm as RMSNorm
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.nn.rotary import RotaryEmbedding
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import PipelineBlock, TensorPointer
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.tensor_parallel.functional import sharded_cross_entropy
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)
from nanotron.random import RandomStates
from nanotron.scaling.parametrization import SpectralMupParametrizator, StandardParametrizator
from nanotron.logging import LogMixin
from nanotron.nn.llama3_ring_attention import llama3_flash_attn_varlen_kvpacked_func, llama3_flash_attn_prepare_cu_seqlens
logger = logging.get_logger(__name__)


class CoreAttention(nn.Module):
    """Core attention module that can use different attention implementations"""

    def __init__(
        self,
        config: Qwen2Config,
        tp_pg: dist.ProcessGroup,
        cp_pg: dist.ProcessGroup,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.local_num_heads = self.num_heads // tp_pg.size()
        self.local_num_kv_heads = self.num_kv_heads // tp_pg.size()
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )  # Important for transformers's `sdpa_attention_forward`
        self._attn_implementation = config._attn_implementation
        self.cp_pg = cp_pg
        self.sliding_window_size = config.sliding_window_size
        self.simple_causal_mask = True  # Use simple causal mask instead of computing custom attention mask if not document masking / sliding window
        self.flex_attention_mask = config.flex_attention_mask if hasattr(config, "flex_attention_mask") else None

    def forward(
        self,
        query_states: torch.Tensor,  # [b*s, num_heads, head_dim]
        key_states: torch.Tensor,  # [b*s, num_kv_heads, head_dim]
        value_states: torch.Tensor,  # [b*s, num_kv_heads, head_dim]
        position_ids: torch.Tensor,  # [b*s]
        seq_length: Optional[int],
        attention_mask: Optional[torch.Tensor] = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        """Forward pass applying the chosen attention implementation"""
        # Get the appropriate attention function
        attention_func = ALL_ATTENTION_FUNCTIONS[self._attn_implementation]

        # Initialize variables for attention parameters
        cu_seqlens = kwargs.get("cu_seqlens", None)

        # Shape tensors according to attention implementation
        if self._attn_implementation == "ring_flash_triton":
            query_states = query_states.view(-1, seq_length, self.local_num_heads, self.head_dim)
            key_states = key_states.view(-1, seq_length, self.local_num_kv_heads, self.head_dim)
            value_states = value_states.view(-1, seq_length, self.local_num_kv_heads, self.head_dim)
        elif self._attn_implementation == "ring":
            # Warning: Since this uses _flash_attn_varlen_forward make sure we count padding tokens in cu_seqlens
            query_states = query_states.view(-1, self.local_num_heads, self.head_dim)
            key_states = key_states.view(-1, self.local_num_kv_heads, self.head_dim)
            value_states = value_states.view(-1, self.local_num_kv_heads, self.head_dim)
        else:
            # Process attention mask based on implementation
            if self.simple_causal_mask:
                assert attention_mask is None, "Simple causal mask is not supported with custom attention mask"
                assert self.sliding_window_size is None, "Simple causal mask is not supported with sliding window"
            elif attention_mask is None and position_ids is not None:
                # Determine if we need to create an attention mask from position_ids
                if self._attn_implementation == "flex_attention" and self.sliding_window_size is not None:
                    # For FlexAttention with sliding window, we don't need an explicit mask
                    # The mask_mod function will handle it
                    pass
                else:
                    # For other implementations, generate the attention mask if needed
                    # Only calculate if cu_seqlens wasn't passed
                    if cu_seqlens is None:
                        attention_mask, cu_seqlens = get_attention_mask(position_ids, seq_length=seq_length)

                    if attention_mask is not None:
                        # Add batch and head dimensions for proper broadcasting
                        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length, seq_length]

        attn_output = attention_func(
            self,
            query_states,  # [b, num_heads, seq_len, head_dim]
            key_states,  # [b, num_kv_heads, seq_len, head_dim]
            value_states,  # [b, num_kv_heads, seq_len, head_dim]
            attention_mask,  # [b, num_heads, seq_len, seq_len]
            max_seqlen=seq_length,
            dropout=dropout,
            scaling=None,  # by default, scaling is head_dim**-0.5
            sliding_window=self.sliding_window_size,
            ring_pg=self.cp_pg,
            position_ids=position_ids if self._attn_implementation == "flex_attention" else None,
            document_ids=kwargs.get("document_ids", None) if self._attn_implementation == "flex_attention" else None,
            flex_attention_mask=self.flex_attention_mask if self._attn_implementation == "flex_attention" else None,
            **kwargs,  # Pass remaining kwargs
        )[0]

        return attn_output.view(
            -1, self.local_num_heads * self.head_dim
        )  # [b*s, num_heads, head_dim] -> [b*s, num_heads*head_dim]

class Qwen2Attention(LogMixin, nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        cp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.tp_pg_size = tp_pg.size()
        self.cp_pg_size = cp_pg.size()
        self.cp_pg = cp_pg

        # Head configuration
        self.num_heads = config.num_attention_heads
        self.local_num_heads = self.num_heads // self.tp_pg_size

        # KV head configuration
        self.num_kv_heads = config.num_key_value_heads
        self.local_num_kv_heads = self.num_kv_heads // self.tp_pg_size

        # Dimensions
        self.head_dim = config.hidden_size // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.local_q_size = self.local_num_heads * self.head_dim
        self.local_kv_size = self.local_num_kv_heads * self.head_dim

        # TP mode configuration
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        qkv_contiguous_chunks = (
            self.q_size,  # Q chunk size
            self.kv_size,  # K chunk size
            self.kv_size,  # V chunk size
        )
        self.qkv_proj = TensorParallelColumnLinear(
            self.hidden_size,
            self.q_size + 2 * self.kv_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=config.attention_bias,  # Qwen2 uses bias for QKV, Llama doesn't
            async_communication=tp_linear_async_communication,
            contiguous_chunks=qkv_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )
        self.o_proj = TensorParallelRowLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
        )
        if config._use_qkv_packed:
            from nanotron.nn.rotary import FlashRotaryEmbedding
            self.rotary_emb = FlashRotaryEmbedding(
                dim=self.head_dim,
                base=config.rope_theta,
                interleaved=config.rope_interleaved,
                seq_len_interpolation_factor=config.rope_seq_len_interpolation_factor,
            )
        else:
            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                max_seq_len=config.max_position_embeddings,
                base=config.rope_theta,
                interleaved=config.rope_interleaved,
                seq_len_scaling_factor=config.rope_seq_len_scaling_factor,
                fused=config._fused_rotary_emb,
            )
        self.attention = CoreAttention(config, tp_pg, cp_pg, layer_idx)
        self.simple_causal_mask = True
        self._use_qkv_packed = config._use_qkv_packed
        self.sliding_window_size = config.sliding_window_size
        self.log_attn_probs = config.log_attn_probs
        self.heads_k_stride = config.ring_attn_heads_k_stride
        # TODO: support SFT

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch_size*seq_length, hidden_size]
        position_ids: torch.Tensor,  # [batch_size, seq_length] where -1 is padding
        cu_seqlens: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,  # Added cu_seqlens argument
    ):
        # [0, 1, 2, 3, 4, 0, 1, 2, -1, -1, -1] # 2 documents with 5 and 3 tokens then padding
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # 1 document with 11 tokens
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1] # 1 document with 10 tokens then padding
        # Replace -1 with 0 in position_ids to mark every padding token as a separate sequence. Ideally we want to get rid of padding tokens from qkv
        # position_ids = position_ids.masked_fill(position_ids == -1, 0)
        seq_length = position_ids.shape[1] // self.cp_pg_size # in CP, position_ids are global
        # Keep original position_ids shape for return, flatten for internal use
        position_ids = position_ids.view(-1)  # [batch_size*seq_length]

        qkv = self.qkv_proj(hidden_states)

        if self._use_qkv_packed:
            attn_output = self._forward_packed(qkv, seq_length, position_ids, cu_seqlens)
        # else:
        #     q, k, v = qkv.split(
        #         [self.local_q_size, self.local_kv_size, self.local_kv_size], dim=-1
        #     )  # [batch_size*seq_length, q_size], [batch_size*seq_length, kv_size]
        #     q = q.view(-1, self.local_num_heads, self.head_dim)  # [b*s, num_heads, head_dim]
        #     k = k.view(-1, self.local_num_kv_heads, self.head_dim)  # [b*s, num_kv_heads, head_dim]
        #     v = v.view(-1, self.local_num_kv_heads, self.head_dim)  # [b*s, num_kv_heads, head_dim]
        #     if self.config.no_rope_layer is None or (self.layer_idx + 1) % self.config.no_rope_layer != 0:
        #         rotary_pos_emb = self.rotary_emb(
        #             position_ids=position_ids if not self.simple_causal_mask else None, seq_length=seq_length
        #         )  # [b*s, dim] or [seq_length, dim]
        #         q = self.rotary_emb.apply_rotary_pos_emb(
        #             q, rotary_pos_emb, seq_length=seq_length
        #         )  # [b*s, num_heads, head_dim]
        #         k = self.rotary_emb.apply_rotary_pos_emb(
        #             k, rotary_pos_emb, seq_length=seq_length
        #         )  # [b*s, num_kv_heads, head_dim]
        #     else:
        #         log_rank(f"skipping rotary for layer {self.layer_idx + 1}", logger=logger, level=logging.DEBUG, rank=0)
        #     attn_output = self.attention(
        #         q, k, v, position_ids=position_ids, seq_length=seq_length, cu_seqlens=cu_seqlens
        #     )
        output = self.o_proj(attn_output)
        # Return original position_ids shape
        return {"hidden_states": output, "position_ids": position_ids.view(-1, seq_length)}

    def _forward_packed(self, qkv, seq_length, position_ids, cu_seqlens):
        assert cu_seqlens is not None, "cu_seqlens must be provided for packed attention"
        q = qkv[..., : self.local_num_heads * self.head_dim]  # Not contiguous, similar to flash_attn
        kv = qkv[..., self.local_num_heads * self.head_dim :]  # Not contiguous, similar to flash_attn
        q = q.view(-1, seq_length, self.local_num_heads, self.head_dim)
        kv = kv.view(-1, seq_length, 2, self.local_num_kv_heads, self.head_dim)
        if self.config.no_rope_layer is None or (self.layer_idx + 1) % self.config.no_rope_layer != 0:
            seqlen_offset = dist.get_rank(self.cp_pg) * seq_length
            q, kv = self.rotary_emb(
                q, kv, seqlen_offset=seqlen_offset, max_seqlen=seq_length*self.cp_pg_size
            )
        else:
            log_rank(f"skipping rotary for layer {self.layer_idx + 1}", logger=logger, level=logging.DEBUG, rank=0)
            self.sliding_window_size = None # WARNING: we skip sliding window for no-rope

        q = q.view(-1, self.local_num_heads, self.head_dim)
        kv = kv.view(-1, 2, self.local_num_kv_heads, self.head_dim)
        max_seqlen = seq_length  # TODO: should this be max position_ids?


        if self.config._attn_implementation == "llama3_ring_attention":
            attn_output = llama3_flash_attn_varlen_kvpacked_func(
                q,
                kv,
                cu_seqlens_q=cu_seqlens["cu_seqlens_q"],
                cu_seqlens_k=cu_seqlens["cu_seqlens_k"],
                max_seqlen_q=cu_seqlens["max_seqlen_q"],
                max_seqlen_k=cu_seqlens["max_seqlen_k"],
                heads_k_stride=self.heads_k_stride,
                local_k_slice=cu_seqlens["local_k_slice"],
                dropout_p=0.0,
                softmax_scale=None,
                causal=True,
                alibi_slopes=None,
                window_size=(self.sliding_window_size - 1, 0) if self.sliding_window_size is not None else (-1, -1),
                deterministic=False,
                return_attn_probs=self.log_attn_probs,
                group=self.cp_pg,
            )  # Not contiguous, similar to flash_attn
        else:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            attn_output = flash_attn_varlen_kvpacked_func(
                q,
                kv,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                0.0,
                softmax_scale=None,
                causal=True,
                alibi_slopes=None,
                window_size=(self.sliding_window_size - 1, 0) if self.sliding_window_size is not None else (-1, -1),
                deterministic=False,
                return_attn_probs=self.log_attn_probs,
            )  # Not contiguous, similar to flash_attn

        if self.log_attn_probs:
            attn_output, attn_probs, _ = attn_output
            # log attn_probs
            self.tbi_logger({"attn_probs": attn_probs})
        # flash_attn use rearrange instead of reshape https://github.com/Dao-AILab/flash-attention/blob/1a58058a6da83bd7baaf4c512e8a1abe0240bb77/flash_attn/modules/mha.py#L730
        return attn_output.reshape(-1, self.local_num_heads * self.head_dim)  # [b*s, num_heads*head_dim]


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
        self.recompute_layer = parallel_config.recompute_layer

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


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        cp_pg: dist.ProcessGroup,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Use fused RMSNorm if configured
        norm_class = TritonRMSNorm if config._fused_rms_norm else RMSNorm
        self.input_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Qwen2Attention(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
            cp_pg=cp_pg,
            layer_idx=layer_idx,
        )
        self.post_attention_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)

        # Use MoE layer if this layer is in the MoE layers list
        if config.moe_config and layer_idx in config.moe_config.layers:
            self.mlp = Qwen2MoELayer(
                config=config,
                parallel_config=parallel_config,
                tp_pg=tp_pg,
                layer_idx=layer_idx,
            )
        else:
            self.mlp = Qwen2MLP(
                config=config,
                parallel_config=parallel_config,
                tp_pg=tp_pg,
            )

        self.recompute_layer = parallel_config.recompute_layer

    def _core_forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],  # [batch_size*seq_length, hidden_size]
        position_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length] where -1 is padding
        cu_seqlens: Union[torch.Tensor, TensorPointer],
    ) -> List[Union[torch.Tensor, TensorPointer]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        output = self.attn(hidden_states=hidden_states, position_ids=position_ids, cu_seqlens=cu_seqlens)
        hidden_states = output["hidden_states"]
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states=hidden_states)["hidden_states"]
        hidden_states = hidden_states + residual

        return hidden_states, position_ids, cu_seqlens

    def _checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return CheckpointFunction.apply(self._core_forward, True, hidden_states, position_ids, cu_seqlens)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        position_ids: Union[torch.Tensor, TensorPointer],
        cu_seqlens: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        if self.recompute_layer and not isinstance(hidden_states, TensorPointer):
            hidden_states, position_ids, cu_seqlens = self._checkpointed_forward(
                hidden_states, position_ids, cu_seqlens
            )
        else:
            hidden_states, position_ids, cu_seqlens = self._core_forward(hidden_states, position_ids, cu_seqlens)

        return {
            "hidden_states": hidden_states,
            "position_ids": position_ids,
            "cu_seqlens": cu_seqlens,
        }


class Embedding(nn.Module):
    def __init__(self, tp_pg: dist.ProcessGroup, config: Qwen2Config, parallel_config: Optional[ParallelismArgs]):
        super().__init__()
        self.token_embedding = TensorParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            pg=tp_pg,
            mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
        )
        self.pg = tp_pg

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor):  # [batch_size, seq_length]
        input_ids = input_ids.view(-1)  # [batch_size*seq_length]
        input_embeds = self.token_embedding(input_ids)  # [batch_size*seq_length, hidden_size]
        return {"input_embeds": input_embeds, "position_ids": position_ids}


class Qwen2Model(nn.Module):
    """Build pipeline graph for Qwen2 model"""

    def __init__(
        self,
        config: Qwen2Config,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
    ):
        super().__init__()

        # Declare all the nodes
        self.p2p = P2P(parallel_context.pp_pg, device=torch.device("cuda"))
        self.config = config
        self.parallel_config = parallel_config
        self.parallel_context = parallel_context
        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        self.token_position_embeddings = PipelineBlock(
            p2p=self.p2p,
            module_builder=Embedding,
            module_kwargs={
                "config": config,
                "parallel_config": parallel_config,
                "tp_pg": parallel_context.tp_pg,
            },
            module_input_keys={"input_ids", "position_ids"},
            module_output_keys={"input_embeds", "position_ids"},
        )

        # Create decoder layers
        self.decoder = nn.ModuleList(
            [
                PipelineBlock(
                    p2p=self.p2p,
                    module_builder=Qwen2DecoderLayer,
                    module_kwargs={
                        "config": config,
                        "parallel_config": parallel_config,
                        "tp_pg": parallel_context.tp_pg,
                        "cp_pg": parallel_context.cp_pg,
                        "layer_idx": layer_idx,
                    },
                    module_input_keys={"hidden_states", "position_ids", "cu_seqlens"},
                    module_output_keys={"hidden_states", "position_ids", "cu_seqlens"},
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.final_layer_norm = PipelineBlock(
            p2p=self.p2p,
            module_builder=TritonRMSNorm if config._fused_rms_norm else RMSNorm,
            module_kwargs={"hidden_size": config.hidden_size, "eps": config.rms_norm_eps},
            module_input_keys={"input"},
            module_output_keys={"hidden_states"},
        )

        self.lm_head = PipelineBlock(
            p2p=self.p2p,
            # Return sharded logits that will need to be gathered
            module_builder=TensorParallelColumnLinear,
            module_kwargs={
                "in_features": config.hidden_size,
                "out_features": config.vocab_size,
                "pg": parallel_context.tp_pg,
                "bias": False,
                "mode": self.tp_mode,
                "async_communication": tp_linear_async_communication,
                "tp_recompute_allgather": parallel_config.tp_recompute_allgather,
            },
            module_input_keys={"x"},
            module_output_keys={"logits"},
        )

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        position_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length] where -1 is padding
    ):
        output = self.token_position_embeddings(input_ids=input_ids, position_ids=position_ids)
        # Compute cu_seqlens
        cu_seqlens: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None
        if position_ids.numel() > 0:
            start_indices = torch.where(position_ids.view(-1) == 0)[0]
            cu_seqlens = torch.cat(
                [start_indices, torch.tensor([position_ids.numel()], dtype=torch.int32, device=start_indices.device)]
            ).to(torch.int32)

            # llama3 ring attention
            if self.config._attn_implementation == "llama3_ring_attention":
                local_sequence_length = input_ids.shape[1]
                sequence_length = position_ids.shape[1]
                assert sequence_length == local_sequence_length * self.parallel_context.cp_pg.size(), f"sequence_length={sequence_length} must be equal to local_sequence_length={local_sequence_length} * cp_pg.size()={self.parallel_context.cp_pg.size()}"
                assert sequence_length % (2 * self.parallel_context.cp_pg.size()) == 0, f"Sequence length {sequence_length} must be divisible by {2 * self.parallel_context.cp_pg.size()} when using llama3 ring attention"
                (
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                local_k_slice,
                ) = llama3_flash_attn_prepare_cu_seqlens(
                    cu_seqlens, # global cu_seqlens
                    causal=True,
                    rank=self.parallel_context.cp_pg.rank(),
                    world_size=self.parallel_context.cp_pg.size(),
                )
                cu_seqlens = {
                    "cu_seqlens_q": cu_seqlens_q,
                    "cu_seqlens_k": cu_seqlens_k,
                    "max_seqlen_q": max_seqlen_q,
                    "max_seqlen_k": max_seqlen_k,
                    "local_k_slice": local_k_slice,
                }

        decoder_states = {
            "hidden_states": output["input_embeds"],
            "position_ids": output["position_ids"],
            "cu_seqlens": cu_seqlens,
        }

        for decoder_layer in self.decoder:
            decoder_states = decoder_layer(**decoder_states)

        hidden_states = self.final_layer_norm(input=decoder_states["hidden_states"])["hidden_states"]

        sharded_logits = self.lm_head(x=hidden_states)["logits"]

        return sharded_logits

    def get_block_compute_costs(self):
        """Computes the compute cost of each block in the model for load balancing."""
        model_config = self.config
        d_ff = model_config.intermediate_size
        d_qkv = model_config.hidden_size // model_config.num_attention_heads
        block_compute_costs = {
            # Self-attention (qkv proj + attn out) + MLP
            Qwen2DecoderLayer: 4 * model_config.num_attention_heads * d_qkv * model_config.hidden_size
            + 3 * d_ff * model_config.hidden_size,
            # Final LM head
            TensorParallelColumnLinear: model_config.vocab_size * model_config.hidden_size,
        }
        return block_compute_costs

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """Get flops per second for the model"""
        world_size = self.parallel_context.world_pg.size()

        # Get number of KV heads, accounting for potential absence in config
        try:
            num_key_value_heads = self.config.num_key_value_heads
        except AttributeError:
            num_key_value_heads = self.config.num_attention_heads

        model_flops, hardware_flops = get_flops(
            num_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            vocab_size=self.config.vocab_size,
            ffn_hidden_size=self.config.intermediate_size,
            seq_len=sequence_length,
            batch_size=global_batch_size,
        )

        model_flops_per_s = model_flops / (iteration_time_in_sec * world_size * 1e12)
        hardware_flops_per_s = hardware_flops / (iteration_time_in_sec * world_size * 1e12)
        return model_flops_per_s, hardware_flops_per_s


@torch.jit.script
def masked_mean(loss, label_mask, dtype):
    # type: (Tensor, Tensor, torch.dtype) -> Tensor
    return (loss * label_mask).sum(dtype=dtype) / label_mask.sum()


class Loss(nn.Module):
    def __init__(self, tp_pg: dist.ProcessGroup):
        super().__init__()
        self.tp_pg = tp_pg

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [batch_size*seq_length, logits]
        label_ids: torch.Tensor,  # [batch_size, seq_length]
        label_mask: torch.Tensor,  # [batch_size, seq_length]
    ) -> Dict[str, torch.Tensor]:
        sharded_logits = sharded_logits.view(label_ids.shape[0], label_ids.shape[1], -1)
        loss = sharded_cross_entropy(sharded_logits, label_ids.contiguous(), group=self.tp_pg, dtype=torch.float)
        loss = masked_mean(loss, label_mask, dtype=torch.float)
        return {"loss": loss}


class LossWithZLoss(Loss):
    def __init__(self, tp_pg: dist.ProcessGroup, z_loss_coefficient: float):
        super().__init__(tp_pg)
        self.z_loss_coef = z_loss_coefficient

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [batch_size*seq_length, logits]
        label_ids: torch.Tensor,  # [batch_size, seq_length]
        label_mask: torch.Tensor,  # [batch_size, seq_length]
    ) -> Dict[str, torch.Tensor]:
        sharded_logits = sharded_logits.view(label_ids.shape[0], label_ids.shape[1], -1)
        loss, z_loss = sharded_cross_entropy(
            sharded_logits, label_ids.contiguous(), group=self.tp_pg, dtype=torch.float, z_loss_coef=self.z_loss_coef
        )
        loss = masked_mean(loss, label_mask, dtype=torch.float)
        z_loss = masked_mean(z_loss.detach(), label_mask, dtype=torch.float)
        return {"loss": loss, "z_loss": z_loss}

from nanotron.logging import LoggingCollectorMixin
class Qwen2ForTraining(NanotronModel, LoggingCollectorMixin):
    def __init__(
        self,
        config: Qwen2Config,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        super().__init__()
        self.model = Qwen2Model(config=config, parallel_context=parallel_context, parallel_config=parallel_config)

        # Choose the appropriate loss class based on config
        loss_kwargs = {
            "tp_pg": parallel_context.tp_pg,
        }
        if config.z_loss_enabled:
            loss_kwargs["z_loss_coefficient"] = config.z_loss_coefficient

        self.loss = PipelineBlock(
            p2p=self.model.p2p,
            module_builder=LossWithZLoss if config.z_loss_enabled else Loss,
            module_kwargs=loss_kwargs,
            module_input_keys={
                "sharded_logits",
                "label_ids",
                "label_mask",
            },
            module_output_keys={"loss", "z_loss"} if config.z_loss_enabled else {"loss"},
        )
        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        position_ids: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        sharded_logits = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
        )
        loss = self.loss(
            sharded_logits=sharded_logits,
            label_ids=label_ids,
            label_mask=label_mask,
        )
        if self.config.z_loss_enabled:
            return {"loss": loss["loss"], "z_loss": loss["z_loss"]}
        else:
            return {"loss": loss["loss"]}

    @torch.no_grad()
    def init_model_randomly(self, config: Config):
        """Initialize model parameters randomly."""
        init_method = config.model.init_method
        if isinstance(init_method, RandomInit):
            parametrizator_cls = StandardParametrizator
        elif isinstance(init_method, SpectralMupInit):
            parametrizator_cls = SpectralMupParametrizator
        else:
            raise ValueError(f"Unknown init method {init_method}")

        parametrizator = parametrizator_cls(config=config)

        log_rank(
            f"Parametrizing model parameters using {parametrizator.__class__.__name__}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        model = self
        initialized_parameters = set()
        # Handle tensor parallelism
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        # Fix the root_model
        module_id_to_prefix[id(model)] = ""

        for param_name, param in model.named_parameters():
            assert isinstance(param, NanotronParameter)

            module_name, param_name = param_name.rsplit(".", 1)

            if param.is_tied:
                tied_info = param.get_tied_info()
                full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=module_id_to_prefix
                )
            else:
                full_param_name = f"{module_name}.{param_name}"

            if full_param_name in initialized_parameters:
                # Already initialized
                continue

            module = model.get_submodule(module_name)
            parametrizator.parametrize(param_name, module)

            assert full_param_name not in initialized_parameters
            initialized_parameters.add(full_param_name)

        assert initialized_parameters == {
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            if param.is_tied
            else name
            for name, param in model.named_parameters()
        }, f"Somehow the initialized set of parameters don't match:\n - Expected: { {name for name, _ in model.named_parameters()} }\n - Got: {initialized_parameters}"

    def get_embeddings_lm_head_tied_names(self):
        """Get the names of the tied embeddings and lm_head weights"""
        if self.config.tie_word_embeddings is True:
            # Should be similar to ["model.token_position_embeddings.pp_block.token_embedding.weight", "model.lm_head.pp_block.weight"]
            return ["model.token_position_embeddings.pp_block.token_embedding.weight", "model.lm_head.pp_block.weight"]
        else:
            return []

    def get_block_compute_costs(self):
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        return self.model.get_block_compute_costs()

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """Get flops per second for a given model"""
        return self.model.get_flops_per_sec(iteration_time_in_sec, sequence_length, global_batch_size)


def get_flops(
    num_layers,
    hidden_size,
    num_heads,
    num_key_value_heads,
    vocab_size,
    seq_len,
    ffn_hidden_size,
    batch_size=1,
):
    """Counts flops in an decoder-only model
    Args:
        num_layers: number of decoder layers
        hidden_size: hidden size of the model
        num_heads: number of heads in the model
        num_key_value_heads: number of key/value heads in the model
        ffn_hidden_size: hidden size of the FFN
        vocab_size: size of the vocabulary
        seq_len: sequence length of the decoder
        batch_size: batch size
    Returns:
        model_flops: flops in the model (should be independent of the hardware and model implementation)
        hardware_flops: flops in the hardware (actual flops performed on the hardware). Check 6.3 in https://arxiv.org/pdf/2205.05198.pdf
    """
    if num_key_value_heads is None:
        num_key_value_heads = num_heads
    hidden_size_per_head = hidden_size // num_heads
    # In the following we mark the reduced dimension with parentheses
    # decoder
    # self attention
    ## qkv projection
    decoder_qkv_proj_flops_fwd = (
        2 * num_layers * batch_size * seq_len * (hidden_size) * num_heads * hidden_size_per_head
        + 2 * num_layers * batch_size * seq_len * (hidden_size) * 2 * num_key_value_heads * hidden_size_per_head
    )
    ## qk logits
    decoder_qk_logits_flops_fwd = 2 * num_layers * batch_size * num_heads * seq_len * (hidden_size_per_head) * seq_len
    ## v logits
    decoder_v_logits_flops_fwd = 2 * num_layers * batch_size * num_heads * seq_len * (seq_len) * hidden_size_per_head
    ## attn out
    decoder_attn_out_flops_fwd = (
        2 * num_layers * batch_size * num_heads * seq_len * (hidden_size_per_head) * hidden_size
    )
    # FF
    ## 1st layer
    decoder_ffn_1_flops_fwd = 4 * num_layers * batch_size * seq_len * (hidden_size) * ffn_hidden_size
    ## 2nd layer
    decoder_ffn_2_flops_fwd = 2 * num_layers * batch_size * seq_len * (ffn_hidden_size) * hidden_size

    decoder_flops_fwd = (
        decoder_qkv_proj_flops_fwd
        + decoder_qk_logits_flops_fwd
        + decoder_v_logits_flops_fwd
        + decoder_attn_out_flops_fwd
        + decoder_ffn_1_flops_fwd
        + decoder_ffn_2_flops_fwd
    )

    # lm head
    lm_head_flops_fwd = 2 * batch_size * seq_len * (hidden_size) * vocab_size

    # the bwd pass requires double the flops in case of matmuls to calculate the gradients with respect to
    # both input and weight tensors
    model_flops = 3 * (decoder_flops_fwd + lm_head_flops_fwd)  # 1 for fwd + 2 for bwd

    hardware_flops = model_flops  # TODO: This is a placeholder for now

    return model_flops, hardware_flops
