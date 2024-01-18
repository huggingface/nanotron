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
""" PyTorch LLaMa model.
"""
from functools import lru_cache
from typing import Dict, Optional, Union

import torch
from torch import nn
from transformers.activations import ACT2FN

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import LlamaConfig, ParallelismArgs, RecomputeGranularity
from nanotron.logging import log_rank
from nanotron.models import AttachableStore, NanotronModel
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
from nanotron.utils import checkpoint_method

logger = logging.get_logger(__name__)


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.variance_epsilon = eps

    def forward(self, input):
        # TODO @thomasw21: This is actually stupid, it launches too many kernels for a very simple task, maybe I just need to run `torch.optimize`, maybe we need to build our own.
        variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            input = input.to(self.weight.dtype)

        return self.weight * input


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, end: int, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.end = end
        self.theta = theta
        # TODO @nouamane: Figure out why we can't set `DTypeInvariantTensor` ...
        # TODO @thomasw21: Complex buffers break DDP, instead we store float and view them as complex
        self.freqs_cis: torch.Tensor
        self._initialized_buffer = False

    def init_rotary_embeddings(self):
        if self._initialized_buffer is True:
            # Buffer if already initialized
            return

        self.register_buffer(
            "freqs_cis",
            torch.empty(self.end, self.dim // 2, 2, dtype=torch.float, device="cuda"),
            persistent=False,
        )
        assert self.freqs_cis.device.type == "cuda"
        # TODO @nouamane: One we figure out how to do the DTypeInvariantTensor, this can be removed and changed to an assert
        if self.freqs_cis.dtype != torch.float:
            self.freqs_cis = self.freqs_cis.to(torch.float)
        assert self.freqs_cis.dtype == torch.float
        freqs = 1.0 / (
            self.theta
            ** (torch.arange(0, self.dim, 2, dtype=torch.float, device="cuda")[: (self.dim // 2)] / self.dim)
        )
        t = torch.arange(self.end, device="cuda")
        freqs = torch.outer(t, freqs).float()
        complex_freqs = torch.polar(torch.ones_like(freqs), freqs)
        freqs = torch.view_as_real(complex_freqs)
        self.freqs_cis.copy_(freqs)
        self._initialized_buffer = True

    def forward(
        self,
        x: torch.Tensor,  # [batch_size, num_heads, seq_length, inner_dim]
        position_ids: Optional[torch.LongTensor],  # [batch_size, seq_length]
    ):
        batch_size, num_heads, seq_length, inner_dim = x.shape
        if (
            position_ids is not None and position_ids[-1, -1] >= self.end
        ) or seq_length >= self.end:  # TODO @nouamane: check if this causes cpu-gpu sync
            self.end *= 2
            self._initialized_buffer = False
        if self._initialized_buffer is False:
            self.init_rotary_embeddings()
        dtype = x.dtype
        assert inner_dim % 2 == 0
        x = x.view(
            batch_size, num_heads, seq_length, inner_dim // 2, 2
        )  # [batch_size, num_heads, q_length, inner_dim]
        if x.dtype == torch.bfloat16:
            x = x.float()
        complex_x = torch.view_as_complex(x)
        if position_ids is None:
            freqs_cis = self.freqs_cis[None, None, :seq_length, :]
        else:
            freqs_cis = self.freqs_cis[position_ids][:, None, :, :]
        complex_freqs = torch.view_as_complex(freqs_cis)
        x_out = torch.view_as_real(complex_x * complex_freqs).view(batch_size, num_heads, seq_length, inner_dim)
        return x_out.type(dtype)


class MLP(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        parallel_config: Optional["ParallelismArgs"],
        tp_pg: dist.ProcessGroup,
    ):
        super().__init__()

        # TODO @thomasw21: refactor so that we store that default in a single place.
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        gate_up_contiguous_chunks = (
            config.intermediate_size,  # shape of gate_linear
            config.intermediate_size,  # shape of up_linear
        )
        self.gate_up_proj = TensorParallelColumnLinear(
            config.hidden_size,
            2 * config.intermediate_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=gate_up_contiguous_chunks,
        )

        self.down_proj = TensorParallelRowLinear(
            config.intermediate_size,
            config.hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication and tp_mode is TensorParallelLinearMode.REDUCE_SCATTER,
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):  # [seq_length, batch_size, hidden_dim]
        merged_states = self.gate_up_proj(hidden_states)
        gate_states, up_states = torch.split(merged_states, merged_states.shape[-1] // 2, dim=-1)
        # TODO @nouamane: how can I fuse self.act with self.down_proj?
        hidden_states = self.down_proj(self.act(gate_states) * up_states)
        return {"hidden_states": hidden_states}


class CoreAttention(nn.Module):
    def __init__(self, config: LlamaConfig, parallel_config: Optional["ParallelismArgs"], layer_idx: int):
        super().__init__()
        # TODO @thomasw21: GPT has a weird `d_kv` config which I'm guessing is essentically a `d_qkv`
        assert (
            config.hidden_size % config.num_attention_heads == 0
        ), f"Hidden size {config.hidden_size} must be divisible by number of attention heads {config.num_attention_heads}."
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads

        self.checkpoint_attention = (
            parallel_config is not None and parallel_config.recompute_granularity is RecomputeGranularity.SELECTIVE
        )

    @checkpoint_method(attr_name="checkpoint_attention")
    def forward(
        self,
        query_states: torch.Tensor,  # [batch_size, num_heads, q_length, inner_dim]
        key_states: torch.Tensor,  # [batch_size, num_heads, kv_length, inner_dim]
        value_states: torch.Tensor,  # [batch_size, num_heads, kv_length, inner_dim]
        attention_mask: torch.Tensor,  # torch.BoolTensor [batch_size, num_heads, q_length, kv_length] (can be broadcasted to that size)
    ):
        # TODO @thomasw21: Megatron-LM stores states in (length, batch_size, num_heads * inner_dim). Maybe that's a bit faster.

        batch_size, n_heads, q_length, _ = query_states.shape
        kv_length = key_states.shape[2]

        scores = torch.bmm(
            query_states.view(batch_size * n_heads, q_length, self.d_qk),
            key_states.view(batch_size * n_heads, kv_length, self.d_qk).transpose(1, 2),
        )

        scores = scores / (self.d_qk**0.5)

        dtype = scores.dtype
        if scores.dtype == torch.float16:
            scores = scores.float()

        scores = torch.masked_fill(
            scores.view(batch_size, n_heads, q_length, kv_length),
            ~attention_mask,
            torch.finfo(scores.dtype).min,
        )
        attn_weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(dtype=dtype)

        attn_output = torch.matmul(attn_weights, value_states).view(
            batch_size, n_heads, q_length, self.d_v
        )  # (batch_size, n_heads, seq_length, dim)

        return attn_output


def _prepare_causal_attention_mask(q_sequence_mask: torch.Tensor, k_sequence_mask: torch.Tensor) -> torch.BoolTensor:
    """
    Prepare causal attention mask used for multi-head self-attention. (False upper)
    Adapted from transformers.models.bloom.modeling_bloom.BloomModel._prepare_attn_mask

    Input:
    q_sequence_mask: [batch_size, query_length]
    k_sequence_mask: [batch_size, key_length]
    Output:
    [batch_size, 1, query_length, key_length]

    Note:
    The dimension 1 is added to be broadcastable to [batch_size, number_of_heads, query_length, key_length].
    """
    _, key_length = k_sequence_mask.shape
    if key_length > 1:
        causal_mask = ~_make_causal_mask(
            q_sequence_mask_shape=q_sequence_mask.shape,
            k_sequence_mask_shape=k_sequence_mask.shape,
            device=q_sequence_mask.device,
        )  # False upper [batch_size, 1, query_length, key_length]
        combined_attention_mask = causal_mask * k_sequence_mask[:, None, None, :]
    else:
        combined_attention_mask = k_sequence_mask[:, None, None, :]
    return combined_attention_mask


class CausalSelfAttention(nn.Module, AttachableStore):
    def __init__(
        self,
        config: LlamaConfig,
        parallel_config: Optional["ParallelismArgs"],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        super().__init__()
        # Tensor parallel considerations: We split tensors along head dimension
        assert (
            config.num_attention_heads % tp_pg.size() == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by TP size ({tp_pg.size()})."
        try:
            assert (
                config.num_key_value_heads % tp_pg.size() == 0
            ), f"Number of key/value heads ({config.num_key_value_heads}) must be divisible by TP size ({tp_pg.size()})."
        except AttributeError:
            log_rank(
                "WARNING: num_key_value_heads not defined, assuming it is equal to num_attention_heads",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
            # If num_key_value_heads is not defined, we assume that it is equal to num_attention_heads
            config.num_key_value_heads = config.num_attention_heads
        assert (
            config.num_attention_heads % config.num_key_value_heads == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by number of key/value heads ({config.num_key_value_heads})."
        self.n_local_q_heads = config.num_attention_heads // tp_pg.size()
        self.n_local_kv_heads = config.num_key_value_heads // tp_pg.size()
        self.n_repeats = config.num_attention_heads // config.num_key_value_heads
        self.is_gqa = config.num_attention_heads != config.num_key_value_heads  # Whether we are using GQA or not
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.d_model = config.hidden_size

        # TODO @thomasw21: refactor so that we store that default in a single place.
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        # build the slice config for self.qkv for save/load
        # shard are done within the contiguous chunk
        qkv_contiguous_chunks = (
            config.num_attention_heads * self.d_qk,  # shape of q
            config.num_key_value_heads * self.d_qk,  # shape of k
            config.num_key_value_heads * self.d_qk,  # shape of v
        )
        self.qkv_proj = TensorParallelColumnLinear(
            self.d_model,
            config.num_attention_heads * self.d_qk + 2 * config.num_key_value_heads * self.d_qk,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=qkv_contiguous_chunks,
        )

        self.rotary_embedding = RotaryEmbedding(dim=self.d_qk, end=config.max_position_embeddings)

        self.o_proj = TensorParallelRowLinear(
            config.num_attention_heads * self.d_qk,
            self.d_model,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication and tp_mode is TensorParallelLinearMode.REDUCE_SCATTER,
        )

        self.attention = CoreAttention(
            config,
            parallel_config=parallel_config,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states,  # (seq_length, batch_size, hidden_size)
        sequence_mask,  # (batch_size, seq_length)
    ):
        qkv_states = self.qkv_proj(
            hidden_states
        )  # [seq_length, batch_size, n_local_q_heads * d_qk + 2 * n_local_kv_heads]
        q_length, batch_size, _ = qkv_states.size()

        if self.is_gqa:
            query_states, key_states, value_states = torch.split(
                qkv_states,
                [
                    self.n_local_q_heads * self.d_qk,
                    self.n_local_kv_heads * self.d_qk,
                    self.n_local_kv_heads * self.d_qk,
                ],
                dim=-1,
            )

            query_states = (
                query_states.contiguous()
                .view(q_length, batch_size, self.n_local_q_heads, self.d_qk)
                .transpose(0, 1)
                .transpose(1, 2)
            )
            key_states = (
                key_states.contiguous()
                .view(q_length, batch_size, self.n_local_kv_heads, self.d_qk)
                .transpose(0, 1)
                .transpose(1, 2)
            )

            value_states = (
                value_states.contiguous()
                .view(q_length, batch_size, self.n_local_kv_heads, self.d_qk)
                .transpose(0, 1)
                .transpose(1, 2)
            )
        else:
            query_states, key_states, value_states = (
                qkv_states.view(q_length, batch_size, 3, self.n_local_q_heads, self.d_qk)
                .permute(2, 1, 3, 0, 4)
                .contiguous()
            )  # [3, batch_size, n_local_q_heads, seq_length, d_qk]

        # Get cached key/values from store if available
        store = self.get_local_store()
        if store is not None:
            # Double check that we use store only at inference time
            assert key_states.requires_grad is False
            assert value_states.requires_grad is False

            if "position_offsets" in store:
                old_position_offsets = store["position_offsets"]
                position_ids = old_position_offsets[:, None] + sequence_mask
            else:
                position_ids = torch.cumsum(sequence_mask, dim=-1) - 1

            # Compute rotary embeddings
            position_ids.masked_fill_(~sequence_mask, 0)
            query_states = self.rotary_embedding(query_states, position_ids=position_ids)
            key_states = self.rotary_embedding(key_states, position_ids=position_ids)

            # Pull pre-computed key/value states
            if "key" in store:
                # We assume that "key"/"value"/"sequence_mask" are all added once initialized
                old_key = store["key"]
                old_value = store["value"]
                old_sequence_mask = store["sequence_mask"]

                # Concatenate with new key/value
                key_states = torch.concat([old_key, key_states], dim=-2)
                value_states = torch.concat([old_value, value_states], dim=-2)
                all_sequence_mask = torch.concat([old_sequence_mask, sequence_mask], dim=-1)
                attention_mask = _prepare_causal_attention_mask(
                    q_sequence_mask=sequence_mask,
                    k_sequence_mask=all_sequence_mask,
                )  # (batch_size, 1, query_length, key_length) (True upper)
            else:
                attention_mask = _prepare_causal_attention_mask(
                    q_sequence_mask=sequence_mask,
                    k_sequence_mask=sequence_mask,
                )  # (batch_size, 1, query_length, key_length) (True upper)
                all_sequence_mask = sequence_mask

            # Store new key/value in store
            position_offsets = position_ids[:, -1]
            store.update(
                {
                    "key": key_states,
                    "value": value_states,
                    "sequence_mask": all_sequence_mask,
                    "position_offsets": position_offsets,
                }
            )
        else:
            # Apply rotary embeddings to query/key states
            query_states = self.rotary_embedding(query_states, position_ids=None)
            key_states = self.rotary_embedding(key_states, position_ids=None)

            attention_mask = _prepare_causal_attention_mask(
                q_sequence_mask=sequence_mask,
                k_sequence_mask=sequence_mask,
            )  # (batch_size, 1, query_length, key_length) (True upper)

        if self.is_gqa:
            kv_length = key_states.shape[-2]
            key_states = (
                key_states[:, :, None, :, :]
                .expand(batch_size, self.n_local_kv_heads, self.n_repeats, kv_length, self.d_qk)
                .contiguous()
                .view(batch_size, self.n_local_q_heads, kv_length, self.d_qk)
            )  # [batch_size * kv_length, self.n_local_q_heads, d_qk]
            value_states = (
                value_states[:, :, None, :, :]
                .expand(batch_size, self.n_local_kv_heads, self.n_repeats, kv_length, self.d_qk)
                .contiguous()
                .view(batch_size, self.n_local_q_heads, kv_length, self.d_qk)
            )

        attention_output = self.attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
        )

        attention_output = (
            attention_output.permute(2, 0, 1, 3)
            .contiguous()
            .view(q_length, batch_size, self.n_local_q_heads * self.d_v)
        )
        output = self.o_proj(attention_output)

        return {"hidden_states": output, "sequence_mask": sequence_mask}


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        parallel_config: Optional["ParallelismArgs"],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
            layer_idx=layer_idx,
        )

        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config=config, parallel_config=parallel_config, tp_pg=tp_pg)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        output = self.attn(hidden_states=hidden_states, sequence_mask=sequence_mask)
        hidden_states = output["hidden_states"]
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states=hidden_states)["hidden_states"]
        hidden_states = hidden_states + residual

        return {
            "hidden_states": hidden_states,
            "sequence_mask": output["sequence_mask"],
        }


@lru_cache(maxsize=1)
def _make_causal_mask(
    q_sequence_mask_shape: torch.Size,
    k_sequence_mask_shape: torch.Size,
    device: torch.device,
) -> torch.BoolTensor:
    """
    Make causal mask used for self-attention. (True upper). The mask is broadcasted to
    shape (batch_size, 1, query_length, key_length) from (query_length, key_length).
    """
    batch_size, query_length = q_sequence_mask_shape
    batch_size, key_length = k_sequence_mask_shape
    past_key_length = key_length - query_length
    mask = torch.empty((query_length, key_length), dtype=torch.bool, device=device)
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(query_length, device=device)
    mask[:, past_key_length:] = seq_ids[:, None] < seq_ids[None, :]
    if past_key_length > 0:
        mask[:, :past_key_length] = False

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, query_length, key_length)
    return expanded_mask


class Embedding(nn.Module, AttachableStore):
    def __init__(self, tp_pg: dist.ProcessGroup, config: LlamaConfig, parallel_config: Optional["ParallelismArgs"]):
        super().__init__()
        self.token_embedding = TensorParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            pg=tp_pg,
            mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
        )
        self.pg = tp_pg

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor):  # [batch_size, seq_length]
        store = self.get_local_store()
        if store is not None:
            if "past_length" in store:
                past_length = store["past_length"]
            else:
                past_length = torch.zeros(1, dtype=torch.long, device=input_ids.device).expand(input_ids.shape[0])

            cumsum_mask = input_mask.cumsum(-1, dtype=torch.long)
            # Store new past_length in store
            store["past_length"] = past_length + cumsum_mask[:, -1]

        # Format input in `[seq_length, batch_size]` to support high TP with low batch_size
        input_ids = input_ids.transpose(0, 1)
        input_embeds = self.token_embedding(input_ids)
        return {"input_embeds": input_embeds}


class LlamaModel(nn.Module):
    """Build pipeline graph"""

    def __init__(
        self,
        config: LlamaConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional["ParallelismArgs"],
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
                "tp_pg": parallel_context.tp_pg,
                "config": config,
                "parallel_config": parallel_config,
            },
            module_input_keys={"input_ids", "input_mask"},
            module_output_keys={"input_embeds"},
        )

        self.decoder = nn.ModuleList(
            [
                PipelineBlock(
                    p2p=self.p2p,
                    module_builder=LlamaDecoderLayer,
                    module_kwargs={
                        "config": config,
                        "parallel_config": parallel_config,
                        "tp_pg": parallel_context.tp_pg,
                        "layer_idx": layer_idx,
                    },
                    module_input_keys={"hidden_states", "sequence_mask"},
                    module_output_keys={"hidden_states", "sequence_mask"},
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.final_layer_norm = PipelineBlock(
            p2p=self.p2p,
            module_builder=RMSNorm,
            module_kwargs={"normalized_shape": config.hidden_size, "eps": config.rms_norm_eps},
            module_input_keys={"input"},
            module_output_keys={"hidden_states"},
        )  # TODO

        self.lm_head = PipelineBlock(
            p2p=self.p2p,
            # Understand that this means that we return sharded logits that are going to need to be gathered
            module_builder=TensorParallelColumnLinear,
            module_kwargs={
                "in_features": config.hidden_size,
                "out_features": config.vocab_size,
                "pg": parallel_context.tp_pg,
                "bias": False,
                # TODO @thomasw21: refactor so that we store that default in a single place.
                "mode": self.tp_mode,
                "async_communication": tp_linear_async_communication,
            },
            module_input_keys={"x"},
            module_output_keys={"logits"},
        )

        self.cast_to_fp32 = PipelineBlock(
            p2p=self.p2p,
            module_builder=lambda: lambda x: x.float(),
            module_kwargs={},
            module_input_keys={"x"},
            module_output_keys={"output"},
        )

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        input_mask: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
    ):
        return self.forward_with_hidden_states(input_ids=input_ids, input_mask=input_mask)[0]

    def forward_with_hidden_states(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        input_mask: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
    ):
        # all tensors are optional as most ranks don't need anything from the dataloader.

        output = self.token_position_embeddings(input_ids=input_ids, input_mask=input_mask)

        hidden_encoder_states = {
            "hidden_states": output["input_embeds"],
            "sequence_mask": input_mask,
        }
        for encoder_block in self.decoder:
            hidden_encoder_states = encoder_block(**hidden_encoder_states)

        hidden_states = self.final_layer_norm(input=hidden_encoder_states["hidden_states"])["hidden_states"]

        sharded_logits = self.lm_head(x=hidden_states)["logits"]

        fp32_sharded_logits = self.cast_to_fp32(x=sharded_logits)["output"]

        return fp32_sharded_logits, hidden_states

    def get_block_compute_costs(self):
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        model_config = self.config
        d_ff = model_config.intermediate_size
        d_qkv = model_config.hidden_size // model_config.num_attention_heads
        block_compute_costs = {
            # CausalSelfAttention (qkv proj + attn out) + MLP
            LlamaDecoderLayer: 4 * model_config.num_attention_heads * d_qkv * model_config.hidden_size
            + 3 * d_ff * model_config.hidden_size,
            # This is the last lm_head
            TensorParallelColumnLinear: model_config.vocab_size * model_config.hidden_size,
        }
        return block_compute_costs

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """Get flops per second for a given model"""
        world_size = self.parallel_context.world_pg.size()
        try:
            num_key_values_heads = self.config.num_key_value_heads
        except AttributeError:
            num_key_values_heads = self.config.num_attention_heads

        model_flops, hardware_flops = get_flops(
            num_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_key_value_heads=num_key_values_heads,
            vocab_size=self.config.vocab_size,
            ffn_hidden_size=self.config.intermediate_size,
            seq_len=sequence_length,
            batch_size=global_batch_size,
            recompute_granularity=self.parallel_config.recompute_granularity,
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
        sharded_logits: torch.Tensor,  # [seq_length, batch_size, logits]
        label_ids: torch.Tensor,  # [batch_size, seq_length]
        label_mask: torch.Tensor,  # [batch_size, seq_length]
    ) -> Dict[str, torch.Tensor]:
        # Megatron by defaults cast everything in fp32. `--f16-lm-cross-entropy` is an option you can use to keep current precision.
        # https://github.com/NVIDIA/Megatron-LM/blob/f267e6186eae1d6e2055b412b00e2e545a8e896a/megatron/model/gpt_model.py#L38
        loss = sharded_cross_entropy(
            sharded_logits, label_ids.transpose(0, 1).contiguous(), group=self.tp_pg, dtype=torch.float
        ).transpose(0, 1)
        # TODO @thomasw21: It's unclear what kind of normalization we want to do.
        loss = masked_mean(loss, label_mask, dtype=torch.float)
        # I think indexing causes a sync we don't actually want
        # loss = loss[label_mask].sum()
        return {"loss": loss}


class LlamaForTraining(NanotronModel):
    def __init__(
        self,
        config: LlamaConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional["ParallelismArgs"],
        random_states: Optional[RandomStates] = None,
    ):
        super().__init__()
        self.model = LlamaModel(config=config, parallel_context=parallel_context, parallel_config=parallel_config)
        self.loss = PipelineBlock(
            p2p=self.model.p2p,
            module_builder=Loss,
            module_kwargs={"tp_pg": parallel_context.tp_pg},
            module_input_keys={
                "sharded_logits",
                "label_ids",
                "label_mask",
            },
            module_output_keys={"loss"},
        )
        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        sharded_logits = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
        )
        loss = self.loss(
            sharded_logits=sharded_logits,
            label_ids=label_ids,
            label_mask=label_mask,
        )["loss"]
        return {"loss": loss}

    @torch.no_grad()
    def init_model_randomly(self, init_method, scaled_init_method):
        """Initialize model parameters randomly.
        Args:
            init_method (callable): Used for embedding/position/qkv weight in attention/first layer weight of mlp/ /lm_head/
            scaled_init_method (callable): Used for o weight in attention/second layer weight of mlp/

        Note:
            Layernorm weight all 0 or 1 depending on `apply_layernorm_1p`
        """
        model = self
        initialized_parameters = set()
        # Handle tensor parallelism
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        # Fix the root_model
        module_id_to_prefix[id(model)] = ""

        for module_name, module in model.named_modules():
            if isinstance(module, TensorParallelColumnLinear):
                # Somehow Megatron-LM does something super complicated, https://github.com/NVIDIA/Megatron-LM/blob/2360d732a399dd818d40cbe32828f65b260dee11/megatron/core/tensor_parallel/layers.py#L96
                # What it does:
                #  - instantiate a buffer of the `full size` in fp32
                #  - run init method on it
                #  - shard result to get only a specific shard
                # Instead I'm lazy and just going to run init_method, since they are scalar independent
                assert {"weight"} == {name for name, _ in module.named_parameters()} or {"weight"} == {
                    name for name, _ in module.named_parameters()
                }
                for param_name, param in module.named_parameters():
                    assert isinstance(param, NanotronParameter)
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

                    if "weight" == param_name:
                        init_method(param)
                    elif "bias" == param_name:
                        param.zero_()
                    else:
                        raise ValueError(f"Who the fuck is {param_name}?")

                    assert full_param_name not in initialized_parameters
                    initialized_parameters.add(full_param_name)
            elif isinstance(module, TensorParallelRowLinear):
                # Somehow Megatron-LM does something super complicated, https://github.com/NVIDIA/Megatron-LM/blob/2360d732a399dd818d40cbe32828f65b260dee11/megatron/core/tensor_parallel/layers.py#L96
                # What it does:
                #  - instantiate a buffer of the `full size` in fp32
                #  - run init method on it
                #  - shard result to get only a specific shard
                # Instead I'm lazy and just going to run init_method, since they are scalar independent
                assert {"weight"} == {name for name, _ in module.named_parameters()} or {"weight"} == {
                    name for name, _ in module.named_parameters()
                }
                for param_name, param in module.named_parameters():
                    assert isinstance(param, NanotronParameter)
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

                    if "weight" == param_name:
                        scaled_init_method(param)
                    elif "bias" == param_name:
                        param.zero_()
                    else:
                        raise ValueError(f"Who the fuck is {param_name}?")

                    assert full_param_name not in initialized_parameters
                    initialized_parameters.add(full_param_name)
            elif isinstance(module, RMSNorm):
                assert {"weight"} == {name for name, _ in module.named_parameters()}
                for param_name, param in module.named_parameters():
                    assert isinstance(param, NanotronParameter)
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

                    if "weight" == param_name:
                        # TODO @thomasw21: Sometimes we actually want 0
                        param.fill_(1)
                    elif "bias" == param_name:
                        param.zero_()
                    else:
                        raise ValueError(f"Who the fuck is {param_name}?")

                    assert full_param_name not in initialized_parameters
                    initialized_parameters.add(full_param_name)
            elif isinstance(module, TensorParallelEmbedding):
                # TODO @thomasw21: Handle tied embeddings
                # Somehow Megatron-LM does something super complicated, https://github.com/NVIDIA/Megatron-LM/blob/2360d732a399dd818d40cbe32828f65b260dee11/megatron/core/tensor_parallel/layers.py#L96
                # What it does:
                #  - instantiate a buffer of the `full size` in fp32
                #  - run init method on it
                #  - shard result to get only a specific shard
                # Instead I'm lazy and just going to run init_method, since they are scalar independent
                assert {"weight"} == {name for name, _ in module.named_parameters()}

                assert isinstance(module.weight, NanotronParameter)
                if module.weight.is_tied:
                    tied_info = module.weight.get_tied_info()
                    full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                        module_id_to_prefix=module_id_to_prefix
                    )
                else:
                    full_param_name = f"{module_name}.weight"

                if full_param_name in initialized_parameters:
                    # Already initialized
                    continue

                init_method(module.weight)
                assert full_param_name not in initialized_parameters
                initialized_parameters.add(full_param_name)

        assert initialized_parameters == {
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            if param.is_tied
            else name
            for name, param in model.named_parameters()
        }, f"Somehow the initialized set of parameters don't match:\n - Expected: { {name for name, _ in model.named_parameters()} }\n - Got: {initialized_parameters}"

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
    recompute_granularity=None,
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
        recompute_granularity: Activation recomputation method. Either None, FULL or SELECTIVE. Check Megatron-LM docs for more info.
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

    if recompute_granularity is None:
        hardware_flops = model_flops
    elif recompute_granularity is RecomputeGranularity.FULL:
        # Note: we don't recompute lm head activs
        hardware_flops = model_flops + decoder_flops_fwd  # + activ recomputation
    elif recompute_granularity is RecomputeGranularity.SELECTIVE:
        # all terms with s^2 are flops that are recomputed
        # ref. appendix A: https://arxiv.org/pdf/2205.05198.pdf
        recomputed_decoder_flops = decoder_qk_logits_flops_fwd + decoder_v_logits_flops_fwd
        hardware_flops = model_flops + recomputed_decoder_flops
    else:
        raise ValueError("recompute_granularity must be one of 'full' or 'selective'")

    return model_flops, hardware_flops
