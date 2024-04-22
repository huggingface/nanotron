from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import einsum, rearrange, reduce
from torch import nn

# from jaxtyping import Float, Array
from torchtyping import TensorType

from nanotron.config import LlamaConfig, ParallelismArgs
from nanotron.parallel.tensor_parallel.nn import TensorParallelRowLinear


class InfiniAttention(nn.Module):
    def __init__(
        self, config: LlamaConfig, parallel_config: Optional[ParallelismArgs], tp_pg: dist.ProcessGroup, layer_idx: int
    ):
        super().__init__()

        from nanotron.models.llama import CausalSelfAttention

        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        self.config = config

        # self.balance_factor = 0.5

        self.attn = CausalSelfAttention(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
            layer_idx=layer_idx,
        )

        d_model = config.hidden_size
        self.d_head = config.hidden_size // config.num_attention_heads

        self.o_proj = TensorParallelRowLinear(
            config.num_attention_heads * self.d_head,
            d_model,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
        )

        assert self.o_proj.weight.shape == self.attn.o_proj.weight.shape

        # self.balance_factor = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        hidden_states: TensorType["seq_length", "batch_size", "hidden_size"],
        sequence_mask: TensorType["batch_size", "seq_length"],
    ):
        attn_outputs = self.attn(hidden_states=hidden_states, sequence_mask=sequence_mask, return_qkv_states=True)

        local_attn_outputs = attn_outputs["attention_output"]

        # NOTE: query_states.shape = [batch_size * q_length, self.n_heads, d_qk]
        # NOTE: key_states.shape or value_states.shape = [batch_size * kv_length, self.n_heads, d_qk]
        query_states, key_states, value_states = attn_outputs["qkv_states"]

        batch_size = hidden_states.shape[1]

        query_states = rearrange(
            query_states,
            "(batch_size seq_length) n_heads d_head -> batch_size n_heads seq_length d_head",
            batch_size=batch_size,
        )
        # NOTE: because the number of heads are splited in TP
        # so we find them on the fly
        N_HEADS = query_states.shape[1]

        key_states = rearrange(
            key_states,
            "(batch_size seq_length) n_heads d_head -> batch_size n_heads seq_length d_head",
            batch_size=batch_size,
        )
        value_states = rearrange(
            value_states,
            "(batch_size seq_length) n_heads d_head -> batch_size n_heads seq_length d_head",
            batch_size=batch_size,
        )

        # query_states = query_states.view(batch_size, -1, n_heads, d_head) # [batch_size, q_length, n_heads, d_qk]
        # key_states = key_states.view(batch_size, -1, n_heads, d_head)
        # value_states = value_states.view(batch_size, -1, n_heads, d_head)

        # memory = torch.matmul(key_states, value_states.transpose(-1, -2))
        # normalization = key_states.sum(dim=-1, keepdim=True)

        # memory = einsum(key_states, value_states, 'b n h d, b n h d -> b h d d')
        # memory = einsum(key_states, value_states, 'b i j h, b i l h -> b i j l')

        # memory = einsum(F.elu(key_states) + 1, value_states, 'batch_size k_length n_heads d_head, batch_size v_length n_heads d_head -> batch_size n_heads k_length v_length')
        # normalization = reduce(key_states, 'batch_size seq_length n_heads d_head -> batch_size seq_length n_heads', reduction='sum')

        # query_states = F.elu(query_states) + 1
        # retrieved_memory = einsum(query_states, memory, 'batch_size q_length n_heads d_head, batch_size n_heads k_length v_length -> batch_size q_length s d')
        # retrieved_memory = torch.matmul(query_states, memory) / torch.matmul(query_states, normalization)

        memory, normalization = self._get_memory(key_states, value_states)
        retrieved_memory = self._retrieve_from_memory(query_states, memory, normalization)
        local_attn_outputs = rearrange(
            local_attn_outputs,
            "seq_length batch_size (n_heads d_head) -> batch_size n_heads seq_length d_head",
            d_head=self.d_head,
        )
        # assert self.balance_factor.shape == n_heads

        balance_factors = torch.randn(N_HEADS, device=local_attn_outputs.device, dtype=local_attn_outputs.dtype)

        global_weights = F.sigmoid(balance_factors)
        global_attn_outputs = global_weights[None, :, None, None] * retrieved_memory

        local_weights = F.sigmoid(1 - balance_factors)
        local_attn_outputs = local_weights[None, :, None, None] * local_attn_outputs

        attention_output = global_attn_outputs + local_attn_outputs
        attention_output = rearrange(
            attention_output, "batch_size n_heads seq_len d_head -> batch_size seq_len (n_heads d_head)"
        )

        self.o_proj(attention_output)

        assert 1 == 1
        # NOTE: update memory

    def _get_memory(self, key_states, value_states):
        key_states = F.elu(key_states) + 1
        memory = torch.matmul(key_states.transpose(-2, -1), value_states)
        # memory = einsum(key_states, value_states, 'batch_size n_heads k_length d_head, batch_size n_heads v_length d_head -> batch_size n_heads k_length v_length')
        normalization = reduce(
            key_states, "batch_size n_heads seq_length d_head -> batch_size n_heads d_head", reduction="sum"
        )
        return memory, normalization

    def _retrieve_from_memory(self, query_states, memory, normalization):
        query_states = F.elu(query_states) + 1
        retrieved_memory = einsum(
            query_states,
            memory,
            "batch_size n_heads seq_length d_k, batch_size n_heads d_k d_v -> batch_size n_heads seq_length d_v",
        )

        denominator = einsum(
            query_states,
            normalization,
            "batch_size n_heads seq_length d_k, batch_size n_heads d_k -> batch_size n_heads seq_length",
        )
        # [batch_size, n_heads, seq_length, d_v] / [batch_size, n_heads, seq_length, 1], so each d_v is divide by the normalized value
        retrieved_memory = retrieved_memory / denominator[:, :, :, None]
        return retrieved_memory
