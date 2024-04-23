from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import einsum, rearrange, reduce
from torch import nn

# from jaxtyping import Float, Array
from torchtyping import TensorType

from nanotron.config import LlamaConfig, ParallelismArgs
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)


class InfiniAttention(nn.Module):
    def __init__(
        self, config: LlamaConfig, parallel_config: Optional[ParallelismArgs], tp_pg: dist.ProcessGroup, layer_idx: int
    ):
        super().__init__()

        self.n_segments = 50

        from nanotron.models.llama import CausalSelfAttention

        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        self.config = config

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
        # NOTE: because the number of heads are splited in TP
        # so we find them on the fly
        self.n_local_heads = self.attn.n_local_q_heads

        device = self.o_proj.weight.device
        dtype = self.o_proj.weight.dtype
        # self.balance_factors = TensorParallelRowLinear(config.num_attention_heads * self.d_head, 1, pg=tp_pg, mode=tp_mode)
        self.balance_factors = nn.Parameter(torch.randn(self.n_local_heads, device=device, dtype=dtype))

        # assert self.o_proj.weight.shape == self.attn.o_proj.weight.shape

        # self.balance_factor = nn.Parameter(torch.tensor(0.5))

        # for p in self.attn.o_proj.parameters():
        #     p.requires_grad = False

    def forward(
        self,
        hidden_states: TensorType["sharded_seq_length", "batch_size", "hidden_size"],
        sequence_mask: TensorType["batch_size", "seq_length"],
    ):
        batch_size = hidden_states.shape[1]
        seq_len = hidden_states.shape[0]
        segment_length = seq_len // self.n_segments
        hidden_size = hidden_states.shape[2]

        segment_hidden_states = torch.chunk(hidden_states, chunks=self.n_segments, dim=0)
        segment_sequence_masks = torch.chunk(sequence_mask, chunks=self.n_segments, dim=1)

        memory = None
        normalization = None

        outputs = []

        # sequence_masks = []
        for segment_hidden_state, segment_sequence_mask in zip(segment_hidden_states, segment_sequence_masks):
            attn_outputs = self.attn(
                hidden_states=segment_hidden_state, sequence_mask=segment_sequence_mask, return_qkv_states=True
            )

            local_attn_outputs = attn_outputs["attention_output"]
            query_states, key_states, value_states = attn_outputs["qkv_states"]

            query_states = rearrange(
                query_states,
                "(batch_size seq_len) n_heads d_head -> batch_size n_heads seq_len d_head",
                batch_size=batch_size,
                # n_heads=self.n_local_heads,
            )
            key_states = rearrange(
                key_states,
                "(batch_size seq_len) n_heads d_head -> batch_size n_heads seq_len d_head",
                batch_size=batch_size,
                # n_heads=self.n_local_heads,
            )
            value_states = rearrange(
                value_states,
                "(batch_size seq_len) n_heads d_head -> batch_size n_heads seq_len d_head",
                batch_size=batch_size,
                # n_heads=self.n_local_heads,
            )

            # assert query_states.shape == (batch_size, self.n_local_heads, segment_length, self.d_head)
            assert query_states.shape == key_states.shape == value_states.shape

            retrieved_memory = self._retrieve_from_memory(
                query_states, prev_memory=memory, prev_normalization=normalization
            )
            # retrieved_memory = retrieved_memory.detach()

            local_attn_outputs = rearrange(
                local_attn_outputs,
                "seq_len batch_size (n_heads d_head) -> batch_size n_heads seq_len d_head",
                d_head=self.d_head,
            )

            global_weights = F.sigmoid(self.balance_factors)
            global_weights = rearrange(global_weights, "n_heads -> 1 n_heads 1 1")

            local_weights = 1 - F.sigmoid(self.balance_factors)
            local_weights = rearrange(local_weights, "n_heads -> 1 n_heads 1 1")

            # assert torch.allclose(global_weights + local_weights, torch.ones_like(global_weights))

            attention_output = global_weights * retrieved_memory + local_weights * local_attn_outputs

            # assert attention_output.shape == (batch_size, self.n_local_heads, segment_length, self.d_head)

            attention_output = rearrange(
                attention_output, "batch_size n_heads seq_len d_head -> seq_len batch_size (n_heads d_head)"
            )

            output = self.o_proj(attention_output)

            assert output.shape == (segment_length, batch_size, hidden_size)

            memory, normalization = self._update_memory(memory, normalization, key_states, value_states)
            # memory = memory.detach()
            # normalization = normalization.detach()

            outputs.append(output)

            # NOTE: update memory
        outputs = torch.cat(outputs, dim=0)  # concat along sequence dimension
        assert outputs.shape == hidden_states.shape

        # sequence_masks = torch.cat(sequence_masks, dim=1)
        # assert sequence_masks.shape == sequence_mask.shape
        return_outputs = {"hidden_states": outputs, "sequence_mask": sequence_mask}
        return return_outputs

    def _retrieve_from_memory(self, query_states, prev_memory, prev_normalization):
        # return torch.zeros_like(query_states)
        if prev_memory is None:
            return torch.zeros_like(query_states)

        query_states = F.elu(query_states) + 1
        retrieved_memory = einsum(
            query_states,
            prev_memory,
            "batch_size n_heads seq_length d_k, batch_size n_heads d_k d_v -> batch_size n_heads seq_length d_v",
        )

        denominator = einsum(
            query_states,
            prev_normalization,
            "batch_size n_heads seq_length d_head, batch_size n_heads d_head -> batch_size n_heads seq_length",
        )
        denominator = rearrange(denominator, "batch_size n_heads seq_length -> batch_size n_heads seq_length 1")
        # [batch_size, n_heads, seq_length, d_v] / [batch_size, n_heads, seq_length, 1], so each d_v is divide by the normalized value

        # NOTE: because normalization is the sum of all the keys, so each word should have the same normalization
        retrieved_memory = retrieved_memory / denominator
        return retrieved_memory

    def _update_memory(self, prev_memory, prev_normalization, key_states, value_states):
        TYPE = "delta"
        key_states = F.elu(key_states) + 1

        if TYPE == "linear":
            # memory = torch.matmul(key_states.transpose(-2, -1), value_states)
            new_value_states = value_states
        else:
            if prev_memory is None or prev_normalization is None:
                new_value_states = value_states
            else:
                numerator = einsum(
                    key_states,
                    prev_memory,
                    "batch_size n_heads seq_length d_k, batch_size n_heads d_k d_v -> batch_size n_heads seq_length d_v",
                )

                # denominator = einsum(
                #     key_states, prev_normalization,
                #     "batch_size n_heads seq_len d_head, batch_size n_heads d_head -> batch_size seq_len"
                # )
                denominator = einsum(
                    key_states,
                    prev_normalization,
                    "batch_size n_heads seq_length d_k, batch_size n_heads d_k -> batch_size n_heads seq_length",
                )
                denominator = rearrange(
                    denominator, "batch_size n_heads seq_length -> batch_size n_heads seq_length 1"
                )

                prev_v = numerator / denominator
                new_value_states = value_states - prev_v

        memory = torch.matmul(key_states.transpose(-2, -1), new_value_states)

        # memory = einsum(key_states, value_states, 'batch_size n_heads k_length d_head, batch_size n_heads v_length d_head -> batch_size n_heads k_length v_length')
        normalization = reduce(
            key_states, "batch_size n_heads seq_length d_head -> batch_size n_heads d_head", reduction="sum"
        )

        memory += prev_memory if prev_memory is not None else 0
        normalization += prev_normalization if prev_normalization is not None else 0

        return memory, normalization
