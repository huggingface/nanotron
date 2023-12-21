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
""" PyTorch GPT model with Multi-Query Attention."""

import os
from functools import lru_cache
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import LayerNorm
from transformers import GPTBigCodeConfig
from transformers.activations import ACT2FN

from nanotron.config import ParallelismArgs, RecomputeGranularity
from nanotron.core import distributed as dist
from nanotron.core.distributed import get_global_rank
from nanotron.core.parallel.parameters import NanotronParameter
from nanotron.core.parallel.pipeline_parallelism.block import PipelineBlock
from nanotron.core.parallel.pipeline_parallelism.p2p import P2P
from nanotron.core.parallel.pipeline_parallelism.tensor_pointer import TensorPointer
from nanotron.core.parallel.sharded_parameters import SplitConfig, mark_all_parameters_in_module_as_sharded
from nanotron.core.parallel.tensor_parallelism.distributed_differentiable_primitives import (
    differentiable_all_gather,
    differentiable_identity,
)
from nanotron.core.parallel.tensor_parallelism.functional import sharded_cross_entropy
from nanotron.core.parallel.tensor_parallelism.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)
from nanotron.core.parallel.tied_parameters import create_tied_parameter
from nanotron.core.process_groups import DistributedProcessGroups
from nanotron.core.random import RandomStates, branch_random_state
from nanotron.core.utils import checkpoint_method
from nanotron.models import AttachableStore, NanotronModel


class MLP(nn.Module):
    def __init__(
        self,
        config: GPTBigCodeConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
    ):
        super().__init__()

        # TODO @thomasw21: refactor so that we store that default in a single place.
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        d_ff = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
        self.c_fc = TensorParallelColumnLinear(
            config.hidden_size,
            d_ff,
            pg=tp_pg,
            mode=tp_mode,
            bias=True,
            async_communication=tp_linear_async_communication,
        )
        self.act = ACT2FN[config.activation_function]
        self.c_proj = TensorParallelRowLinear(
            d_ff,
            config.hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=True,
            async_communication=tp_linear_async_communication and tp_mode is TensorParallelLinearMode.REDUCE_SCATTER,
        )

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return {"hidden_states": hidden_states}


class CoreMQA(nn.Module):
    """
    Attention module similar to CoreAttention where only the query is multi-headed.
    """

    def __init__(self, config: GPTBigCodeConfig, parallel_config: Optional[ParallelismArgs], layer_idx: int):
        super().__init__()
        assert (
            config.hidden_size % config.num_attention_heads == 0
        ), f"Hidden size {config.hidden_size} must be divisible by number of attention heads {config.num_attention_heads}."
        self.d_qk = config.hidden_size // config.num_attention_heads
        # we still divide the value dimension by the number of heads https://arxiv.org/pdf/1911.02150.pdf
        self.d_v = config.hidden_size // config.num_attention_heads
        self.dropout = config.attn_pdrop

        self.softmax_dtype = torch.float32

        self.scale_factor = 1.0

        if config.scale_attn_weights:
            self.scale_factor = self.scale_factor / (self.d_qk**0.5)

        self.checkpoint_attention = (
            parallel_config is not None and parallel_config.recompute_granularity is RecomputeGranularity.SELECTIVE
        )

    @checkpoint_method(attr_name="checkpoint_attention")
    def forward(
        self,
        query_states: torch.Tensor,  # [batch_size, q_length, num_heads, inner_dim]
        key_states: torch.Tensor,  # [batch_size, kv_length, inner_dim]
        value_states: torch.Tensor,  # [batch_size, kv_length, inner_dim]
        attention_mask: torch.Tensor,  # torch.BoolTensor [batch_size, num_heads, q_length, kv_length] (can be broadcasted to that size)
    ):
        batch_size, q_length, num_heads, inner_dim = query_states.shape
        kv_length = key_states.shape[1]
        query_states = query_states.view(
            (batch_size, q_length * num_heads, inner_dim)
        )  # [batch_size, q_length * num_heads, inner_dim]
        key_states = key_states.transpose(1, 2)  # [batch_size, inner_dim, kv_length]
        value_states = value_states  # [batch_size, kv_length, inner_dim]
        scores = torch.bmm(query_states, key_states)  # [batch_size, q_length * num_heads, kv_length]

        if self.scale_factor != 1.0:
            scores = scores * self.scale_factor

        dtype = scores.dtype
        if scores.dtype != self.softmax_dtype:
            scores = scores.to(self.softmax_dtype)

        scores = torch.masked_fill(
            scores.view((batch_size, q_length, num_heads, kv_length)),
            ~attention_mask.transpose(1, 2),
            torch.finfo(scores.dtype).min,
        ).view((batch_size, q_length * num_heads, kv_length))

        attn_weights = nn.functional.softmax(scores, dim=-1, dtype=self.softmax_dtype).to(
            dtype=dtype
        )  # [batch_size, q_length * num_heads, kv_length]

        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_weights, value_states)  # [batch_size, q_length * num_heads, inner_dim]

        attn_output = attn_output.view(
            (batch_size, q_length, num_heads, inner_dim)
        )  # [batch_size, q_length, num_heads, inner_dim]
        return attn_output


class MQAColumnLinears(nn.Module):
    def __init__(
        self,
        in_features: int,
        q_out_features: int,
        kv_out_features: int,
        pg: dist.ProcessGroup,
        mode: TensorParallelLinearMode,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.pg = pg
        self.world_size = pg.size()

        assert in_features % self.world_size == 0

        self.in_features = in_features
        self.q_out_features = q_out_features // self.world_size
        self.kv_out_features = kv_out_features

        # torch modules
        self.q = nn.Linear(self.in_features, self.q_out_features, bias=bias, device=device, dtype=dtype)
        self.kv = nn.Linear(self.in_features, self.kv_out_features, bias=bias, device=device, dtype=dtype)

        # Tp mode
        self.mode = mode

        # Marking as tied/sharded
        mark_all_parameters_in_module_as_sharded(self.q, pg=self.pg, split_config=SplitConfig(split_dim=0))
        self._mark_kv_parameters_in_module_as_tied()

    def _mark_kv_parameters_in_module_as_tied(self):
        for name, param in list(self.kv.named_parameters()):
            new_param = create_tied_parameter(
                parameter=param,
                name=name,
                global_ranks=tuple(sorted((get_global_rank(self.pg, i) for i in range(self.pg.size())))),
                # Always has to be ReduceOp SUM as now this is always duplicated regardless of tp mode
                reduce_op=dist.ReduceOp.SUM,
                root_module=self.kv,
            )
            setattr(self.kv, name, new_param)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode is TensorParallelLinearMode.ALL_REDUCE:
            x = differentiable_identity(x, group=self.pg)
        elif self.mode is TensorParallelLinearMode.REDUCE_SCATTER:
            x = differentiable_all_gather(x, group=self.pg)
        else:
            raise ValueError(f"Got unexpected mode: {self.mode}.")

        return self.q(x), self.kv(x)


class CausalSelfMQA(nn.Module, AttachableStore):
    def __init__(
        self,
        config: GPTBigCodeConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        super().__init__()
        # Tensor parallel considerations: We split tensors along head dimension
        assert (
            config.num_attention_heads % tp_pg.size() == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by TP size ({tp_pg.size()})."
        self.tp_pg_size = tp_pg.size()
        self.n_heads = config.num_attention_heads // tp_pg.size()
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.d_model = config.hidden_size

        # TODO @thomasw21: refactor so that we store that default in a single place.
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        self.mode = tp_mode
        self.pg = tp_pg

        # only Q_size is parallelized
        self.qkv = MQAColumnLinears(
            in_features=self.d_model,
            q_out_features=config.num_attention_heads * self.d_qk,
            kv_out_features=self.d_qk + self.d_v,
            pg=tp_pg,
            mode=tp_mode,
            bias=True,
        )

        self.o = TensorParallelRowLinear(
            config.num_attention_heads * self.d_v,
            self.d_model,
            pg=tp_pg,
            mode=tp_mode,
            bias=True,
            async_communication=tp_linear_async_communication and tp_mode is TensorParallelLinearMode.REDUCE_SCATTER,
        )

        self.attention = CoreMQA(
            config,
            parallel_config=parallel_config,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states,  # [seq_length, batch_size, hidden_dim]
        sequence_mask,  # [batch_size, seq_length]
    ):
        # get query/key/value states
        query_states, kv_states = self.qkv(
            hidden_states
        )  # [seq_length, batch_size, self.n_heads * d_qk], [batch_size, seq_length, d_qk + d_v]

        # Get cached key/values from store if available
        store = self.get_local_store()
        if store is not None:
            # Double check that we use store only at inference time
            assert kv_states.requires_grad is False

            # Pull pre-computed key/value states
            if "kv_states" in store:
                # We assume that "key"/"value"/"sequence_mask" are all added once initialized
                old_kv_states = store["kv_states"]
                old_kv_sequence_mask = store["kv_sequence_mask"]

                # Concatenate with new key/value on seq_length dim
                kv_states = torch.cat([old_kv_states, kv_states], dim=0)
                kv_sequence_mask = torch.cat([old_kv_sequence_mask, sequence_mask], dim=-1)
                q_sequence_mask = sequence_mask
            else:
                q_sequence_mask = sequence_mask
                kv_sequence_mask = sequence_mask

            # Store new key/value in store
            store.update({"kv_states": kv_states, "kv_sequence_mask": kv_sequence_mask})
        else:
            q_sequence_mask = sequence_mask
            kv_sequence_mask = sequence_mask

        query_states = (
            query_states.view(query_states.shape[:2] + (self.n_heads, self.d_qk)).transpose(0, 1).contiguous()
        )  # [batch_size, q_length, self.n_heads, d_qk]
        kv_states = kv_states.transpose(0, 1).contiguous()
        key_states, value_states = torch.split(kv_states, [self.d_qk, self.d_v], dim=-1)
        # [batch_size, kv_length, d_qk + d_v]

        attention_mask = _prepare_causal_attention_mask(
            q_sequence_mask=q_sequence_mask,
            k_sequence_mask=kv_sequence_mask,
        )  # [batch_size, 1, q_length, kv_length] (False upper)

        attn_output = self.attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
        )

        attn_output = (
            attn_output.view(attn_output.shape[:2] + (self.n_heads * self.d_v,)).transpose(0, 1).contiguous()
        )  # [q_length, batch_size, self.n_heads * d_v]
        output = self.o(attn_output)

        return {"hidden_states": output, "sequence_mask": sequence_mask}


class GPTBlock(nn.Module):
    def __init__(
        self,
        config: GPTBigCodeConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        random_states: RandomStates,
        layer_idx: int,
    ):
        super(GPTBlock, self).__init__()
        self.ln_1 = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfMQA(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
            layer_idx=layer_idx,
        )
        self.attn_dropout = nn.Dropout(config.resid_pdrop)

        self.ln_2 = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ff = MLP(config=config, parallel_config=parallel_config, tp_pg=tp_pg)
        self.ff_dropout = nn.Dropout(config.resid_pdrop)

        self.random_states = random_states
        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE

    def forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],  # [seq_length, batch_size, hidden_dim]
        sequence_mask: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        output = self.attn(hidden_states=hidden_states, sequence_mask=sequence_mask)

        with branch_random_state(
            self.random_states, "tp_synced", enabled=self.tp_mode is TensorParallelLinearMode.ALL_REDUCE
        ):
            hidden_states = self.attn_dropout(output["hidden_states"])

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.ff(hidden_states=hidden_states)["hidden_states"]

        with branch_random_state(
            self.random_states, "tp_synced", enabled=self.tp_mode is TensorParallelLinearMode.ALL_REDUCE
        ):
            hidden_states = self.ff_dropout(hidden_states)

        hidden_states = residual + hidden_states
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
    Make causal mask used for self-attention.
    1 is considered attended, 0 is masked.
    The mask is broadcasted to shape (batch_size, 1, query_length, key_length) from (query_length, key_length).
    """
    batch_size, query_length = q_sequence_mask_shape
    batch_size, key_length = k_sequence_mask_shape
    past_key_length = key_length - query_length
    mask = torch.empty((query_length, key_length), dtype=torch.bool, device=device)
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(query_length, device=device)
    mask[:, past_key_length:] = seq_ids[:, None] >= seq_ids[None, :]
    if past_key_length > 0:
        mask[:, :past_key_length] = True

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, query_length, key_length)
    return expanded_mask


def _prepare_causal_attention_mask(q_sequence_mask: torch.Tensor, k_sequence_mask: torch.Tensor) -> torch.BoolTensor:
    """
    Prepare causal attention mask used for multi-head self-attention. (False upper)
    1 are considered tokens that are kept, and 0 are tokens that are masked
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
        causal_mask = _make_causal_mask(
            q_sequence_mask_shape=q_sequence_mask.shape,
            k_sequence_mask_shape=k_sequence_mask.shape,
            device=q_sequence_mask.device,
        )  # False upper [batch_size, 1, query_length, key_length]
        combined_attention_mask = causal_mask * k_sequence_mask[:, None, None, :]
    else:
        combined_attention_mask = k_sequence_mask[:, None, None, :]
    return combined_attention_mask


class Embedding(nn.Module, AttachableStore):
    def __init__(self, tp_pg: dist.ProcessGroup, config: GPTBigCodeConfig, parallel_config: Optional[ParallelismArgs]):
        super().__init__()
        self.token_embedding = TensorParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            pg=tp_pg,
            mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
        )
        self.position_embedding = TensorParallelEmbedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
            pg=tp_pg,
            mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
        )

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor):
        store = self.get_local_store()
        if store is not None:
            if "past_length" in store:
                past_length = store["past_length"]
            else:
                past_length = torch.zeros(1, dtype=torch.long, device=input_ids.device).expand(input_ids.shape[0])

            cumsum_mask = input_mask.cumsum(-1, dtype=torch.long)
            position_ids = cumsum_mask + (past_length[:, None] - 1)
            # Store new past_length in store
            store["past_length"] = past_length + cumsum_mask[:, -1]

        else:
            position_ids = input_mask.cumsum(-1, dtype=torch.long) - 1

        # fill the -1 position with value 0
        position_ids.masked_fill_(input_mask == 0, 0)

        # Format input in `[seq_length, batch_size]` to support high TP with low batch_size
        input_ids = input_ids.transpose(0, 1)
        position_ids = position_ids.transpose(0, 1)
        input_embeds = self.token_embedding(input_ids)
        position_bias = self.position_embedding(position_ids)
        return {"input_embeds": input_embeds + position_bias}


class GPTModel(nn.Module):
    """Build pipeline graph"""

    def __init__(
        self,
        config: GPTBigCodeConfig,
        dpg: DistributedProcessGroups,
        parallel_config: Optional[ParallelismArgs],
        random_states: RandomStates,
    ):
        super().__init__()

        # Declare all the nodes
        self.p2p = P2P(dpg.pp_pg, device=torch.device("cuda"))
        self.random_states = random_states
        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE

        self.token_position_embeddings = PipelineBlock(
            p2p=self.p2p,
            module_builder=Embedding,
            module_kwargs={
                "tp_pg": dpg.tp_pg,
                "config": config,
                "parallel_config": parallel_config,
            },
            module_input_keys={"input_ids", "input_mask"},
            module_output_keys={"input_embeds"},
        )

        self.embeds_dropout = PipelineBlock(
            p2p=self.p2p,
            module_builder=nn.Dropout,
            module_kwargs={"p": config.embd_pdrop},
            module_input_keys={"input"},
            module_output_keys={"hidden_states"},
        )

        self.decoder = nn.ModuleList(
            [
                PipelineBlock(
                    p2p=self.p2p,
                    module_builder=GPTBlock,
                    module_kwargs={
                        "config": config,
                        "parallel_config": parallel_config,
                        "tp_pg": dpg.tp_pg,
                        "random_states": random_states,
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
            module_builder=LayerNorm,
            module_kwargs={"normalized_shape": config.hidden_size, "eps": config.layer_norm_epsilon},
            module_input_keys={"input"},
            module_output_keys={"hidden_states"},
        )

        self.lm_head = PipelineBlock(
            p2p=self.p2p,
            # Understand that this means that we return sharded logits that are going to need to be gathered
            module_builder=TensorParallelColumnLinear,
            module_kwargs={
                "in_features": config.hidden_size,
                "out_features": config.vocab_size,
                "pg": dpg.tp_pg,
                "bias": False,
                # TODO @thomasw21: refactor so that we store that default in a single place.
                "mode": self.tp_mode,
                "async_communication": parallel_config.tp_linear_async_communication
                if parallel_config is not None
                else False,
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
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
    ):
        # all tensors are optional as most ranks don't need anything from the dataloader.

        input_embeds = self.token_position_embeddings(input_ids=input_ids, input_mask=input_mask)["input_embeds"]

        with branch_random_state(
            self.random_states, "tp_synced", enabled=self.tp_mode == TensorParallelLinearMode.ALL_REDUCE
        ):
            hidden_states = self.embeds_dropout(input=input_embeds)["hidden_states"]

        hidden_encoder_states = {"hidden_states": hidden_states, "sequence_mask": input_mask}
        for encoder_block in self.decoder:
            hidden_encoder_states = encoder_block(**hidden_encoder_states)

        hidden_states = self.final_layer_norm(input=hidden_encoder_states["hidden_states"])["hidden_states"]

        sharded_logits = self.lm_head(x=hidden_states)["logits"]

        fp32_sharded_logits = self.cast_to_fp32(x=sharded_logits)["output"]

        return fp32_sharded_logits


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
        loss = (loss * label_mask).sum(dtype=torch.float) / label_mask.sum()
        # I think indexing causes a sync we don't actually want
        # loss = loss[label_mask].sum()
        return {"loss": loss}


class GPTForTraining(NanotronModel):
    def __init__(
        self,
        config: GPTBigCodeConfig,
        dpg: DistributedProcessGroups,
        parallel_config: Optional[ParallelismArgs],
        random_states: RandomStates,
    ):
        super().__init__()
        self.model = GPTModel(config=config, dpg=dpg, parallel_config=parallel_config, random_states=random_states)
        self.loss = PipelineBlock(
            p2p=self.model.p2p,
            module_builder=Loss,
            module_kwargs={"tp_pg": dpg.tp_pg},
            module_input_keys={
                "sharded_logits",
                "label_ids",
                "label_mask",
            },
            module_output_keys={"loss"},
        )
        self.config = config
        self.parallel_config = parallel_config
        self.dpg = dpg

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
    ) -> Union[torch.Tensor, TensorPointer]:
        sharded_logits = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
        )
        return {
            "loss": self.loss(
                sharded_logits=sharded_logits,
                label_ids=label_ids,
                label_mask=label_mask,
            )["loss"]
        }

    @torch.no_grad()
    def init_model_randomly(self, init_method, scaled_init_method):
        model = self
        # Set to 0: layernorm bias / all bias
        initialized_parameters = set()
        # Handle tensor parallelism
        with torch.no_grad():
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
                    assert {"weight", "bias"} == {name for name, _ in module.named_parameters()} or {"weight"} == {
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
                    assert {"weight", "bias"} == {name for name, _ in module.named_parameters()} or {"weight"} == {
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
                elif isinstance(module, LayerNorm):
                    assert {"weight", "bias"} == {name for name, _ in module.named_parameters()}
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
                elif os.environ.get("USE_FAST") and isinstance(module, MQAColumnLinears):
                    # TODO @thomasw21: Handle the non fast version
                    # Somehow Megatron-LM does something super complicated, https://github.com/NVIDIA/Megatron-LM/blob/2360d732a399dd818d40cbe32828f65b260dee11/megatron/core/tensor_parallel/layers.py#L96
                    # What it does:
                    #  - instantiate a buffer of the `full size` in fp32
                    #  - run init method on it
                    #  - shard result to get only a specific shard
                    # Instead I'm lazy and just going to run init_method, since they are scalar independent
                    # TODO @thomasw21: handle the case there's no bias
                    assert {"q.weight", "q.bias", "kv.weight", "kv.bias"} == {
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

                        if ".weight" in param_name:
                            init_method(param)
                        elif ".bias" in param_name:
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

    @staticmethod
    def get_embeddings_lm_head_tied_names():
        return [
            "model.token_position_embeddings.pp_block.token_embedding.weight",
            "model.lm_head.pp_block.weight",
        ]

    def get_block_compute_costs(self):
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        model_config = self.config
        d_ff = model_config.n_inner if model_config.n_inner is not None else 4 * model_config.hidden_size
        d_qkv = model_config.hidden_size // model_config.num_attention_heads
        block_compute_costs = {
            # CausalSelfAttention (qkv proj + attn out) + MLP
            GPTBlock: 4 * model_config.num_attention_heads * d_qkv * model_config.hidden_size
            + 2 * d_ff * model_config.hidden_size,
            # This is the last lm_head
            TensorParallelColumnLinear: model_config.vocab_size * model_config.hidden_size,
        }
        return block_compute_costs

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """Get flops per second for a given model"""
        world_size = self.dpg.world_pg.size()
        model_flops, hardware_flops = get_flops(
            num_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            vocab_size=self.config.vocab_size,
            ffn_hidden_size=self.config.n_inner if self.config.n_inner is not None else 4 * self.config.hidden_size,
            seq_len=sequence_length,
            batch_size=global_batch_size,
            recompute_granularity=self.parallel_config.recompute_granularity,
            kv_channels=None,
            glu_activation=False,
        )
        model_flops_per_s = model_flops / (iteration_time_in_sec * world_size * 1e12)
        hardware_flops_per_s = hardware_flops / (iteration_time_in_sec * world_size * 1e12)
        return model_flops_per_s, hardware_flops_per_s


def get_flops(
    num_layers,
    hidden_size,
    num_heads,
    vocab_size,
    seq_len,
    kv_channels=None,
    ffn_hidden_size=None,
    batch_size=1,
    recompute_granularity=None,
    glu_activation=False,
):
    """Counts flops in an decoder-only model
    Args:
        num_layers: number of decoder layers
        hidden_size: hidden size of the model
        num_heads: number of heads in the model
        kv_channels: hidden size of the key and value heads
        ffn_hidden_size: hidden size of the FFN
        vocab_size: size of the vocabulary
        seq_len: sequence length of the decoder
        batch_size: batch size
        recompute_granularity: Activation recomputation method. Either None, FULL or SELECTIVE. Check Megatron-LM docs for more info.
        glu_activation: Whether to use GLU activation in FFN. Check T5 v1.1 for more info.
    Returns:
        model_flops: flops in the model (should be independent of the hardware and model implementation)
        hardware_flops: flops in the hardware (actual flops performed on the hardware). Check 6.3 in https://arxiv.org/pdf/2205.05198.pdf
    """

    if kv_channels is None:
        assert hidden_size % num_heads == 0
        kv_channels = hidden_size // num_heads
    if ffn_hidden_size is None:
        ffn_hidden_size = 4 * hidden_size

    # In the following we mark the reduced dimension with parentheses
    # decoder
    # self attention
    ## qkv projection
    decoder_qkv_proj_flops_fwd = 2 * num_layers * batch_size * seq_len * (hidden_size) * 3 * num_heads * kv_channels
    ## qk logits
    decoder_qk_logits_flops_fwd = 2 * num_layers * batch_size * num_heads * seq_len * (kv_channels) * seq_len
    ## v logits
    decoder_v_logits_flops_fwd = 2 * num_layers * batch_size * num_heads * seq_len * (seq_len) * kv_channels
    ## attn out
    decoder_attn_out_flops_fwd = 2 * num_layers * batch_size * num_heads * seq_len * (kv_channels) * hidden_size
    # FF
    ## 1st layer
    decoder_ffn_1_flops_fwd = 2 * num_layers * batch_size * seq_len * (hidden_size) * ffn_hidden_size
    if glu_activation:
        # 3 matmuls instead of 2 in FFN
        # ref. https://arxiv.org/pdf/2002.05202.pdf
        # Used for example in T5 v1.1
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
