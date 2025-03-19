from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import CheckpointFunction
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import Config, ParallelismArgs
from nanotron.config.models_config import Qwen2Config, RandomInit, SpectralMupInit
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
from nanotron.nn.activations import ACT2FN
from nanotron.nn.layer_norm import TritonRMSNorm as RMSNorm
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

logger = logging.get_logger(__name__)


class CoreAttention(nn.Module):
    """Core attention module that can use different attention implementations"""

    def __init__(
        self,
        config: Qwen2Config,
        tp_pg: dist.ProcessGroup,
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

    def forward(
        self,
        query_states: torch.Tensor,  # [b*s, num_heads, head_dim]
        key_states: torch.Tensor,  # [b*s, num_kv_heads, head_dim]
        value_states: torch.Tensor,  # [b*s, num_kv_heads, head_dim]
        attention_mask: Optional[torch.Tensor] = None,  # [seq_length, seq_length]
        dropout: float = 0.0,
        sliding_window: Optional[int] = None,
        **kwargs,
    ):
        """Forward pass applying the chosen attention implementation"""
        # Get the appropriate attention function
        attention_func = ALL_ATTENTION_FUNCTIONS[self._attn_implementation]

        # Check if sliding window should be applied based on config
        if (
            getattr(self.config, "use_sliding_window", False)
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= getattr(self.config, "max_window_layers", 0)
        ):
            sliding_window = getattr(self.config, "sliding_window", None)

        # need to put sequence after num_heads
        seq_length = attention_mask.shape[0]
        query_states = query_states.view(-1, seq_length, self.local_num_heads, self.head_dim).transpose(
            1, 2
        )  # [b, num_heads, seq_length, head_dim]
        key_states = key_states.view(-1, seq_length, self.local_num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(-1, seq_length, self.local_num_kv_heads, self.head_dim).transpose(1, 2)

        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length, seq_length]

        # Call the attention implementation
        attn_output = attention_func(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=dropout,
            scaling=self.head_dim**-0.5,
            sliding_window=sliding_window,
            **kwargs,
        )[
            0
        ]  # [1, b*s, num_heads, head_dim] TODO: assert we always have this shape
        # attn_output = attn_output.view(-1, seq_length, self.local_num_heads, self.head_dim).transpose(1, 2) # [b, num_heads, seq_length, head_dim]
        return attn_output.view(
            -1, self.local_num_heads * self.head_dim
        )  # [b*s, num_heads, head_dim] -> [b*s, num_heads*head_dim]


class Qwen2Attention(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.tp_pg_size = tp_pg.size()

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
            bias=True,  # Qwen2 uses bias for QKV
            async_communication=tp_linear_async_communication,
            contiguous_chunks=qkv_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )
        self.o_proj = TensorParallelRowLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,  # Qwen2 doesn't use bias for output projection
            async_communication=tp_linear_async_communication,
        )
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta,
            interleaved=True,
            seq_len_scaling_factor=None,
        )
        self.attention = CoreAttention(config, tp_pg, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch_size*seq_length, hidden_size]
        position_ids: torch.Tensor,  # [batch_size, seq_length] where -1 is padding
    ):
        # [0, 1, 2, 3, 4, 0, 1, 2, -1, -1, -1] # 2 documents with 5 and 3 tokens then padding
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # 1 document with 11 tokens
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1] # 1 document with 10 tokens then padding
        # Replace -1 with 0 in position_ids to mark every padding token as a separate sequence. Ideally we want to get rid of padding tokens from qkv
        position_ids = position_ids.masked_fill(position_ids == -1, 0)
        position_ids = position_ids.view(-1)  # [batch_size*seq_length]

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.local_q_size, self.local_kv_size, self.local_kv_size], dim=-1
        )  # [batch_size, seq_length, q_size], [batch_size, seq_length, kv_size]

        rotary_pos_emb = self.rotary_emb(position_ids=position_ids)  # [b*s, dim] or [seq_length, dim]
        rotary_pos_emb = rotary_pos_emb.unsqueeze(1)  # [b*s, 1, dim] or [seq_length, 1, dim]

        q = q.view(-1, self.local_num_heads, self.head_dim)  # [b*s, num_heads, head_dim]
        k = k.view(-1, self.local_num_kv_heads, self.head_dim)  # [b*s, num_kv_heads, head_dim]
        v = v.view(-1, self.local_num_kv_heads, self.head_dim)  # [b*s, num_kv_heads, head_dim]
        q = self.rotary_emb.apply_rotary_pos_emb(q, rotary_pos_emb)  # [b*s, num_heads, head_dim]
        k = self.rotary_emb.apply_rotary_pos_emb(k, rotary_pos_emb)  # [b*s, num_kv_heads, head_dim]

        # TODO @nouamane: optimize this, and make sure it works with flashattn and flexattn
        def get_attention_mask(position_ids, seq_length):
            # position_ids is [b*s]
            attention_mask = torch.zeros(seq_length, seq_length, device=position_ids.device)
            start_indices = torch.where(position_ids == 0)[0]
            cu_seqlens = torch.cat(
                [start_indices, torch.tensor([seq_length], dtype=start_indices.dtype, device=start_indices.device)]
            )
            # make trius for each document
            for i in range(len(cu_seqlens) - 1):
                attention_mask[cu_seqlens[i] : cu_seqlens[i + 1], cu_seqlens[i] : cu_seqlens[i + 1]] = torch.tril(
                    torch.ones(cu_seqlens[i + 1] - cu_seqlens[i], cu_seqlens[i + 1] - cu_seqlens[i])
                )
            return attention_mask.to(torch.bool)  # [seq_length, seq_length]

        attn_output = self.attention(q, k, v, attention_mask=get_attention_mask(position_ids, q.shape[0]))
        output = self.o_proj(attn_output)
        return {"hidden_states": output, "position_ids": position_ids}


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


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2Attention(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
            layer_idx=layer_idx,
        )
        self.mlp = Qwen2MLP(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.recompute_layer = parallel_config.recompute_layer

    def _core_forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        position_ids: Union[torch.Tensor, TensorPointer],
    ) -> List[Union[torch.Tensor, TensorPointer]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        output = self.self_attn(hidden_states=hidden_states, position_ids=position_ids)
        hidden_states = output["hidden_states"]
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states=hidden_states)["hidden_states"]
        hidden_states = hidden_states + residual

        return hidden_states, output["position_ids"]

    def _checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return CheckpointFunction.apply(self._core_forward, True, hidden_states, position_ids)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        position_ids: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        if self.recompute_layer and not isinstance(hidden_states, TensorPointer):
            hidden_states, position_ids = self._checkpointed_forward(hidden_states, position_ids)
        else:
            hidden_states, position_ids = self._core_forward(hidden_states, position_ids)

        return {
            "hidden_states": hidden_states,
            "position_ids": position_ids,
        }


class Embedding(nn.Module):
    def __init__(self, tp_pg: dist.ProcessGroup, config: Qwen2Config, parallel_config: Optional[ParallelismArgs]):
        super().__init__()
        self.embed_tokens = TensorParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            pg=tp_pg,
            mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
        )
        self.pg = tp_pg

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor):  # [batch_size, seq_length]
        input_embeds = self.embed_tokens(input_ids)
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

        self.embed_tokens = PipelineBlock(
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

        log_rank(f"Initialize RoPE Theta = {config.rope_theta}", logger=logger, level=logging.INFO, rank=0)

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
                        "layer_idx": layer_idx,
                    },
                    module_input_keys={"hidden_states", "position_ids"},
                    module_output_keys={"hidden_states", "position_ids"},
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.final_layer_norm = PipelineBlock(
            p2p=self.p2p,
            module_builder=RMSNorm,
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
        output = self.embed_tokens(input_ids=input_ids, position_ids=position_ids)
        decoder_states = {
            "hidden_states": output["input_embeds"],
            "position_ids": output["position_ids"],
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


class Qwen2ForTraining(NanotronModel):
    def __init__(
        self,
        config: Qwen2Config,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        super().__init__()
        self.model = Qwen2Model(config=config, parallel_context=parallel_context, parallel_config=parallel_config)
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
        )["loss"]
        return {"loss": loss}

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

        parametrizator = parametrizator_cls(config=config.model)

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
            return ["model.embed_tokens.pp_block.embed_tokens.weight", "model.lm_head.pp_block.weight"]
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
