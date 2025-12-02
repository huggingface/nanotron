"""ClimLlama model with hybrid positional embeddings for climate data.

This module implements the ClimLlama model which extends Qwen2 with:
- Learned absolute positional embeddings for climate-specific information (variable, resolution, leadtime)
- Continuous spatial-temporal encodings (x, y, z coordinates and time features)
- RoPE for relative position encoding (inherited from Qwen2)
"""

from typing import Dict, Optional, Union

import torch
from torch import nn

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import Config, ParallelismArgs
from nanotron.config.models_config import ClimLlamaConfig, RandomInit, SpectralMupInit
from nanotron.logging import LoggingCollectorMixin, log_rank
from nanotron.models import NanotronModel
from nanotron.models.qwen import (
    Loss,
    LossWithZLoss,
    Qwen2DecoderLayer,
    get_flops,
)
from nanotron.nn.layer_norm import LlamaRMSNorm as RMSNorm
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.nn.llama3_ring_attention import llama3_flash_attn_prepare_cu_seqlens
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import PipelineBlock, TensorPointer
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelLinearMode,
)
from nanotron.random import RandomStates
from nanotron.scaling.parametrization import SpectralMupParametrizator, StandardParametrizator

logger = logging.get_logger(__name__)


class ClimLlamaEmbedding(nn.Module):
    """Custom embedding layer for ClimLlama that combines multiple position encodings.

    Combines:
    1. Token embeddings (from vocabulary)
    2. Variable index embeddings (discrete)
    3. Resolution level embeddings (discrete)
    4. Lead time embeddings (discrete)
    5. Spatial-temporal continuous position encoding (x, y, z, time)
    """

    def __init__(
        self,
        tp_pg: dist.ProcessGroup,
        config: ClimLlamaConfig,
        parallel_config: Optional[ParallelismArgs],
    ):
        super().__init__()
        self.config = config
        self.pg = tp_pg

        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE

        # Token embeddings (standard)
        self.token_embedding = TensorParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            pg=tp_pg,
            mode=tp_mode,
        )

        # Discrete position embeddings
        if config.use_absolute_position_embeddings:
            self.var_embedding = TensorParallelEmbedding(
                num_embeddings=config.var_vocab_size,
                embedding_dim=config.hidden_size,
                pg=tp_pg,
                mode=tp_mode,
            )

            self.res_embedding = TensorParallelEmbedding(
                num_embeddings=config.res_vocab_size,
                embedding_dim=config.hidden_size,
                pg=tp_pg,
                mode=tp_mode,
            )

            self.leadtime_embedding = TensorParallelEmbedding(
                num_embeddings=config.leadtime_vocab_size,
                embedding_dim=config.hidden_size,
                pg=tp_pg,
                mode=tp_mode,
            )

        # Continuous spatial-temporal encoding
        if config.use_spatial_temporal_encoding:
            # MLP to project 7D spatial-temporal features to hidden_size
            # Input: [x, y, z, cos_hour, sin_hour, cos_day, sin_day]
            self.spatial_temporal_proj = nn.Linear(
                in_features=7,
                out_features=config.spatial_temporal_encoding_dim,
                bias=True,
            )
            self.spatial_temporal_proj2 = nn.Linear(
                in_features=config.spatial_temporal_encoding_dim,
                out_features=config.hidden_size,
                bias=True,
            )
            self.activation = nn.GELU()

    def forward(
        self,
        input_ids: torch.Tensor,  # [batch, seq_len]
        position_ids: torch.Tensor,  # [batch, seq_len]
        var_idx: Optional[torch.Tensor] = None,  # [batch, seq_len]
        res_idx: Optional[torch.Tensor] = None,  # [batch, seq_len]
        leadtime_idx: Optional[torch.Tensor] = None,  # [batch, seq_len]
        spatial_temporal_features: Optional[torch.Tensor] = None,  # [batch, seq_len, 7]
    ) -> Dict[str, torch.Tensor]:
        # Flatten input_ids for embedding lookup
        input_ids_flat = input_ids.view(-1)  # [batch*seq_len]
        token_embeds = self.token_embedding(input_ids_flat)  # [batch*seq_len, hidden]

        # Add discrete position embeddings
        if (
            self.config.use_absolute_position_embeddings
            and var_idx is not None
            and res_idx is not None
            and leadtime_idx is not None
        ):
            var_idx_flat = var_idx.view(-1)  # [batch*seq_len]
            res_idx_flat = res_idx.view(-1)  # [batch*seq_len]
            leadtime_idx_flat = leadtime_idx.view(-1)  # [batch*seq_len]

            var_embeds = self.var_embedding(var_idx_flat)  # [batch*seq_len, hidden]
            res_embeds = self.res_embedding(res_idx_flat)  # [batch*seq_len, hidden]
            leadtime_embeds = self.leadtime_embedding(leadtime_idx_flat)  # [batch*seq_len, hidden]

            token_embeds = token_embeds + var_embeds + res_embeds + leadtime_embeds

        # Add continuous spatial-temporal encoding
        if self.config.use_spatial_temporal_encoding and spatial_temporal_features is not None:
            spatial_temporal_flat = spatial_temporal_features.view(-1, 7)  # [batch*seq_len, 7]
            spatial_embeds = self.activation(self.spatial_temporal_proj(spatial_temporal_flat))
            spatial_embeds = self.spatial_temporal_proj2(spatial_embeds)  # [batch*seq_len, hidden]
            token_embeds = token_embeds + spatial_embeds

        return {"input_embeds": token_embeds, "position_ids": position_ids}


class ClimLlamaModel(nn.Module):
    """ClimLlama model with custom positional embeddings.

    Build pipeline graph for ClimLlama model that uses ClimLlamaEmbedding
    for climate-specific position information while maintaining
    RoPE in attention layers.
    """

    def __init__(
        self,
        config: ClimLlamaConfig,
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

        # Custom embedding with climate-specific position encodings
        self.token_position_embeddings = PipelineBlock(
            p2p=self.p2p,
            module_builder=ClimLlamaEmbedding,
            module_kwargs={
                "config": config,
                "parallel_config": parallel_config,
                "tp_pg": parallel_context.tp_pg,
            },
            module_input_keys={
                "input_ids",
                "position_ids",
                "var_idx",
                "res_idx",
                "leadtime_idx",
                "spatial_temporal_features",
            },
            module_output_keys={"input_embeds", "position_ids"},
        )

        # Create decoder layers (same as Qwen2)
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

        tp_recompute_allgather = parallel_config.tp_recompute_allgather if parallel_config is not None else False

        self.lm_head = PipelineBlock(
            p2p=self.p2p,
            module_builder=TensorParallelColumnLinear,
            module_kwargs={
                "in_features": config.hidden_size,
                "out_features": config.vocab_size,
                "pg": parallel_context.tp_pg,
                "bias": False,
                "mode": self.tp_mode,
                "async_communication": tp_linear_async_communication,
                "tp_recompute_allgather": tp_recompute_allgather,
            },
            module_input_keys={"x"},
            module_output_keys={"logits"},
        )

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        position_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        var_idx: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        res_idx: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        leadtime_idx: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        spatial_temporal_features: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length, 7]
    ):
        # Get embeddings with climate-specific position encodings
        output = self.token_position_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            var_idx=var_idx,
            res_idx=res_idx,
            leadtime_idx=leadtime_idx,
            spatial_temporal_features=spatial_temporal_features,
        )

        # Compute cu_seqlens for flash attention
        # Note: position_ids is torch.Tensor at this point (TensorPointer only on non-participating PP ranks)
        cu_seqlens: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None
        if isinstance(position_ids, torch.Tensor) and position_ids.numel() > 0:
            start_indices = torch.where(position_ids.view(-1) == 0)[0]
            cu_seqlens = torch.cat(
                [start_indices, torch.tensor([position_ids.numel()], dtype=torch.int32, device=start_indices.device)]
            ).to(torch.int32)

            # llama3 ring attention support
            if self.config._attn_implementation == "llama3_ring_attention":
                assert isinstance(input_ids, torch.Tensor)  # Type narrowing for pipeline parallelism
                local_sequence_length = input_ids.shape[1]
                sequence_length = position_ids.shape[1]
                assert (
                    sequence_length == local_sequence_length * self.parallel_context.cp_pg.size()
                ), f"sequence_length={sequence_length} must be equal to local_sequence_length={local_sequence_length} * cp_pg.size()={self.parallel_context.cp_pg.size()}"
                assert (
                    sequence_length % (2 * self.parallel_context.cp_pg.size()) == 0
                ), f"Sequence length {sequence_length} must be divisible by {2 * self.parallel_context.cp_pg.size()} when using llama3 ring attention"
                (
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    local_k_slice,
                ) = llama3_flash_attn_prepare_cu_seqlens(
                    cu_seqlens,
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


class ClimLlamaForTraining(NanotronModel, LoggingCollectorMixin):
    """Training wrapper for ClimLlama model."""

    def __init__(
        self,
        config: ClimLlamaConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        super().__init__()
        self.model = ClimLlamaModel(config=config, parallel_context=parallel_context, parallel_config=parallel_config)

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
        var_idx: Union[torch.Tensor, TensorPointer],
        res_idx: Union[torch.Tensor, TensorPointer],
        leadtime_idx: Union[torch.Tensor, TensorPointer],
        spatial_temporal_features: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        sharded_logits = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            var_idx=var_idx,
            res_idx=res_idx,
            leadtime_idx=leadtime_idx,
            spatial_temporal_features=spatial_temporal_features,
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
            return [
                "model.token_position_embeddings.pp_block.token_embedding.weight",
                "model.lm_head.pp_block.weight",
            ]
        else:
            return []

    def get_block_compute_costs(self):
        """Computes the compute cost of each block in the model for load balancing."""
        return self.model.get_block_compute_costs()

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """Get flops per second for a given model"""
        return self.model.get_flops_per_sec(iteration_time_in_sec, sequence_length, global_batch_size)
