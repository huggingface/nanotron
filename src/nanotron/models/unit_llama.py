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

from typing import Dict, Optional, Union

import torch
from torch import nn

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import Config, LlamaConfig, ParallelismArgs
from nanotron.config.models_config import RandomInit, SpectralMupInit
from nanotron.generation.generate_store import AttachableStore
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
from nanotron.nn.activations import ACT2FN
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
from nanotron.scaling.parametrization import (
    SpectralMupParametrizator, StandardParametrizator,
    UnitMupParametrizator
)
from nanotron.utils import checkpoint_method
import unit_scaling as uu

logger = logging.get_logger(__name__)


import unit_scaling as uu
import unit_scaling.functional as U
from torch import Tensor
import einops


class UmupTransformerLayer(nn.Module):
    def __init__(self, width: int, layer_idx: int, intermediate_size: int, num_hidden_layers, config) -> None:
        super().__init__()
        depth = num_hidden_layers
        self.config = config

        assert config.num_attention_heads == config.num_key_value_heads
        self.head_size = config.hidden_size // config.num_attention_heads

        self.attn_norm = uu.LayerNorm(width)
        self.attn_qkv = uu.Linear(width, 3 * width)
        self.attn_out = uu.Linear(width, width)

        self.mlp_norm = uu.LayerNorm(width)
        self.mlp_up = uu.Linear(width, intermediate_size)
        self.mlp_gate = uu.Linear(width, intermediate_size)
        self.mlp_down = uu.Linear(intermediate_size, width)

        tau_rule = uu.transformer_residual_scaling_rule()
        self.attn_tau = tau_rule(2 * layer_idx, 2 * depth)
        self.mlp_tau = tau_rule(2 * layer_idx + 1, 2 * depth)

    def forward(self, input: Tensor) -> Tensor:
        residual, skip = U.residual_split(input, self.attn_tau)
        residual = self.attn_norm(residual)
        q, k, v = einops.rearrange(self.attn_qkv(residual), "b s (z h d) -> z b h s d", d=self.head_size, z=3)
        qkv = U.scaled_dot_product_attention(q, k, v, is_causal=True)
        residual = self.attn_out(einops.rearrange(qkv, "b h s d -> b s (h d)"))
        input = U.residual_add(residual, skip, self.attn_tau)

        residual, skip = U.residual_split(input, self.mlp_tau)
        residual = self.mlp_norm(residual)
        residual = self.mlp_down(U.silu_glu(self.mlp_up(residual), self.mlp_gate(residual)))
        return {"input": U.residual_add(residual, skip, self.mlp_tau)}




class LlamaModel(nn.Module):
    """Build pipeline graph"""

    def __init__(
        self,
        config: LlamaConfig,
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

        # self.token_position_embeddings = PipelineBlock(
        #     p2p=self.p2p,
        #     module_builder=Embedding,
        #     module_kwargs={
        #         "tp_pg": parallel_context.tp_pg,
        #         "config": config,
        #         "parallel_config": parallel_config,
        #     },
        #     module_input_keys={"input_ids", "input_mask"},
        #     module_output_keys={"input_embeds"},
        # )

        self.token_position_embeddings = PipelineBlock(
            p2p=self.p2p,
            module_builder=uu.Embedding,
            module_kwargs={
                "num_embeddings": config.vocab_size,
                "embedding_dim": config.hidden_size,
            },
            module_input_keys={"input"},
            module_output_keys={"input_embeds"},
        )

        # self.decoder = nn.ModuleList(
        # self.decoder = uu.DepthModuleList(
        #     [
        #         PipelineBlock(
        #             p2p=self.p2p,
        #             module_builder=LlamaDecoderLayer,
        #             module_kwargs={
        #                 "config": config,
        #                 "parallel_config": parallel_config,
        #                 "tp_pg": parallel_context.tp_pg,
        #                 "layer_idx": layer_idx,
        #             },
        #             module_input_keys={"hidden_states", "sequence_mask"},
        #             module_output_keys={"hidden_states", "sequence_mask"},
        #         )
        #         for layer_idx in range(config.num_hidden_layers)
        #     ]
        # )

        self.decoder = uu.DepthModuleList(
            [
                PipelineBlock(
                    p2p=self.p2p,
                    module_builder=UmupTransformerLayer,
                    module_kwargs={
                        "width": config.hidden_size,
                        "layer_idx": layer_idx,
                        "num_hidden_layers": config.num_hidden_layers,
                        "intermediate_size": config.intermediate_size,
                        "config": config
                    },
                    # module_input_keys={"hidden_states"},
                    module_input_keys={"input"},
                    module_output_keys={"input"},
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.final_layer_norm = PipelineBlock(
            p2p=self.p2p,
            # module_builder=TritonRMSNorm,
            # module_kwargs={"hidden_size": config.hidden_size, "eps": config.rms_norm_eps},
            # module_builder=nn.LayerNorm,
            module_builder=uu.LayerNorm,
            module_kwargs={"normalized_shape": config.hidden_size},
            module_input_keys={"input"},
            module_output_keys={"hidden_states"},
        )  # TODO

        self.lm_head = PipelineBlock(
            p2p=self.p2p,
            # Understand that this means that we return sharded logits that are going to need to be gathered
            module_builder=uu.LinearReadout,
            module_kwargs={
                "in_features": config.hidden_size,
                "out_features": config.vocab_size,
            },
            module_input_keys={"input"},
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
        return self.forward_with_hidden_states(input_ids=input_ids, input_mask=input_mask)

    def forward_with_hidden_states(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        input_mask: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
    ):
        # all tensors are optional as most ranks don't need anything from the dataloader.

        # output = self.token_position_embeddings(input_ids=input_ids, input_mask=input_mask)
        output = self.token_position_embeddings(input=input_ids)

        # hidden_encoder_states = {
        #     "hidden_states": output["input_embeds"],
        #     "sequence_mask": input_mask,
        # }

        hidden_encoder_states = {
            "input": output["input_embeds"],
            # "sequence_mask": input_mask,
        }

        for encoder_block in self.decoder:
            hidden_encoder_states = encoder_block(**hidden_encoder_states)

        hidden_states = self.final_layer_norm(input=hidden_encoder_states["input"])["hidden_states"]

        sharded_logits = self.lm_head(input=hidden_states)["logits"]

        logits = self.cast_to_fp32(x=sharded_logits)["output"]

        return logits

    def get_block_compute_costs(self):
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        model_config = self.config
        d_ff = model_config.intermediate_size
        d_qkv = model_config.hidden_size // model_config.num_attention_heads
        block_compute_costs = {
            # CausalSelfAttention (qkv proj + attn out) + MLP
            # LlamaDecoderLayer: 4 * model_config.num_attention_heads * d_qkv * model_config.hidden_size
            # + 3 * d_ff * model_config.hidden_size,

            UmupTransformerLayer: 4 * model_config.num_attention_heads * d_qkv * model_config.hidden_size
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
        )

        model_flops_per_s = model_flops / (iteration_time_in_sec * world_size * 1e12)
        hardware_flops_per_s = hardware_flops / (iteration_time_in_sec * world_size * 1e12)
        return model_flops_per_s, hardware_flops_per_s


@torch.jit.script
def masked_mean(loss, label_mask, dtype):
    # type: (Tensor, Tensor, torch.dtype) -> Tensor
    return (loss * label_mask).sum(dtype=dtype) / label_mask.sum()


# class Loss(nn.Module):
#     def __init__(self, tp_pg: dist.ProcessGroup):
#         super().__init__()
#         self.tp_pg = tp_pg

#     def forward(
#         self,
#         sharded_logits: torch.Tensor,  # [seq_length, batch_size, logits]
#         label_ids: torch.Tensor,  # [batch_size, seq_length]
#         label_mask: torch.Tensor,  # [batch_size, seq_length]
#     ) -> Dict[str, torch.Tensor]:
#         # Megatron by defaults cast everything in fp32. `--f16-lm-cross-entropy` is an option you can use to keep current precision.
#         # https://github.com/NVIDIA/Megatron-LM/blob/f267e6186eae1d6e2055b412b00e2e545a8e896a/megatron/model/gpt_model.py#L38

#         loss = sharded_cross_entropy(
#             sharded_logits, label_ids.transpose(0, 1).contiguous(), group=self.tp_pg, dtype=torch.float
#         ).transpose(0, 1)
#         # TODO @thomasw21: It's unclear what kind of normalization we want to do.
#         loss = masked_mean(loss, label_mask, dtype=torch.float)
#         # I think indexing causes a sync we don't actually want
#         # loss = loss[label_mask].sum()
#         return {"loss": loss}


class LlamaForTraining(NanotronModel):
    def __init__(
        self,
        config: LlamaConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        super().__init__()
        self.model = LlamaModel(config=config, parallel_context=parallel_context, parallel_config=parallel_config)
        # self.loss = PipelineBlock(
        #     p2p=self.model.p2p,
        #     module_builder=Loss,
        #     module_kwargs={"tp_pg": parallel_context.tp_pg},
        #     module_input_keys={
        #         "sharded_logits",
        #         "label_ids",
        #         "label_mask",
        #     },
        #     module_output_keys={"loss"},
        # )
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
        logits = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
        )
        assert 1 == 1
        # loss = self.loss(
        #     sharded_logits=sharded_logits,
        #     label_ids=label_ids,
        #     label_mask=label_mask,
        # )["loss"]
        # return {"loss": loss}

        return U.cross_entropy(
            logits[..., :-1, :].flatten(end_dim=-2), input_ids[..., 1:].flatten()
        )

    @torch.no_grad()
    def init_model_randomly(self, config: Config):
        """Initialize model parameters randomly.
        Note:
            Layernorm weight all 0 or 1 depending on `apply_layernorm_1p`
        """
        init_method = config.model.init_method
        # if isinstance(init_method, RandomInit):
        #     parametrizator_cls = StandardParametrizator
        # elif isinstance(init_method, SpectralMupInit):
        #     parametrizator_cls = SpectralMupParametrizator
        # elif isinstance(init_method, SpectralMupInit):
        #     parametrizator_cls = SpectralMupParametrizator
        # else:
        #     raise ValueError(f"Unknown init method {init_method}")
        
        parametrizator_cls = SpectralMupParametrizator
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
