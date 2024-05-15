from typing import Dict, Optional, Union, List

import torch
from torch import nn

import torch.distributed as dist
from nanotron.config import LlamaConfig, ParallelismArgs
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.block import PipelineBlock, TensorPointer
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelLinearMode,
)
from nanotron.models.llama import LlamaModel, Embedding, LlamaDecoderLayer, CausalSelfAttention, MLP
from nanotron.mod.mod import MixtureOfDepth, Router


# class LlamaDecoderLayer(nn.Module):
#     def __init__(
#         self,
#         config: LlamaConfig,
#         parallel_config: Optional[ParallelismArgs],
#         tp_pg: dist.ProcessGroup,
#         layer_idx: int,
#     ):
#         super().__init__()
#         self.input_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.attn = CausalSelfAttention(
#             config=config,
#             parallel_config=parallel_config,
#             tp_pg=tp_pg,
#             layer_idx=layer_idx,
#         )

#         self.post_attention_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.mlp = MLP(config=config, parallel_config=parallel_config, tp_pg=tp_pg)
#         self.router = Router(seq_len=1024, top_k=10)

#     def forward(
#         self,
#         hidden_states: Union[torch.Tensor, TensorPointer],
#         sequence_mask: Union[torch.Tensor, TensorPointer],
#     ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
#         residual = hidden_states
#         hidden_states = self.input_layernorm(hidden_states)

#         output = self.attn(hidden_states=hidden_states, sequence_mask=sequence_mask)
#         hidden_states = output["hidden_states"]
#         hidden_states = hidden_states + residual

#         residual = hidden_states
#         hidden_states = self.post_attention_layernorm(hidden_states)
#         hidden_states = self.mlp(hidden_states=hidden_states)["hidden_states"]
#         hidden_states = hidden_states + residual

#         return {
#             "hidden_states": hidden_states,
#             "sequence_mask": output["sequence_mask"],
#         }


class MoDLlamaModel(nn.Module, LlamaModel):
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
            module_builder=TritonRMSNorm,
            module_kwargs={"hidden_size": config.hidden_size, "eps": config.rms_norm_eps},
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
