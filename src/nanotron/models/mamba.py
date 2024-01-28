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
""" PyTorch Mamba model.
"""
from typing import Dict, Optional, Union
import math
import torch
from flash_attn import bert_padding
from flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)
from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding
from torch import nn
from transformers.activations import ACT2FN
from functools import partial

from nanotron.config.utils_config import str_to_dtype
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import ParallelismArgs, RecomputeGranularity
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
from nanotron.generation.generate_store import AttachableStore
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import (
    PipelineBlock,
    TensorPointer,
)
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
from nanotron.config.models_config import MambaConfig

#NOTE(fmom): mamba_ssm=1.1.1
from mamba_ssm.models.mixer_seq_simple import create_block, Mamba, _init_weights

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

logger = logging.get_logger(__name__)

class Embedding(nn.Module, AttachableStore):
    def __init__(self, tp_pg: dist.ProcessGroup, config: MambaConfig, parallel_config: Optional[ParallelismArgs]):
        super().__init__()
        self.token_embedding = TensorParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
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

class MambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):    
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        self.block = create_block(
                    config.d_model,
                    ssm_cfg=config.ssm_cfg,
                    norm_epsilon=config.rms_norm_eps,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=layer_idx,
                    **factory_kwargs,
                )

    def forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
        residual: Optional[Union[torch.Tensor, TensorPointer]] = None,
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        hidden_states, residual = self.block(hidden_states)

        return {
            "hidden_states": hidden_states,
            "sequence_mask": sequence_mask,  # NOTE(fmom): dunno how to use it for now. Just keep it
            "residual": residual,
        }


class MambaModel(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
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
                    module_builder=MambaDecoderLayer,
                    module_kwargs={
                        "config": config,
                        "parallel_config": parallel_config,
                        "tp_pg": parallel_context.tp_pg,
                        "layer_idx": layer_idx,
                        "device": self.p2p.device,
                        "dtype": str_to_dtype[config.dtype],
                    },
                    module_input_keys={"hidden_states", "sequence_mask", "residual"},
                    module_output_keys={"hidden_states", "sequence_mask", "residual"},
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.final_layer_norm = PipelineBlock(
            p2p=self.p2p,
            module_builder=RMSNorm,
            module_kwargs={"hidden_size": config.d_model, "eps": config.rms_norm_eps},
            module_input_keys={"x"},
            module_output_keys={"hidden_states"},
        )

        self.lm_head = PipelineBlock(
            p2p=self.p2p,
            # Understand that this means that we return sharded logits that are going to need to be gathered
            module_builder=TensorParallelColumnLinear,
            module_kwargs={
                "in_features": config.d_model,
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

        # TODO(fmom): call tied weights here

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

        residual = None

        hidden_encoder_states = {
            "hidden_states": output["input_embeds"],
            "sequence_mask": input_mask,
            "residual": residual,
        }

        for block in self.decoder:
            hidden_encoder_states = block(**hidden_encoder_states)

        hidden_states = self.final_layer_norm(x=hidden_encoder_states["hidden_states"])["hidden_states"]

        sharded_logits = self.lm_head(x=hidden_states)["logits"]

        fp32_sharded_logits = self.cast_to_fp32(x=sharded_logits)["output"]

        return fp32_sharded_logits, hidden_states


    def get_block_compute_costs(self):
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        # model_config = self.config
        # d_ff = model_config.intermediate_size
        # d_qkv = model_config.d_model // model_config.num_attention_heads
        # block_compute_costs = {
        #     # CausalSelfAttention (qkv proj + attn out) + MLP
        #     LlamaDecoderLayer: 4 * model_config.num_attention_heads * d_qkv * model_config.d_model
        #     + 3 * d_ff * model_config.d_model,
        #     # This is the last lm_head
        #     TensorParallelColumnLinear: model_config.vocab_size * model_config.d_model,
        # }
        model_config = self.config

        block_compute_costs = {
            # CausalSelfAttention (qkv proj + attn out) + MLP
            MambaDecoderLayer: 0,
            # This is the last lm_head
            TensorParallelColumnLinear: 0,
        }
        log_rank(f"get_block_compute_costs() Not implemented yet", logger=logger, level=logging.INFO, rank=0)
        return block_compute_costs

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """Get flops per second for a given model"""
        # world_size = self.parallel_context.world_pg.size()
        # try:
        #     num_key_values_heads = self.config.num_key_value_heads
        # except AttributeError:
        #     num_key_values_heads = self.config.num_attention_heads

        # model_flops, hardware_flops = get_flops(
        #     num_layers=self.config.num_hidden_layers,
        #     hidden_size=self.config.d_model,
        #     num_heads=self.config.num_attention_heads,
        #     num_key_value_heads=num_key_values_heads,
        #     vocab_size=self.config.vocab_size,
        #     ffn_hidden_size=self.config.intermediate_size,
        #     seq_len=sequence_length,
        #     batch_size=global_batch_size,
        #     recompute_granularity=self.parallel_config.recompute_granularity,
        # )

        # model_flops_per_s = model_flops / (iteration_time_in_sec * world_size * 1e12)
        # hardware_flops_per_s = hardware_flops / (iteration_time_in_sec * world_size * 1e12)
        
        # TODO(fmom): undo hardcoding of model_flops_per_s and  hardware_flops_per_s
        model_flops_per_s = 0
        hardware_flops_per_s = 0
        log_rank(f"get_flops_per_sec() Not implemented yet", logger=logger, level=logging.INFO, rank=0)
        return model_flops_per_s, hardware_flops_per_s


torch.jit.script
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


class MambaForTraining(NanotronModel):
    def __init__(
        self,
        config: MambaConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        super().__init__()
        
        self.model = MambaModel(
            config=config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=random_states,
        )
        
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

    @torch.no_grad()
    def init_mamba_weights(self, n_layer, initializer_range, rescale_prenorm_residual, n_residuals_per_layer):
        
        model = self
        initialized_parameters = set()
        
        # Handle tensor parallelism
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        # Fix the root_model
        module_id_to_prefix[id(model)] = ""

        #TODO(fmom): port initiliaztion from mamba_ssm.mamba_simple.Mamba to here

        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                
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
                        pass
                    elif "bias" == param_name:
                        param.zero_()
                    else:
                        raise ValueError(f"Who the fuck is {param_name}?")

                    assert full_param_name not in initialized_parameters
                    initialized_parameters.add(full_param_name)
            elif isinstance(module, TensorParallelEmbedding):
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

                nn.init.normal_(module.weight, std=initializer_range)
                assert full_param_name not in initialized_parameters
                initialized_parameters.add(full_param_name)                

            if rescale_prenorm_residual:
                # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
                #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
                #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
                #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
                #
                # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        # get fullname
                        assert isinstance(p, NanotronParameter)
                        if p.is_tied:
                            tied_info = p.get_tied_info()
                            full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                                module_id_to_prefix=module_id_to_prefix
                            )
                        else:
                            full_param_name = f"{module_name}.{param_name}"
                        
                        # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                        # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                        # We need to reinit p since this code could be called multiple times
                        # Having just p *= scale would repeatedly scale it down
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                        with torch.no_grad():
                            p /= math.sqrt(n_residuals_per_layer * n_layer)
                                                
                        assert full_param_name not in initialized_parameters
                        initialized_parameters.add(full_param_name)
                                                    
        # #TODO(fmom): perform check
        # assert initialized_parameters == {
        #     param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
        #     if param.is_tied
        #     else name
        #     for name, param in model.named_parameters()
        # }, f"Somehow the initialized set of parameters don't match:\n - Expected: { {name for name, _ in model.named_parameters()} }\n - Got: {initialized_parameters}"

    # TODO(fmom): implement get_block_compute_costs
    def get_block_compute_costs(self):
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        return self.model.get_block_compute_costs()

    # TODO(fmom): implement get_flops_per_sec
    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """Get flops per second for a given model"""
        return self.model.get_flops_per_sec(iteration_time_in_sec, sequence_length, global_batch_size)
