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

from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import CheckpointFunction
from torchtyping import TensorType

from nanotron import logging
from nanotron.config import Config, Llama4Config, ParallelismArgs
from nanotron.config.models_config import RandomInit, SpectralMupInit
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
from nanotron.models.llama import (
    MLP,
    CausalSelfAttention,
    Embedding,
    GLUActivation,
    LlamaDecoderLayer,
    Loss,
    LossWithZLoss,
    get_flops,
)
from nanotron.moe.expert_context import ExpertContext
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import PipelineBlock, TensorPointer
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelLinearMode,
)
from nanotron.random import RandomStates
from nanotron.scaling.parametrization import SpectralMupParametrizator, StandardParametrizator

logger = logging.get_logger(__name__)


import torch
import torch.distributed as dist


class SequentialMLP(nn.Module):
    def __init__(
        self,
        config: Llama4Config,
    ):
        super().__init__()

        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        self.split_silu_mul = GLUActivation(config.hidden_act)

    def forward(self, hidden_states):  # [seq_length, batch_size, hidden_dim]
        merged_states = self.gate_up_proj(hidden_states)
        hidden_states = self.down_proj(self.split_silu_mul(merged_states))
        return {"hidden_states": hidden_states}


class Llama4TextMoELayer(nn.Module):
    """Mixture of experts Layer for Qwen2 models."""

    def __init__(
        self,
        config: Llama4Config,
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

        # Router for selecting experts
        self.router = TensorParallelColumnLinear(
            self.hidden_size,
            self.num_experts,
            pg=tp_pg,
            mode=TensorParallelLinearMode.ALL_REDUCE,
            bias=False,
            async_communication=False,
        )
        # self.router = nn.Linear(
        #     self.hidden_size,
        #     self.num_experts,
        #     bias=False,
        # )

        # Enable shared experts if configured
        # self.enable_shared_expert = getattr(config.moe_config, "enable_shared_expert", False)
        # TODO: configurable
        self.shared_expert = MLP(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
        )
        # self.shared_expert_gate = TensorParallelColumnLinear(
        #     self.hidden_size,
        #     1,
        #     pg=tp_pg,
        #     mode=tp_mode,
        #     bias=False,
        #     async_communication=tp_linear_async_communication,
        # )

        # Create the expert MLPs
        self.experts = nn.ModuleList(
            [
                MLP(
                    config=config,
                    parallel_config=parallel_config,
                    tp_pg=tp_pg,
                )
                for _ in range(self.num_local_experts)
            ]
        )

        # Whether to recompute MoE layer during backward pass for memory efficiency
        self.recompute_layer = getattr(parallel_config, "recompute_layer", False)
        self.tp_pg = tp_pg

        # Token dispatcher type - determines communication pattern
        # self.token_dispatcher_type = getattr(config.moe_config, "token_dispatcher_type", "alltoall")
        # For more sophisticated implementations, we would add token dispatcher logic here

    def _compute_router_probabilities(self, hidden_states):
        """Compute routing probabilities for each token to each expert."""
        from einops import rearrange

        from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import differentiable_all_gather

        seq_len = hidden_states.shape[0]
        bs = hidden_states.shape[1]

        # assert_tensor_equal_across_processes(hidden_states, self.tp_pg)

        parallel_router_logits = self.router(hidden_states)  # [batch_size*seq_length, num_experts]
        # # NOTE: add noise to the router logits
        # parallel_router_logits += torch.randn_like(parallel_router_logits) * 0.01
        # TODO: check the tp_mode of the router

        # NOTE: router weight shouldn't be the same across tp ranks
        # assert_tensor_equal_across_processes(self.router.weight.data, self.tp_pg)

        # NOTE: it shouldn't be equal across processes
        # assert_tensor_equal_across_processes(parallel_router_logits, self.tp_pg)
        router_logits = differentiable_all_gather(parallel_router_logits, dim=-1, group=self.tp_pg)

        assert router_logits.shape == (seq_len, bs, self.num_local_experts)
        # Get the top-k experts per token
        rotuing_router_logits, routing_indices = torch.topk(router_logits, k=self.num_experts_per_token, dim=-1)
        rotuing_router_logits = rearrange(rotuing_router_logits, "seq_len bs 1 -> seq_len bs", seq_len=seq_len)
        routing_indices = rearrange(routing_indices, "seq_len bs 1 -> seq_len bs", seq_len=seq_len)

        routing_scores = F.sigmoid(rotuing_router_logits)

        return routing_scores, routing_indices

    def _dispatch_tokens(
        self, hidden_states, routing_weights: TensorType["seq_len", "bs"], routing_indices: TensorType["seq_len", "bs"]
    ):
        """
        Dispatches tokens to their selected experts.
        In a full implementation, this would handle the actual token routing logic
        including communication between devices.

        routing_weights: where a value specifies the weight of the expert
        routing_indices: where a value specifies the index of the expert
        """
        from einops import einsum, rearrange

        dispatched_inputs = []
        expert_counts = []

        total_tokens = routing_weights.numel()
        router_loss = torch.tensor(0.0, device=routing_weights.device, dtype=routing_weights.dtype)
        frac_of_tokens_routed_to_expert_list = []

        routing_weights = rearrange(routing_weights, "seq_len bs -> (seq_len bs)")
        for expert_idx in range(self.num_local_experts):
            # Find tokens that have this expert in their top-k
            # expert_mask.shape = [seq_len, bs]
            # true for all tokens that have this expert in their top-k
            expert_mask = routing_indices == expert_idx
            tokens_for_expert = hidden_states[expert_mask]

            # Get the routing weights for this expert
            idx_of_tokens_for_expert = torch.nonzero(expert_mask.view(-1)).squeeze()
            routing_weights_for_expert = routing_weights[idx_of_tokens_for_expert]

            if routing_weights_for_expert.numel() == 1:
                routing_weights_for_expert = routing_weights_for_expert.unsqueeze(0)

            # NOTE: no scaling
            scaled_inputs = einsum(
                tokens_for_expert, routing_weights_for_expert, "n_tokens d_model, n_tokens -> n_tokens d_model"
            )
            dispatched_inputs.append(scaled_inputs)
            expert_counts.append(len(tokens_for_expert))

            frac_of_tokens_routed_to_expert = idx_of_tokens_for_expert.numel() / total_tokens
            frac_of_router_prob_routed_to_expert = routing_weights_for_expert.sum() / total_tokens
            expert_router_loss = frac_of_tokens_routed_to_expert * frac_of_router_prob_routed_to_expert
            router_loss += expert_router_loss
            frac_of_tokens_routed_to_expert_list.append(frac_of_tokens_routed_to_expert)

            # log_rank(f"Expert {expert_idx} has {len(tokens_for_expert)} tokens", logger=logger, level=logging.INFO, rank=0)

        router_aux_loss_coef = 0.001
        router_loss = router_loss * router_aux_loss_coef * self.num_experts
        ExpertContext.get_instance().push_aux_loss(router_loss)
        return dispatched_inputs, expert_counts

    def _combine_expert_outputs(self, expert_outputs, routing_indices, original_shape):
        """
        Combines outputs from different experts back to the original tensor layout.
        """
        # Initialize output tensor with zeros
        combined_output = torch.zeros(original_shape, device=expert_outputs[0].device, dtype=expert_outputs[0].dtype)

        for expert_idx, expert_output in enumerate(expert_outputs):
            if expert_output.shape[0] == 0:  # Skip if no tokens were routed to this expert
                continue

            # Find positions where this expert was in the top-k
            expert_mask = routing_indices == expert_idx
            combined_output[expert_mask] += expert_output

        return combined_output

    def _core_forward(self, hidden_states):
        """Core forward logic for MoE layer."""
        routing_weights, routing_indices = self._compute_router_probabilities(hidden_states)

        # Dispatch tokens to experts
        dispatched_inputs, expert_counts = self._dispatch_tokens(hidden_states, routing_weights, routing_indices)

        expert_outputs = []
        for expert_idx, (inputs, count) in enumerate(zip(dispatched_inputs, expert_counts)):
            if count == 0:  # Skip computation if no tokens assigned
                expert_outputs.append(torch.tensor([], device=hidden_states.device, dtype=hidden_states.dtype))
                continue

            # Forward through the expert
            output = self.experts[expert_idx](hidden_states=inputs)["hidden_states"]
            expert_outputs.append(output)

        output = self._combine_expert_outputs(expert_outputs, routing_indices, hidden_states.shape)

        # TODO: configurable
        # Add shared expert contribution if enabled
        shared_expert_output = self.shared_expert(hidden_states=hidden_states)["hidden_states"]
        output = output + shared_expert_output

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


class Llama4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Llama4Config,
        parallel_config: Optional[ParallelismArgs],
        parallel_context: ParallelContext,
        layer_idx: int,
    ):

        super().__init__()
        tp_pg = parallel_context.tp_pg
        self.parallel_context = parallel_context
        # ep_pg = parallel_context.ep_pg

        self.input_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # NOTE: we don't ALL_REDUCE for an attention layer that is before a MoE layer
        # so we don't have to do all-gather on the output of the attention layer
        # parallel_config_for_attn = deepcopy(parallel_config)
        # parallel_config_for_attn.tp_mode = TensorParallelLinearMode.ALL_REDUCE
        # parallel_config_for_attn.tp_linear_async_communication = False
        self.attn = CausalSelfAttention(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
            layer_idx=layer_idx,
        )

        self.post_attention_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Llama4TextMoELayer(config=config, parallel_config=parallel_config, tp_pg=tp_pg)

        self.recompute_layer = parallel_config.recompute_layer

    def _core_forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
    ) -> List[Union[torch.Tensor, TensorPointer]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        output = self.attn(hidden_states=hidden_states, sequence_mask=sequence_mask)
        hidden_states = output["hidden_states"]

        # dist.barrier()
        # assert_tensor_equal_across_processes(hidden_states, self.parallel_context.tp_pg)

        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        dist.barrier()
        # assert_tensor_equal_across_processes(hidden_states, self.parallel_context.tp_pg)
        # NOTE: we already the compute output of attention, so we can just pass it to the MLP
        hidden_states = self.mlp(hidden_states=hidden_states)["hidden_states"]
        hidden_states = hidden_states + residual

        return hidden_states, output["sequence_mask"]

    def _checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        return CheckpointFunction.apply(self._core_forward, True, hidden_states, sequence_mask)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:

        if self.recompute_layer and not isinstance(hidden_states, TensorPointer):
            hidden_states, sequence_mask = self._checkpointed_forward(hidden_states, sequence_mask)
        else:
            hidden_states, sequence_mask = self._core_forward(hidden_states, sequence_mask)

        return {
            "hidden_states": hidden_states,
            "sequence_mask": sequence_mask,
        }


class Llama4Model(nn.Module):
    """
    Llama4 causal text model
    Build pipeline graph
    """

    def __init__(
        self,
        config: Llama4Config,
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
        log_rank(f"Initialize RoPE Theta = {config.rope_theta}", logger=logger, level=logging.INFO, rank=0)
        if config.rope_interleaved:
            log_rank(
                "The RoPE interleaved version differs from the Transformers implementation. It's better to set rope_interleaved=False if you need to convert the weights to Transformers",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
        self.decoder = nn.ModuleList(
            [
                PipelineBlock(
                    p2p=self.p2p,
                    module_builder=Llama4DecoderLayer,
                    module_kwargs={
                        "config": config,
                        "parallel_config": parallel_config,
                        "parallel_context": parallel_context,
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
                "tp_recompute_allgather": parallel_config.tp_recompute_allgather,
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
        model_config = self.config

        try:
            num_key_values_heads = model_config.num_key_value_heads
        except AttributeError:
            num_key_values_heads = model_config.num_attention_heads

        model_flops, hardware_flops = get_flops(
            num_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            num_heads=model_config.num_attention_heads,
            num_key_value_heads=num_key_values_heads,
            vocab_size=model_config.vocab_size,
            ffn_hidden_size=model_config.intermediate_size,
            seq_len=sequence_length,
            batch_size=global_batch_size,
        )

        model_flops_per_s = model_flops / (iteration_time_in_sec * world_size * 1e12)
        hardware_flops_per_s = hardware_flops / (iteration_time_in_sec * world_size * 1e12)
        return model_flops_per_s, hardware_flops_per_s


class Llama4ForTraining(NanotronModel):
    def __init__(
        self,
        config: Llama4Config,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        super().__init__()
        self.model = Llama4Model(config=config, parallel_context=parallel_context, parallel_config=parallel_config)

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
        )
        if self.config.z_loss_enabled:
            return {"loss": loss["loss"], "z_loss": loss["z_loss"]}
        else:
            return {"loss": loss["loss"]}

    @torch.no_grad()
    def init_model_randomly(self, config: Config):
        """Initialize model parameters randomly.
        Note:
            Layernorm weight all 0 or 1 depending on `apply_layernorm_1p`
        """
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
            return ["model.token_position_embeddings.pp_block.token_embedding.weight", "model.lm_head.pp_block.weight"]
        else:
            return []

    def get_block_compute_costs(self):
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        return self.model.get_block_compute_costs()

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """Get flops per second for a given model"""
        return self.model.get_flops_per_sec(iteration_time_in_sec, sequence_length, global_batch_size)
