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
"""PyTorch Mamba model."""

import math
from functools import partial
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MambaModelConfig
from einops import rearrange, repeat
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import ParallelismArgs
from nanotron.config.utils_config import cast_str_to_torch_dtype
from nanotron.generation.generate_store import AttachableStore
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import PipelineBlock, TensorPointer
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.sharded_parameters import SplitConfig, create_sharded_parameter_from_config
from nanotron.parallel.tensor_parallel.functional import sharded_cross_entropy
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)
from nanotron.random import RandomStates
from selective_scan_interface import mamba_inner_fn, selective_scan_fn
from torch.nn import init

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

logger = logging.get_logger(__name__)


class Mamba(nn.Module, AttachableStore):
    def __init__(
        self,
        d_model: int,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,  # Fused kernel options
        layer_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        assert tp_mode == TensorParallelLinearMode.ALL_REDUCE or parallel_config.tp_linear_async_communication is False
        "Only ALL_REDUCE and tp_linear_async_communication=False are supported"

        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        # Get current tensor parallel rank
        self.tp_pg = tp_pg
        self.tp_rank = dist.get_rank(self.tp_pg)

        self.in_proj = TensorParallelColumnLinear(
            in_features=self.d_model,
            out_features=self.d_inner * 2,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=False,
            contiguous_chunks=None,
        )

        assert self.d_inner % self.tp_pg.size() == 0

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner // self.tp_pg.size(),
            out_channels=self.d_inner // self.tp_pg.size(),
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner // self.tp_pg.size(),
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.conv1d.weight = create_sharded_parameter_from_config(
            parameter=self.conv1d.weight, pg=self.tp_pg, split_config=SplitConfig(split_dim=0)
        )
        if conv_bias:
            self.conv1d.bias = create_sharded_parameter_from_config(
                parameter=self.conv1d.bias, pg=self.tp_pg, split_config=SplitConfig(split_dim=0)
            )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = TensorParallelRowLinear(
            in_features=self.d_inner,
            out_features=self.dt_rank + self.d_state * 2,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=None,
        )

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // self.tp_pg.size(), bias=True, **factory_kwargs)

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner // self.tp_pg.size(), **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        self.dt_proj.weight = create_sharded_parameter_from_config(
            parameter=self.dt_proj.weight, pg=self.tp_pg, split_config=SplitConfig(split_dim=0)
        )
        self.dt_proj.bias = create_sharded_parameter_from_config(
            parameter=self.dt_proj.bias, pg=self.tp_pg, split_config=SplitConfig(split_dim=0)
        )

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // self.tp_pg.size(),
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = create_sharded_parameter_from_config(
            parameter=A_log, pg=self.tp_pg, split_config=SplitConfig(split_dim=0)
        )

        # D "skip" parameter
        self.D = create_sharded_parameter_from_config(
            parameter=torch.ones(self.d_inner // self.tp_pg.size(), device=device),
            pg=self.tp_pg,
            split_config=SplitConfig(split_dim=0),
        )

        # self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj = TensorParallelRowLinear(
            in_features=self.d_inner,
            out_features=self.d_model,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=None,
        )

    def forward(self, hidden_states: Union[torch.Tensor, TensorPointer]):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch_size, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None

        store = self.get_local_store()
        if store is not None:
            if "key_value_memory_list" not in store:
                store["key_value_memory_list"] = []

            if "seqlen_offset" not in store:
                store["seqlen_offset"] = 0

            conv_state, ssm_state = self._get_states_from_cache(batch_size)

            if store["seqlen_offset"] > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                store["seqlen_offset"] += 1
                return out
            else:
                store["seqlen_offset"] += 1
        # We do matmul and transpose BLH -> HBL at the same time
        xz = self.in_proj(hidden_states).transpose(1, 2)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and store is None:  # Doesn't support outputting the states
            y = mamba_inner_fn(
                d_inner=self.d_inner,
                tp_pg=self.tp_pg,
                xz=xz,
                conv1d_weight=self.conv1d.weight,
                conv1d_bias=self.conv1d.bias,
                x_proj_weight=self.x_proj.weight,
                delta_proj_weight=self.dt_proj.weight,
                A=A,
                B=None,  # input-dependent B
                C=None,  # input-dependent C
                D=self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            if self.tp_pg.size() > 1:
                x, z = xz.view(batch_size, self.d_inner // 2, 2, seqlen).chunk(2, dim=2)
            else:
                x, z = xz.view(batch_size, self.d_inner, 2, seqlen).chunk(2, dim=2)

            x = x.squeeze(2)
            z = z.squeeze(2)

            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")

        out = self.out_proj(y)
        return out

    def step(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ):
        batch_size, seqlen, dim = hidden_states.shape
        dtype = hidden_states.dtype
        assert seqlen == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)

        if self.tp_pg.size() > 1:
            x, z = xz.view(batch_size, self.d_inner // 2, 2).chunk(2, dim=2)
        else:
            x, z = xz.view(batch_size, self.d_inner, 2).chunk(2, dim=2)

        x = x.squeeze(2)  # (B D)
        z = z.squeeze(2)  # (B D)
        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state,
                x,
                dt,
                A,
                B,
                C,
                self.D,
                z=z,
                dt_bias=self.dt_proj.bias,
                dt_softplus=True,
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def _get_states_from_cache(self, batch_size: int, initialize_states: bool = False):
        assert self.layer_idx is not None

        store = self.get_local_store()

        if len(store["key_value_memory_list"]) == 0:
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand // self.tp_pg.size(),
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand // self.tp_pg.size(),
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            store["key_value_memory_list"] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = store["key_value_memory_list"]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Embedding(nn.Module, AttachableStore):
    def __init__(
        self,
        tp_pg: dist.ProcessGroup,
        config: MambaModelConfig,
        parallel_config: Optional[ParallelismArgs],
    ):
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
        # input_ids = input_ids.transpose(0, 1)
        input_embeds = self.token_embedding(input_ids)
        return {"input_embeds": input_embeds}


class MambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MambaModelConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        if config.ssm_cfg is None:
            ssm_cfg = {}
        else:
            ssm_cfg = config.ssm_cfg

        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm

        self.mixer = Mamba(
            d_model=config.d_model,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs,
        )

        self.norm = partial(
            nn.LayerNorm if not config.rms_norm else RMSNorm,
            eps=config.rms_norm_eps,
            **factory_kwargs,
        )(config.d_model)

        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
        residual: Optional[Union[torch.Tensor, TensorPointer]],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        if not self.fused_add_norm:
            # self.layer_idx was assigned when calling create_block
            # residual=None happens only at the first block
            residual = hidden_states if (self.layer_idx == 0) else hidden_states + residual
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=None if (self.layer_idx == 0) else residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states)

        return {
            "hidden_states": hidden_states,
            "sequence_mask": sequence_mask,  # NOTE(fmom): dunno how to use it for now. Just keep it
            "residual": residual,
        }


class MambaModel(nn.Module):
    def __init__(
        self,
        config: MambaModelConfig,
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
                        "dtype": cast_str_to_torch_dtype(config.dtype),
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
            module_input_keys={"x", "residual"},
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
            "residual": output["input_embeds"],
        }

        for block in self.decoder:
            hidden_encoder_states = block(**hidden_encoder_states)

        hidden_states = self.final_layer_norm(
            x=hidden_encoder_states["hidden_states"],
            residual=hidden_encoder_states["residual"],
        )["hidden_states"]

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

        block_compute_costs = {
            # CausalSelfAttention (qkv proj + attn out) + MLP
            MambaDecoderLayer: 1,
            # This is the last lm_head
            TensorParallelColumnLinear: 0,
        }
        log_rank(
            "get_block_compute_costs() Not implemented yet",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        return block_compute_costs

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """
        Get flops per second for a Mamba model.
        Terms such as nonlinearities, biases, and layer normalization are omitted (https://arxiv.org/pdf/2001.08361.pdf)
        """
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
        world_size = self.parallel_context.world_pg.size()

        expand = 2 if ("expand" not in self.config.ssm_cfg) else self.config.ssm_cfg["expand"]
        d_state = 16 if ("d_state" not in self.config.ssm_cfg) else self.config.ssm_cfg["d_state"]
        dt_rank = (
            math.ceil(self.config.d_model / 16)
            if ("dt_rank" not in self.config.ssm_cfg or self.config.ssm_cfg["dt_rank"] == "auto")
            else self.config.ssm_cfg["dt_rank"]
        )

        d_inner = int(expand * self.config.d_model)

        # embeddings (do not include embeddigns as per Chinchilla)
        # embeddings = 2 * sequence_length * self.config.vocab_size * self.config.d_model

        # selective scan, see : https://github.com/state-spaces/mamba/issues/110
        scan = 9 * sequence_length * d_state * self.config.d_model

        # linear projections
        in_proj = 2 * sequence_length * self.config.d_model * d_inner * 2
        x_proj = 2 * sequence_length * d_inner * (dt_rank + d_state * 2)
        dt_proj = 2 * sequence_length * dt_rank * d_inner
        out_proj = 2 * sequence_length * d_inner * self.config.d_model

        # output projection
        projection = 2 * sequence_length * self.config.vocab_size * self.config.d_model

        forward_flops = self.config.num_hidden_layers * (in_proj + scan + x_proj + dt_proj + out_proj) + projection
        backward_flops = 2 * forward_flops
        model_flops = forward_flops + backward_flops
        model_flops_per_s = model_flops * global_batch_size / (iteration_time_in_sec * world_size * 1e12)
        # add hardware flops later
        hardware_flops_per_s = 0

        return model_flops_per_s, hardware_flops_per_s


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

        # NOTE(fmom): undo transpose for now since Mamba is not using TP
        # loss = sharded_cross_entropy(
        #     sharded_logits, label_ids.transpose(0, 1).contiguous(), group=self.tp_pg, dtype=torch.float
        # ).transpose(0, 1)

        loss = sharded_cross_entropy(sharded_logits, label_ids, group=self.tp_pg, dtype=torch.float)

        # TODO @thomasw21: It's unclear what kind of normalization we want to do.
        loss = masked_mean(loss, label_mask, dtype=torch.float)
        # I think indexing causes a sync we don't actually want
        # loss = loss[label_mask].sum()
        return {"loss": loss}


class MambaForTraining(NanotronModel):
    def __init__(
        self,
        config: MambaModelConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        super().__init__()

        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config

        self.model = MambaModel(
            config=self.config,
            parallel_context=self.parallel_context,
            parallel_config=self.parallel_config,
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
    
    def get_named_params_without_weight_decay(self):
        # get full name with "A_log", "D"
        named_param_without_weight_decay = []
        for name, _ in self.model.named_parameters():
            if "A_log" in name or "D" in name:
                named_param_without_weight_decay.append(name)
        return named_param_without_weight_decay

    @torch.no_grad()
    def init_model_randomly(self, config):
        model = self
        initialized_parameters = set()

        # Handle tensor parallelism
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        # Fix the root_model
        module_id_to_prefix[id(model)] = ""

        initializer_range = config.model.init_method.initializer_range
        n_residuals_per_layer = config.model.init_method.n_residuals_per_layer
        num_hidden_layers = config.model.model_config.num_hidden_layers
        rescale_prenorm_residual = config.model.init_method.rescale_prenorm_residual
        d_model = config.model.model_config.d_model

        if config.model.model_config.ssm_cfg is not None:
            dt_init = config.model.model_config.ssm_cfg["dt_init"]
            dt_rank = config.model.model_config.ssm_cfg["dt_rank"]
            dt_scale = config.model.model_config.ssm_cfg["dt_scale"]

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

            if isinstance(module, TensorParallelColumnLinear) or isinstance(module, TensorParallelRowLinear):
                if "weight" == param_name:
                    init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                elif "bias" == param_name:
                    raise ValueError("We don't use bias for TensorParallelColumnLinear and TensorParallelRow")
                else:
                    raise ValueError(f"Who the fuck is {param_name}?")

                if rescale_prenorm_residual and full_param_name.endswith("out_proj.weight"):
                    # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
                    #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
                    #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
                    #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
                    #
                    # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    with torch.no_grad():
                        module.weight /= math.sqrt(n_residuals_per_layer * num_hidden_layers)

            elif isinstance(module, nn.Conv1d):
                fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                if "weight" == param_name:
                    init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                elif "bias" == param_name:
                    bound = 1 / math.sqrt(fan_in) if (fan_in > 0) else 0
                    init.uniform_(module.bias, -bound, bound)
                else:
                    raise ValueError(f"Who the fuck is {param_name}?")

            elif isinstance(module, nn.Linear):
                fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)

                if "weight" == param_name:
                    init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                elif "bias" == param_name:
                    bound = 1 / math.sqrt(fan_in) if (fan_in > 0) else 0
                    init.uniform_(module.bias, -bound, bound)
                else:
                    raise ValueError(f"Who the fuck is {param_name}?")

                if config.model.model_config.ssm_cfg is not None:
                    if dt_rank == "auto":
                        dt_init_std = math.ceil(d_model / 16) ** -0.5 * dt_scale
                    else:
                        dt_init_std = dt_rank**-0.5 * dt_scale

                    if dt_init == "constant":
                        nn.init.constant_(module.weight, dt_init_std)
                    elif dt_init == "random":
                        nn.init.uniform_(module.weight, -dt_init_std, dt_init_std)
                    else:
                        raise NotImplementedError

            elif isinstance(module, TensorParallelEmbedding):
                nn.init.normal_(module.weight, std=initializer_range)

            elif isinstance(module, RMSNorm) or isinstance(module, nn.LayerNorm):
                if "weight" == param_name:
                    # TODO @thomasw21: Sometimes we actually want 0
                    module.weight.fill_(1)
                elif "bias" == param_name:
                    module.bias.zero_()
                else:
                    raise ValueError(f"Who the fuck is {param_name}?")

            elif isinstance(module, Mamba):
                pass
            else:
                raise Exception(f"Parameter {full_param_name} was not initialized")

            assert full_param_name not in initialized_parameters
            initialized_parameters.add(full_param_name)

        assert initialized_parameters == {
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            if param.is_tied
            else name
            for name, param in model.named_parameters()
        }, f"Somehow the initialized set of parameters don't match:\n - Expected: { {name for name, _ in model.named_parameters()} }\n - Got: {initialized_parameters}"

    @staticmethod
    def get_embeddings_lm_head_tied_names():
        return [
            "model.token_position_embeddings.pp_block.token_embedding.weight",
            "model.lm_head.pp_block.weight",
        ]

    # TODO(fmom): implement get_block_compute_costs
    def get_block_compute_costs(self):
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        return self.model.get_block_compute_costs()

    # TODO(fmom): implement get_flops_per_sec
    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """Get flops per second for a given model"""
        return self.model.get_flops_per_sec(iteration_time_in_sec, sequence_length, global_batch_size)
