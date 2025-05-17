from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import CheckpointFunction

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import ParallelismArgs
from nanotron.config.models_config import Qwen2Config
from nanotron.models.base import ignore_init_on_device_and_dtype
from nanotron.nn.activations import ACT2FN
from nanotron.nn.moe import GroupedMLP
from nanotron.nn.moe_utils import scatter_to_tensor_group
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.sharded_parameters import SplitConfig, mark_all_parameters_in_module_as_sharded
from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import differentiable_all_gather
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from .moe_utils import get_capacity
from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import all_to_all, differentiable_reduce_scatter_sum

logger = logging.get_logger(__name__)
from .moe_utils import permute, topk_softmax_with_capacity, unpermute

try:
    import grouped_gemm.ops as ops
except ImportError:
    raise RuntimeError(
        "Grouped GEMM is not available. Please run `pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@main` (takes less than 5 minutes)"
    )

try:
    import transformer_engine as te
except ImportError:
    raise RuntimeError(
        "Transformer Engine is not available. Please run `pip install --no-build-isolation transformer_engine[pytorch]`"
    )


@dataclass
class MoELogging:
    """
    num_local_tokens: List[torch.Tensor]: The number of tokens per local expert per layer
    """

    num_local_tokens: List[torch.Tensor]


def fake_gather_to_tensor_group(tensor, dim, group):
    """Fake gather that returns random data with correct shape for performance testing"""
    # Calculate expected size after all_gather
    gather_size = list(tensor.size())
    gather_size[dim] *= group.size()

    # Create uninitialized tensor with same device/dtype as input
    fake_tensor = torch.empty(gather_size, dtype=tensor.dtype, device=tensor.device)

    # Fill with random values (mimic real data but no communication)
    # fake_tensor.normal_()
    return fake_tensor


class Qwen2MoEMLPLayer(nn.Module):
    """Mixture of experts Layer for Qwen2 models."""

    def __init__(
        self,
        config: Qwen2Config,
        parallel_config: Optional[ParallelismArgs],
        parallel_context: ParallelContext,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        moe_config = config.moe_config
        self.hidden_size = moe_config.moe_hidden_size
        self.intermediate_size = moe_config.moe_intermediate_size

        # MoE specific configurations
        num_local_experts = config.moe_config.num_experts // parallel_config.expert_parallel_size  # Experts per device
        self.num_experts_per_token = config.moe_config.top_k  # Number of experts used per token (top-k)
        self.expert_parallel_size = parallel_config.expert_parallel_size
        self.num_local_experts = num_local_experts  # Experts per device

        # Get TP mode configuration

        self.config = config
        self.expert_parallel_size = parallel_config.expert_parallel_size
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        assert self.config.moe_config.num_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.moe_config.num_experts // self.expert_parallel_size
        local_expert_indices_offset = dist.get_rank(parallel_context.ep_pg) * self.num_local_experts

        self.use_shared_expert = self.config.moe_config.enable_shared_expert
        # self.shared_expert_overlap = self.config.moe_config.shared_expert_overlap

        self.local_expert_indices = [local_expert_indices_offset + i for i in range(self.num_local_experts)]
        assert all((x < self.config.moe_config.num_experts for x in self.local_expert_indices))
        self.router = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher = None
        self.layer_number = layer_idx

        # Router for selecting experts
        # self.router = Router(config, parallel_config, layer_idx)
        self.router = TopKRouter(config=self.config, parallel_context=parallel_context)

        # self.token_dispatcher = AllToAllDispatcher(num_local_experts, num_experts, parallel_context.ep_pg)
        if self.config.moe_config.token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                num_local_experts=self.num_local_experts,
                local_expert_indices=self.local_expert_indices,
                config=self.config,
                parallel_context=parallel_context,
            )
        elif self.config.moe_config.token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                num_local_experts=self.num_local_experts,
                local_expert_indices=self.local_expert_indices,
                config=self.config,
                parallel_context=parallel_context,
            )
        else: # TODO: alltoall
            raise ValueError(f"Unsupported token dispatcher type: {self.config.moe_config.token_dispatcher_type}")

        # Enable shared experts if configured
        self.enable_shared_expert = config.moe_config.enable_shared_expert
        if self.enable_shared_expert:  # TODO: check shared
            from nanotron.models.qwen import Qwen2MLP

            self.shared_expert = Qwen2MLP(
                config=config,
                parallel_config=parallel_config,
                tp_pg=parallel_context.tp_pg,
                hidden_size=moe_config.shared_expert_hidden_size,
                intermediate_size=moe_config.shared_expert_intermediate_size,
            )
            # TODO: duplicte the shared expert gate
            self.shared_expert_gate = nn.Linear(
                self.hidden_size,
                1,
                bias=False,
            )  # TODO: ensure shared_expert_gate is tied across TP

        if self.config.moe_config.grouped_gemm_imple == "transformer_engine":
            self.experts = TEGroupedMLP(self.num_local_experts, config, parallel_config, parallel_context)
        elif self.config.moe_config.grouped_gemm_imple == "megablock_grouped_gemm":
            self.experts = GroupedMLP(config, parallel_config, ep_pg=parallel_context.ep_pg)

        # Whether to recompute MoE layer during backward pass for memory efficiency
        self.recompute_layer = parallel_config.recompute_layer
        self.ep_pg = parallel_context.ep_pg
        self.layer_idx = layer_idx

    def _compute_expert_outputs(self, hidden_states, routing_weights, routing_indices):
        (
            dispatched_inputs,
            inverse_permute_mapping,
            expert_sort_indices,
            num_local_tokens_per_expert,
        ) = self.token_dispatcher.permute(hidden_states, routing_indices)

        expert_outputs = self.experts(dispatched_inputs, num_local_tokens_per_expert)
        output = self.token_dispatcher.unpermute(
            expert_outputs["hidden_states"], inverse_permute_mapping, routing_weights, expert_sort_indices
        )
        return output, num_local_tokens_per_expert

    def _core_forward(self, hidden_states, moe_logging: Optional[MoELogging]):
        """Core forward logic for MoE layer."""
        # Get top-k routing weights and indices
        # routing_weights, routing_indices = self.router(hidden_states)  # [num_tokens, num_experts_per_token]
        probs, indices = self.router(hidden_states)

        # output, num_local_tokens_per_expert = self._compute_expert_outputs(
        #     hidden_states, routing_weights, routing_indices
        # )
        (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(hidden_states, probs, indices)

        if self.config.moe_config.grouped_gemm_imple == "transformer_engine":
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
        elif self.config.moe_config.grouped_gemm_imple == "megablock_grouped_gemm":
            expert_output = self.experts(dispatched_input, tokens_per_expert)
            expert_output = expert_output["hidden_states"]
            mlp_bias = None

        output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)

        if self.enable_shared_expert:
            shared_expert_output = self.shared_expert(hidden_states=hidden_states)["hidden_states"]
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
            output = output + shared_gate * shared_expert_output

        if moe_logging is not None:
            moe_logging[self.layer_idx, :] = tokens_per_expert

        return {"hidden_states": output}

    def _checkpointed_forward(self, hidden_states):
        """Apply gradient checkpointing to save memory during training."""
        return CheckpointFunction.apply(self._core_forward, True, hidden_states)

    def forward(self, hidden_states, moe_logging: Optional[MoELogging] = None):
        """Forward pass for the MoE layer."""
        if self.recompute_layer and self.training:
            outputs = self._checkpointed_forward(hidden_states, moe_logging)
        else:
            outputs = self._core_forward(hidden_states, moe_logging)

        return outputs


from .moe_utils import (
    MoEAuxLossAutoScaler,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
    z_loss_func,
)


class TopKRouter(nn.Module):
    """Route each token to the top-k experts."""

    def __init__(self, config: Qwen2Config, parallel_context: ParallelContext) -> None:
        """Initialize the zero token dropping router."""
        super().__init__()
        self.config = config
        self.num_experts = self.config.moe_config.num_experts
        self.moe_aux_loss_func = None
        self.layer_number = None

        # float32 routing weights
        # NOTE: qwen keep the routing weights in float32
        # https://github.com/huggingface/transformers/blob/27a25bee4fcb865e8799ba026f1ea4455f2cca98/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L608
        with ignore_init_on_device_and_dtype():
            self.weight = nn.Parameter(
                torch.randn(self.num_experts, config.hidden_size, dtype=torch.float32, device="cuda")
            )
        assert self.weight.dtype == torch.float32

        self.topk = self.config.moe_config.top_k
        self.routing_type = self.config.moe_config.router_load_balancing_type
        self.input_jitter = None
        self.tp_size = parallel_context.tensor_parallel_size

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            torch.Tensor: The logits tensor after applying sinkhorn routing.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_config.router_aux_loss_coef == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(logits.to(dtype=torch.float32))  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = _sinkhorn_activation(logits)
            scores = torch.gather(logits, 1, indices)
        else:
            logits = _sinkhorn_activation(logits)
            scores, indices = torch.topk(logits, k=self.topk, dim=1)
        return scores, indices

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): the probabilities tensor after load balancing.
            indices (torch.Tensor): the indices tensor after top-k selection.
        """
        probs, indices, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_config.moe_router_pre_softmax,
            num_groups=self.config.moe_config.moe_router_num_groups,
            group_topk=self.config.moe_config.moe_router_group_topk,
            scaling_factor=self.config.moe_config.moe_router_topk_scaling_factor,
            score_function=self.config.moe_config.moe_router_score_function,
            expert_bias=False, # TODO: no bias
        )

        # Apply load balancing loss
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)

        if self.config.moe_config.router_aux_loss_coef > 0:
            probs = self.apply_load_balancing_loss(scores, tokens_per_expert, activation=probs)

        return probs, indices

    def apply_load_balancing_loss(
        self,
        probs: torch.Tensor,
        num_local_tokens_per_expert: torch.Tensor,
        activation: torch.Tensor,
    ):
        """Applies auxiliary loss to the MoE layer.

        Args:
            probs (torch.Tensor): The probs output by the router for each token. [num_tokens, num_experts]
            num_local_tokens_per_expert (torch.Tensor): The number of tokens per expert. [num_experts]
            activation (torch.Tensor): The activation tensor to attach the gradient function to.

        Returns:
            torch.Tensor: The activation tensor with the attached gradient function.
        """
        moe_aux_loss_coeff = self.config.moe_config.router_aux_loss_coef / self.tp_size
        aux_loss = switch_load_balancing_loss_func(probs, num_local_tokens_per_expert, self.topk, moe_aux_loss_coeff)
        # save_to_aux_losses_tracker(
        #     "load_balancing_loss",
        #     aux_loss / moe_aux_loss_coeff,
        #     self.layer_number,
        #     self.config.num_layers,
        # )
        activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_config.moe_z_loss_coeff is not None:
            moe_z_loss_coeff = self.config.moe_config.moe_z_loss_coeff / self.tp_size
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            # save_to_aux_losses_tracker(
            #     "z_loss",
            #     z_loss / self.config.moe_z_loss_coeff,
            #     self.layer_number,
            #     self.config.num_layers,
            # )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_config.input_jitter_eps is not None:
            eps = self.config.moe_config.input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == "cpu":
            # move weights to GPU
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        # Convert to specified datatype for routing computation if enabled
        router_dtype = input.dtype
        if self.config.moe_config.moe_router_dtype == "fp32":
            router_dtype = torch.float32
        elif self.config.moe_config.moe_router_dtype == "fp64":
            router_dtype = torch.float64
        logits = torch.nn.functional.linear(input.to(router_dtype), self.weight.to(router_dtype))
        return logits

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): the probabilities tensor after load balancing.
            indices (torch.Tensor): the indices tensor after top-k selection.
        """
        logits = logits.view(-1, self.config.moe_config.num_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        # if self.tp_size > 1 and self.config.moe_config.moe_token_dispatcher_type == "alltoall":
        #     # Gather the logits from the TP region
        #     raise NotImplementedError("fix TP in router")
        #     logits = gather_from_sequence_parallel_region(logits)

        if self.routing_type == "sinkhorn":
            scores, indices = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "aux_loss":
            scores, indices = self.aux_loss_load_balancing(logits)
        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            scores, indices, _ = topk_softmax_with_capacity(
                logits,
                self.topk,
                capacity_factor=self.config.moe_config.moe_expert_capacity_factor,
                pad_to_capacity=self.config.moe_config.moe_pad_expert_input_to_capacity,
                drop_policy=self.config.moe_config.moe_token_drop_policy,
            )
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

        return scores, indices

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self.hidden = input.shape[-1]

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)
        logits = logits.view(-1, self.config.moe_config.num_experts)

        scores, indices = self.routing(logits)

        return scores, indices


class TEGroupedLinear(te.pytorch.GroupedLinear):
    """
    Wrapper for the Transformer-Engine's `GroupedLinear` layer.

    Note that if Megatron's parallel_state has not been initialized
    yet, the tp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group().
    """

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        parallel_mode: Optional[str],
        config: Qwen2Config,
        parallel_config: ParallelismArgs,
        parallel_context: ParallelContext,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
    ):
        self.config = config
        self.parallel_config = parallel_config
        sequence_parallel = parallel_config.tp_mode == TensorParallelLinearMode.REDUCE_SCATTER

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = self.config.moe_config.disable_parameter_transpose_cache

        self.expert_parallel = parallel_context.expert_parallel_size > 1

        # The comms between TP and EP group is explicitly handled by MoE token dispatcher.
        # So we disable comms by making TE agnostic of model parallel.
        if is_expert:
            tp_group = parallel_context.ep_tp_pg
        else:
            tp_group = parallel_context.tp_pg
        self.explicit_expert_comm = is_expert and (tp_group.size() > 1 or self.expert_parallel)

        if self.explicit_expert_comm:
            if parallel_mode == "column":
                assert (
                    output_size % tp_group.size() == 0
                ), f"output_size {output_size} must be divisible by tp_group.size() {tp_group.size()}"
                output_size = output_size // tp_group.size()
            elif parallel_mode == "row":
                assert (
                    input_size % tp_group.size() == 0
                ), f"input_size {input_size} must be divisible by tp_group.size() {tp_group.size()}"
                input_size = input_size // tp_group.size()
            parallel_mode = None
            tp_group = None

        super().__init__(
            num_gemms=num_gemms,
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=sequence_parallel,
            fuse_wgrad_accumulation=config.moe_config.gradient_accumulation_fusion,
            tp_group=tp_group,
            tp_size=tp_group.size() if tp_group is not None else 1,
            init_method=None,  # TODO:
            bias=bias,
            return_bias=self.te_return_bias,
            parallel_mode=parallel_mode,
            ub_name=tp_comm_buffer_name,
            # get_rng_state_tracker=None,
            # params_dtype=torch.bfloat16,
            # device=device,
            # rng_tracker_name= # TODO: do i need rng tracker name?
        )

        for param in self.parameters():
            setattr(
                param, "allreduce", not (is_expert and self.expert_parallel)
            )  # TODO: does this work with TE or megatron?

        # self._te_state_cleanup()

    # def _te_state_cleanup(self):
    #     """
    #     Remove uncessary TE states.
    #     """

    #     # remove extra_state because it seems to be related to FP8 training
    #     # https://github.com/NVIDIA/Megatron-LM/blob/460e9611ba7fe07dbfb40219515c91572f622db9/megatron/core/extensions/transformer_engine.py#L1068-L1107
    #     # and it keeps it, it fails
    #     pass

    def forward(self, x, m_splits):
        """Forward."""
        _is_first_microbatch = None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        out = super().forward(x, m_splits, is_first_microbatch=_is_first_microbatch)
        self.is_first_microbatch = False

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:
            return out
        return out, None


class TEGroupedMLP(nn.Module):
    """An efficient implementation of the Experts layer using TE's GroupedLinear.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    def __init__(
        self,
        num_local_experts,
        config: Qwen2Config,
        parallel_config: ParallelismArgs,
        parallel_context: ParallelContext,
    ):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.input_size = config.hidden_size
        self.config = config

        # Double the output width with gated linear unit, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = config.moe_config.moe_intermediate_size
        if config.hidden_act == "silu":  # gated_linear_unit
            ffn_hidden_size *= 2
        else:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")

        self.linear_fc1 = TEGroupedLinear(  # TEColumnParallelGroupedLinear
            num_gemms=self.num_local_experts,
            input_size=self.input_size,
            output_size=ffn_hidden_size,
            parallel_mode="column",
            config=config,
            parallel_config=parallel_config,
            parallel_context=parallel_context,
            bias=False,  # TODO: no bias right?
            skip_bias_add=True,
            is_expert=True,
            tp_comm_buffer_name="fc1",
        )

        self.linear_fc2 = TEGroupedLinear(  # TERowParallelGroupedLinear
            num_gemms=self.num_local_experts,
            input_size=ffn_hidden_size // 2,  # TODO: hack for now. AssertionError: GEMM not possible: inp.shape[-1] = 1024, in_features = 2048
            output_size=config.hidden_size,
            parallel_mode="row",
            config=config,
            parallel_config=parallel_config,
            parallel_context=parallel_context,
            bias=False,
            skip_bias_add=True,
            is_expert=True,
            tp_comm_buffer_name="fc2",
        )

        self.act = ACT2FN[config.hidden_act] #TODO: cleanup

        mark_all_parameters_in_module_as_sharded(
            self,
            pg=parallel_context.ep_pg,
            split_config=SplitConfig(split_dim=0),
        )

    def forward(
        self, permuted_local_hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of TEGroupedMLP

        Args:
            permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the
            local experts.
            tokens_per_expert (torch.Tensor): The number of tokens per expert.

        Return:
            output (torch.Tensor): The output of the local experts.
        """
        # NOTE: transformer engine requires this to be a list
        tokens_per_expert = tokens_per_expert.tolist()

        intermediate_parallel, bias_parallel = self.linear_fc1(permuted_local_hidden_states, tokens_per_expert)

        if self.config.moe_config.bias_activation_fusion:  # TODO: need to fix this, it's true by default in megatron
            intermediate_parallel = bias_swiglu_impl(
                intermediate_parallel,
                bias_parallel,
            )
        else:
            # TODO: we assume gated
            intermediate_parallel = torch.chunk(intermediate_parallel, 2, dim=-1)
            intermediate_parallel = self.act(intermediate_parallel[0]) * intermediate_parallel[1]

        output, output_bias = self.linear_fc2(intermediate_parallel, tokens_per_expert)

        return output, output_bias
    

def bias_swiglu_impl(input, bias, fp8_input_store=False):
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        raise NotImplementedError("Bias is not supported")
    else:
        output = SwiGLUFunction.apply(input, fp8_input_store)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


class MoEAllGatherTokenDispatcher(nn.Module):
    """
    AllGather Based Token dispatcher.
    Note that this allgather spans the communication domain of TP*EP:
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: Qwen2Config,
        parallel_context: ParallelContext,
    ) -> None:
        """
        Initialize the zero token dropping router.
        """
        super().__init__()
        self.config = config
        self.shared_experts = None
        self.etp_size = parallel_context.expert_tensor_parallel_size
        self.ep_size = parallel_context.expert_parallel_size
        self.ep_pg = parallel_context.ep_pg

        self.num_local_experts = num_local_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        assert len(self.local_expert_indices) > 0, "Expected at least one local expert index"
        self.router_topk = config.moe_config.top_k
        self.add_bias = False  # TODO: assume no bias

        # self.global_local_map: 2D tensor. A mask of mapping between global and local tokens where
        # each element is True if it's between the local_expert_indices. Only useful when cross
        # device token permutation is enabled and **AllGahter** is performed.
        self.global_local_map = None

    def token_permutation(self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor):
        """Dispatch tokens to local experts. It's composed of two stages:
        (1) Gather the tokens across the expert parallel devices. After this stage,
        each device receives all of the tokens assigned to its local set of experts
        in its local HBM.
        (2) Permute the tokens locally so that they are grouped by their expert
        assignment.

        Args:
            hidden_states: 3D tensor [S/TP, B, H]. Input tokens.
            probs: 2D tensor [S/TP*B, num_experts]. Each row of probs contains
            the probility distribution across `topk` experts for one local token.
            routing_map: 2D tensor [S/TP*B, num_experts], representing token assignment to
            global experts.

        Returns:
            permuted_local_hidden_states: Permutation of tokens to local experts group.
            tokens_per_expert: the number of tokens each local expert to process.
        """
        self.hidden_shape = hidden_states.shape  # [bs*seq_len, hidden_size]
        # [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Permute the tokens across the expert parallel devices.
        # if self.etp_size > 1 or self.ep_size > 1:
        if self.etp_size > 1 or self.ep_size > 1:
            assert self.etp_size == 1, "Expert tensor parallelism currently not supported"

            ## local_indices calculation
            with torch.no_grad():
                # [num_local_tokens, num_experts] -> [num_global_tokens, num_experts], where:
                #     num_local_tokens=(S/TP)*B, num_global_tokens=S*B*EP
                routing_map = differentiable_all_gather(routing_map, group=self.ep_pg)
                # routing_map = fake_gather_to_tensor_group(routing_map, dim=0, group=self.ep_pg)

            ## local_probs calculation
            # max_prob: [S/TP*B, num_experts] -> global_probs: [S*B*EP, num_experts]
            probs = differentiable_all_gather(probs, group=self.ep_pg)
            # probs = fake_gather_to_tensor_group(probs, dim=0, group=self.ep_pg)

            # Note that this allgather spans the communication domain of TP*EP.
            #  [(S/TP)*B, H] -> [((S/TP)*B)*(TP*EP), H] = [S*B*EP, H]
            # hidden_states = gather_from_sequence_parallel_region(
            #     hidden_states, group=self.tp_ep_group, use_global_buffer=True
            # )
            # hidden_states = gather_from_sequence_parallel_region(
            #     hidden_states, group=self.ep_pg, use_global_buffer=True
            # )
            hidden_states = differentiable_all_gather(hidden_states, group=self.ep_pg)
            # hidden_states = fake_gather_to_tensor_group(hidden_states, dim=0, group=self.ep_pg)

        self.hidden_shape_before_permute = hidden_states.shape

        # The routing map and probs that for local experts.
        self.local_map = routing_map[:, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1].contiguous()
        # probs of global token assignment to local experts.
        self.local_probs = probs[:, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1].contiguous()

        tokens_per_expert = self.local_map.sum(dim=0).long()

        # NOTE: megablock grouped gemm requires tokens_per_expert to be on cpu
        if self.config.moe_config.grouped_gemm_imple == "megablock_grouped_gemm":
            tokens_per_expert = tokens_per_expert.cpu()

        (permuted_local_hidden_states, self.reversed_local_input_permutation_mapping) = permute(
            hidden_states,
            self.local_map,
            num_out_tokens=tokens_per_expert.sum(),
            fused=self.config.moe_config.permute_fusion,
        )

        return permuted_local_hidden_states, tokens_per_expert

    def token_unpermutation(self, hidden_states: torch.Tensor, bias: torch.Tensor = None):
        """
        Reverse process of `dispatch()` which permutes the output of local
        experts locallay and across expert parallel rank into the original order to
        produce the final output.

        Args:
            hidden_states: 2D tensor [num_permuted_tokens_for_local_experts, H],
            output of local experts.
            bias (optional): The bias tensor.

        Returns:
            output_total: un-permuted updated hidden states output from all local experts
            with shape of [S/TP, B, H]
        """

        # # Scale the expert output prior to reduction and subsequent to local unpermutation if k > 1.
        # # Unpermute the expert output and bias
        # permuted_probs = self.local_probs.T.contiguous().masked_select(self.local_map.T.contiguous())
        # # Here may change permuted_tokens to higher precision if probs use fp32/fp64.
        # weighted_hidden_states = hidden_states * permuted_probs.unsqueeze(-1)
        weighted_hidden_states = hidden_states
        permuted_probs = self.local_probs
        unpermuted_local_hidden = unpermute(
            weighted_hidden_states,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            probs=permuted_probs,
            routing_map=self.local_map,
            fused=self.config.moe_config.permute_fusion,
        )

        unpermuted_local_bias = None
        if self.add_bias:
            assert bias is not None
            weighted_bias = bias * permuted_probs.unsqueeze(-1)
            unpermuted_local_bias = unpermute(
                weighted_bias,
                self.reversed_local_input_permutation_mapping,
                restore_shape=self.hidden_shape_before_permute,
                routing_map=self.local_map,
                fused=self.config.moe_config.permute_fusion,
            )

        output_total = unpermuted_local_hidden
        output_bias_total = unpermuted_local_bias

        # Unpermute the tokens across ranks.
        # if self.etp_size > 1 or self.ep_size > 1:
        if self.ep_size > 1:
            if self.etp_size > 1:
                raise NotImplementedError("etp_size>1 not implemented")
                output_total = reduce_scatter_to_sequence_parallel_region(output_total, group=self.tp_ep_group)
                if self.add_bias:
                    # Unpermute the bias across expert parallel devices.
                    # bias is duplicated across tensor parallelism ranks;
                    output_bias_total = (
                        reduce_scatter_to_sequence_parallel_region(output_bias_total, group=self.tp_ep_group)
                        / self.etp_size
                    )

            # output_total = scatter_tensor_along_dim(output_total, dim=0, group=self.ep_pg)
            output_total = scatter_to_tensor_group(output_total, dim=0, group=self.ep_pg)
            # from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import differentiable_reduce_scatter_sum
            # output_total = differentiable_reduce_scatter_sum(output_total, group=self.ep_pg)

        output_total = output_total.view(self.hidden_shape)

        if self.add_bias:
            output_bias_total = output_bias_total.view(self.hidden_shape)

        # Restore the dtype of the output to the original dtype.
        output_total = output_total.to(hidden_states.dtype)
        if bias is not None:
            output_bias_total = output_bias_total.to(bias.dtype)
        return output_total, output_bias_total



class MoEAlltoAllTokenDispatcher(nn.Module):
    """
    AlltoAll-based token dispatcher.

    The workflow of AlltoAll token dispatcher is as follows:
    (1) preprocess(): calculate necessary metadata for communication and permute
    (2) token_permutation(): permute->A2A(EP)->AG(TP)->sort_chunk(if num_local_experts>1)
    (3) token_unpermutation(): sort_chunk(if num_local_experts>1)->RS(TP)->A2A(EP)->unpermute
    """

    def __init__(
        self, num_local_experts: int, local_expert_indices: List[int], config: Qwen2Config, parallel_context: ParallelContext
    ) -> None:
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
        """
        super().__init__()
        self.config = config
        self.shared_experts = None
        self.etp_size = parallel_context.expert_tensor_parallel_size
        self.ep_size = parallel_context.expert_parallel_size
        self.ep_pg = parallel_context.ep_pg
        self.ep_tp_pg = parallel_context.ep_tp_pg

        self.num_local_experts = num_local_experts
        assert config.moe_config.num_experts is not None
        self.num_experts = config.moe_config.num_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        assert (
            len(self.local_expert_indices) == self.num_local_experts
        ), "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert (
                self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1
            ), "local_expert_indices must be continous"

        # [ep_size]. Represents the number of tokens sent by the current rank to other
        # EP ranks.
        self.input_splits = None
        # [ep_size]. Represents the number of tokens received by the current rank from
        # other EP ranks.
        self.output_splits = None
        # [tp_size]. Represents the number of tokens received by the current rank from
        # other TP ranks.
        self.output_splits_tp = None
        self.permute_idx_device = torch.device("cuda") if self.config.moe_config.permute_fusion else None
        input_chunk_idxs = torch.arange(
            self.num_experts * self.etp_size, device=self.permute_idx_device
        )
        # [num_local_experts, tp_size * ep_size]. Sort the input chunks by local experts.
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        # [tp_size * ep_size, num_local_experts]. Restore the output chunks by local experts.
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel()

        # Token drop and padding.
        # Drop and pad the input to capacity.
        self.drop_and_pad = self.config.moe_config.moe_pad_expert_input_to_capacity
        if self.drop_and_pad:
            assert self.config.moe_config.moe_expert_capacity_factor is not None
            self.moe_expert_capacity_factor = self.config.moe_config.moe_expert_capacity_factor
        self.capacity = None

        # A cuda stream synchronization is needed in self.token_permutation() in some cases,
        # because there are several non-blocking DtoH data transfers called in self.preprocess().
        # The synchronization happens at different points based on MoE settings as late as possible.
        # Valid sync points are "before_permutation_1", "before_ep_alltoall", "before_finish",
        # and "no_sync".
        self.cuda_sync_point = "no_sync"

        self.shared_experts = None

    def preprocess(self, routing_map: torch.Tensor) -> torch.Tensor:
        """
        Preprocess token routing map for AlltoAll communication and token permutation.

        This method computes the number of tokens assigned to each expert based on the routing_map.
        It also initializes the necessary data structures for AlltoAll communication, such as input
        and output splits, and the mapping between global tokens and local experts.

        Args:
            routing_map (torch.Tensor): The mapping of tokens to experts, with shape
                [num_tokens, num_experts].

        Returns:
            torch.Tensor: Tensor containing the number of tokens assigned to local expert.
        """
        # [num_experts], number of tokens assigned to each expert from the current rank's input.
        num_local_tokens_per_expert = routing_map.sum(dim=0).long()

        if self.drop_and_pad:
            # Drop and pad the input to capacity.
            num_tokens = routing_map.size(0) * self.config.moe_config.top_k
            self.capacity = get_capacity(
                num_tokens=num_tokens,
                num_experts=self.num_experts,
                capacity_factor=self.moe_expert_capacity_factor,
            )
            self.num_out_tokens = self.capacity * self.num_experts
            # [num_local_experts], number of tokens processed by each expert.
            num_tokens_per_local_expert = torch.full(
                (self.num_local_experts,),
                self.capacity * self.etp_size * self.ep_size,
                dtype=torch.long,
            )
            # [tp_size * ep_size, num_local_experts]. Represents the number of tokens sent
            # to each local expert by all ranks.
            self.num_global_tokens_per_local_expert = torch.full(
                (self.num_experts * self.etp_size,),
                self.capacity,
                dtype=torch.long,
                device=self.permute_idx_device,
            )
            return num_tokens_per_local_expert
        elif self.config.moe_config.moe_expert_capacity_factor is not None:
            # Drop tokens to capacity, no padding.
            # A synchronization is needed before the first
            # permutation to get the `num_out_tokens` CPU value.
            self.num_out_tokens = num_local_tokens_per_expert.sum().to(
                torch.device("cpu"), non_blocking=True
            )
            self.cuda_sync_point = "before_permutation_1"
        else:
            # Dropless
            self.num_out_tokens = routing_map.size(0) * self.config.moe_config.top_k
            if self.ep_size > 1 or self.num_local_experts > 1:
                # Token dropless and enable ep. A synchronization is needed before expert parallel
                # AlltoAll communication to get the `input_splits` and `output_splits` CPU values.
                self.cuda_sync_point = "before_ep_alltoall"
            else:
                # Token dropless and no ep. A synchronization is needed before the returns
                # to get the `tokens_per_expert` CPU value for
                self.cuda_sync_point = "before_finish"

        if self.ep_size > 1 or self.etp_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall/allgather in variable size.
            # ===================================================
            # [ep_size]. Represents the number of tokens sent by the current rank to other
            # EP ranks.
            self.input_splits = (
                num_local_tokens_per_expert.reshape(self.ep_size, self.num_local_experts)
                .sum(axis=1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )
            # Gather the global distribution of tokens across ranks.
            # num_global_tokens_per_expert represents the number of tokens sent to each
            # expert by all ranks.
            # [tp_size, ep_size, num_experts]
            # TODO: do i need this when etp_size==1?
            num_global_tokens_per_expert = (
                differentiable_all_gather(
                    num_local_tokens_per_expert, group=self.ep_tp_pg
                )
                .reshape(self.ep_size, self.etp_size, self.num_experts)
                .transpose(0, 1)
            )
            # [tp_size, ep_size, num_experts] -> [tp_size, ep_size, num_local_experts]
            num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                :, :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
            ].contiguous()
            # [tp_size, ep_size, num_local_experts] -> [tp_size, ep_size]
            num_global_tokens_per_rank = num_global_tokens_per_local_expert.sum(axis=2)
            # [tp_size, ep_size] -> [ep_size]
            # self.output_splits represents the number of tokens received by the current rank
            # from other EP rank.
            self.output_splits = (
                num_global_tokens_per_rank[self.tp_rank]
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )
            # [tp_size, ep_size] -> [tp_size]
            # self.output_splits_tp represents the number of tokens received by the current
            # rank from other TP rank.
            self.output_splits_tp = (
                num_global_tokens_per_rank.sum(axis=1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )
            # [tp_size, ep_size, num_local_experts] -> [num_local_experts]
            num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=(0, 1)).to(
                torch.device("cpu"), non_blocking=True
            )
        else:
            num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                self.num_experts
            )
            num_tokens_per_local_expert = num_local_tokens_per_expert.to(
                torch.device("cpu"), non_blocking=True
            )

        if self.num_local_experts > 1:
            # [tp_size * ep_size, num_local_experts]. Represents the number of tokens sent
            # to each local expert by all ranks.
            self.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(
                -1, self.num_local_experts
            )
            if not self.config.moe_config.permute_fusion:
                self.num_global_tokens_per_local_expert = (
                    self.num_global_tokens_per_local_expert.to(
                        torch.device("cpu"), non_blocking=True
                    )
                )

        return num_tokens_per_local_expert

    def token_permutation(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to local experts using AlltoAll communication.

        This method performs the following steps:
        1. Preprocess the routing map to get metadata for communication and permutation.
        2. Permute input tokens for AlltoAll communication.
        3. Perform expert parallel AlltoAll communication.
        4. Sort tokens by local expert (if multiple local experts exist).

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
        """
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        self.routing_map = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
        assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self.preprocess(self.routing_map)

        if self.shared_experts is not None:
            self.shared_experts.pre_forward_comm(hidden_states.view(self.hidden_shape))

        # Permutation 1: input to AlltoAll input
        self.hidden_shape_before_permute = hidden_states.shape
        if self.cuda_sync_point == "before_permutation_1":
            torch.cuda.current_stream().synchronize()
        permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
            hidden_states,
            routing_map,
            num_out_tokens=self.num_out_tokens,
            fused=self.config.moe_config.permute_fusion,
            drop_and_pad=self.drop_and_pad,
        )

        # Perform expert parallel AlltoAll communication
        if self.cuda_sync_point == "before_ep_alltoall":
            torch.cuda.current_stream().synchronize()
        global_input_tokens = all_to_all(
            permutated_local_input_tokens, self.output_splits, self.input_splits, self.ep_pg
        )
        if self.shared_experts is not None:
            self.shared_experts.linear_fc1_forward_and_act(global_input_tokens)

        if self.etp_size > 1:
            raise NotImplementedError("Not implemented")
            if self.output_splits_tp is None:
                output_split_sizes = None
            else:
                output_split_sizes = self.output_splits_tp.tolist()
            global_input_tokens = gather_from_sequence_parallel_region(
                global_input_tokens, group=self.ep_tp_pg, output_split_sizes=output_split_sizes
            )

        # Permutation 2: Sort tokens by local expert.
        if self.num_local_experts > 1:
            if self.drop_and_pad:
                global_input_tokens = (
                    global_input_tokens.view(
                        self.etp_size * self.ep_size,
                        self.num_local_experts,
                        self.capacity,
                        *global_input_tokens.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
            else:
                global_input_tokens = sort_chunks_by_idxs(
                    global_input_tokens,
                    self.num_global_tokens_per_local_expert.ravel(),
                    self.sort_input_by_local_experts,
                    fused=self.config.moe_config.permute_fusion,
                )

        if self.cuda_sync_point == "before_finish":
            torch.cuda.current_stream().synchronize()

        return global_input_tokens, tokens_per_expert

    def token_unpermutation(
        self, hidden_states: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reverse the token permutation to restore the original order.

        This method performs the following steps:
        1. Unsort tokens by local expert (if multiple local experts exist).
        2. Perform expert parallel AlltoAll communication to restore the original order.
        3. Unpermute tokens to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.
            bias (torch.Tensor, optional): Bias tensor (not supported).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
                - None (bias is not supported).
        """
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"

        # Unpermutation 2: Unsort tokens by local expert.
        if self.num_local_experts > 1:
            if self.drop_and_pad:
                hidden_states = (
                    hidden_states.view(
                        self.num_local_experts,
                        self.etp_size * self.ep_size,
                        self.capacity,
                        *hidden_states.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
            else:
                hidden_states = sort_chunks_by_idxs(
                    hidden_states,
                    self.num_global_tokens_per_local_expert.T.ravel(),
                    self.restore_output_by_local_experts,
                    fused=self.config.moe_config.permute_fusion,
                )

        if self.etp_size > 1:
            raise NotImplementedError("Not implemented")
            if self.output_splits_tp is None:
                input_split_sizes = None
            else:
                input_split_sizes = self.output_splits_tp.tolist()
            # TODO: input_split_sizes
            hidden_states = differentiable_reduce_scatter_sum(
                hidden_states, group=self.ep_tp_pg, input_split_sizes=input_split_sizes
            )

        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        permutated_local_input_tokens = all_to_all(
            hidden_states, self.input_splits, self.output_splits, self.ep_pg
        )
        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(permutated_local_input_tokens)
            self.shared_experts.post_forward_comm()

        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            probs=self.probs,
            routing_map=self.routing_map,
            fused=self.config.moe_config.permute_fusion,
            drop_and_pad=self.drop_and_pad,
        )

        # Reshape the output tensor
        output = output.view(self.hidden_shape)

        # Add shared experts output
        if self.shared_experts is not None:
            shared_expert_output = self.shared_experts.get_output()
            output += shared_expert_output
        return output, None


def sort_chunks_by_idxs(
    input: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor, fused: bool = False
):
    """Split and sort the input tensor based on the split_sizes and sorted indices."""
    if fused:
        if not HAVE_TE or fused_sort_chunks_by_index is None:
            raise ValueError(
                "fused_sort_chunks_by_index is not available. Please install TE >= 2.1.0."
            )
        return fused_sort_chunks_by_index(input, split_sizes, sorted_idxs)

    input = torch.split(input, split_sizes.tolist(), dim=0)
    output = torch.cat([input[i] for i in sorted_idxs.tolist()], dim=0)
    return output


try:
    from transformer_engine.pytorch.permutation import (
        moe_permute,
        moe_sort_chunks_by_index,
        moe_unpermute,
    )
    fused_permute = moe_permute
    fused_unpermute = moe_unpermute
    fused_sort_chunks_by_index = moe_sort_chunks_by_index
    HAVE_TE = True
except ImportError:
    HAVE_TE = False



import torch.nn.functional as F
@torch.compile
def swiglu(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2

@torch.compile
def swiglu_back(g, y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat(
        (g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2, g * F.silu(y_1)), -1
    )

class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, fp8_input_store):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return swiglu(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = swiglu_back(grad_output, input)
        return tmp, None