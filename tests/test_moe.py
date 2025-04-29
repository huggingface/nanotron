import os
from copy import copy
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.distributed as dist
from helpers.qwen_helper import TINY_MOE_QWEN_CONFIG
from helpers.utils import (
    init_distributed,
    rerun_if_address_is_in_use,
)
from nanotron.config.parallelism_config import ParallelismArgs
from nanotron.models.base import init_on_device_and_dtype
from nanotron.nn.moe import GroupedMLP, Qwen2MoELayer
from nanotron.parallel import ParallelContext
from nanotron.parallel.context import ParallelMode
from torch.distributed import ProcessGroup


@dataclass(frozen=True)
class ParalellismConfig:
    tp: int
    dp: int
    pp: int
    expert_parallel_size: int
    expert_tensor_parallel_size: int
    expert_data_parallel_size: int


PARALLEL_CONFIGS_TO_PARALLEL_RANKS = {
    ParalellismConfig(
        tp=1, dp=4, pp=1, expert_parallel_size=2, expert_tensor_parallel_size=1, expert_data_parallel_size=2
    ): {
        "attn_groups": {
            "tp": [[0], [1], [2], [3]],
            "cp": [[0], [1], [2], [3]],
            "pp": [[0], [1], [2], [3]],
            "dp": [[0, 1, 2, 3]],
        },
        "moe_groups": {
            "tp": [[0], [1], [2], [3]],
            "ep": [[0, 1], [2, 3]],
            "pp": [[0], [1], [2], [3]],
            "dp": [[0, 2], [1, 3]],
        },
    },
    ParalellismConfig(
        tp=1, dp=2, pp=1, expert_parallel_size=2, expert_tensor_parallel_size=1, expert_data_parallel_size=1
    ): {
        "attn_groups": {"tp": [[0], [1]], "cp": [[0], [1]], "pp": [[0], [1]], "dp": [[0, 1]]},
        "moe_groups": {"tp": [[0], [1]], "ep": [[0, 1]], "pp": [[0], [1]], "dp": [[0], [1]]},
    },
    ParalellismConfig(
        tp=2, dp=4, pp=1, expert_parallel_size=2, expert_tensor_parallel_size=2, expert_data_parallel_size=2
    ): {
        "attn_groups": {
            "tp": [[0, 1], [2, 3], [4, 5], [6, 7]],
            "cp": [[0], [1], [2], [3], [4], [5], [6], [7]],
            "pp": [[0], [1], [2], [3], [4], [5], [6], [7]],
            "dp": [[0, 2, 4, 6], [1, 3, 5, 7]],
        },
        "moe_groups": {
            "tp": [[0, 1], [2, 3], [4, 5], [6, 7]],
            "ep": [[0, 2], [1, 3], [4, 6], [5, 7]],
            "pp": [[0], [1], [2], [3], [4], [5], [6], [7]],
            "dp": [[0, 4], [1, 5], [2, 6], [3, 7]],
        },
    },
    ParalellismConfig(
        tp=2, dp=4, pp=1, expert_parallel_size=4, expert_tensor_parallel_size=2, expert_data_parallel_size=1
    ): {
        "attn_groups": {
            "tp": [[0, 1], [2, 3], [4, 5], [6, 7]],
            "cp": [[0], [1], [2], [3], [4], [5], [6], [7]],
            "pp": [[0], [1], [2], [3], [4], [5], [6], [7]],
            "dp": [[0, 2, 4, 6], [1, 3, 5, 7]],
        },
        "moe_groups": {
            "tp": [[0, 1], [2, 3], [4, 5], [6, 7]],
            "ep": [[0, 2, 4, 6], [1, 3, 5, 7]],
            "pp": [[0], [1], [2], [3], [4], [5], [6], [7]],
            "dp": [[0], [1], [2], [3], [4], [5], [6], [7]],
        },
    },
}


def test_grouped_mlp():
    parallel_config = ParallelismArgs(
        dp=1,
        pp=1,
        tp=1,
        expert_parallel_size=1,
        expert_tensor_parallel_size=1,
        expert_data_parallel_size=1,
        pp_engine="1f1b",
        tp_mode="REDUCE_SCATTER",
        tp_linear_async_communication=True,
    )
    # NOTE: num_tokens_per_experts.shape = (num_experts,)
    # it should match the number of experts in TINY_MOE_QWEN_CONFIG
    num_tokens_per_experts = torch.tensor([1, 2, 3, 4])
    NUM_TOKENS = num_tokens_per_experts.sum()
    NUM_EXPERTS = TINY_MOE_QWEN_CONFIG.moe_config.num_experts
    HIDDEN_SIZE = TINY_MOE_QWEN_CONFIG.hidden_size
    permuted_hidden_states = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    assert len(num_tokens_per_experts) == NUM_EXPERTS

    with init_on_device_and_dtype(device=torch.device("cuda"), dtype=torch.bfloat16):
        grouped_mlp = GroupedMLP(config=TINY_MOE_QWEN_CONFIG, parallel_config=parallel_config, ep_pg=None)

    output = grouped_mlp(permuted_hidden_states, num_tokens_per_experts)

    assert output["hidden_states"].shape == (NUM_TOKENS, HIDDEN_SIZE)
    assert output["hidden_states"].dtype == torch.bfloat16
    assert output["hidden_states"].device.type == "cuda"


def _test_init_moe_process_groups(parallel_context: ParallelContext):
    assert dist.is_initialized() is True
    assert isinstance(parallel_context.world_pg, ProcessGroup)
    assert isinstance(parallel_context.tp_pg, ProcessGroup) if parallel_context.tensor_parallel_size > 1 else True
    assert isinstance(parallel_context.pp_pg, ProcessGroup) if parallel_context.pipeline_parallel_size > 1 else True
    assert isinstance(parallel_context.dp_pg, ProcessGroup) if parallel_context.data_parallel_size > 1 else True

    assert isinstance(parallel_context.ep_pg, ProcessGroup) if parallel_context.expert_parallel_size > 1 else True
    assert (
        isinstance(parallel_context.ep_tp_pg, ProcessGroup)
        if parallel_context.expert_tensor_parallel_size > 1
        else True
    )
    assert (
        isinstance(parallel_context.ep_dp_pg, ProcessGroup) if parallel_context.expert_data_parallel_size > 1 else True
    )
    assert parallel_context.enabled_moe is True

    expected_parallel_ranks = PARALLEL_CONFIGS_TO_PARALLEL_RANKS[
        ParalellismConfig(
            tp=parallel_context.tensor_parallel_size,
            dp=parallel_context.data_parallel_size,
            pp=parallel_context.pipeline_parallel_size,
            expert_parallel_size=parallel_context.expert_parallel_size,
            expert_tensor_parallel_size=parallel_context.expert_tensor_parallel_size,
            expert_data_parallel_size=parallel_context.expert_data_parallel_size,
        )
    ]

    assert np.all(expected_parallel_ranks["attn_groups"]["dp"] == parallel_context._group_to_ranks[ParallelMode.DP])
    assert np.all(expected_parallel_ranks["attn_groups"]["tp"] == parallel_context._group_to_ranks[ParallelMode.TP])
    assert np.all(expected_parallel_ranks["attn_groups"]["pp"] == parallel_context._group_to_ranks[ParallelMode.PP])
    assert np.all(expected_parallel_ranks["attn_groups"]["cp"] == parallel_context._group_to_ranks[ParallelMode.CP])

    assert np.all(expected_parallel_ranks["moe_groups"]["dp"] == parallel_context._group_to_ranks[ParallelMode.EP_DP])
    assert np.all(expected_parallel_ranks["moe_groups"]["tp"] == parallel_context._group_to_ranks[ParallelMode.EP_TP])
    assert np.all(expected_parallel_ranks["moe_groups"]["pp"] == parallel_context._group_to_ranks[ParallelMode.EP_PP])
    assert np.all(expected_parallel_ranks["moe_groups"]["ep"] == parallel_context._group_to_ranks[ParallelMode.EP])


@pytest.mark.parametrize(
    "tp,dp,pp,expert_parallel_size,expert_tensor_parallel_size,expert_data_parallel_size",
    [
        (1, 4, 1, 2, 1, 2),
        (1, 2, 1, 2, 1, 1),
        (2, 4, 1, 2, 2, 2),
        (2, 4, 1, 4, 2, 1),
    ],
)
@rerun_if_address_is_in_use()
def test_init_moe_process_groups(
    tp: int,
    dp: int,
    pp: int,
    expert_parallel_size: int,
    expert_tensor_parallel_size: int,
    expert_data_parallel_size: int,
):
    enabled_moe = True
    init_distributed(
        tp=tp,
        dp=dp,
        pp=pp,
        expert_parallel_size=expert_parallel_size,
        expert_tensor_parallel_size=expert_tensor_parallel_size,
        expert_data_parallel_size=expert_data_parallel_size,
        enabled_moe=enabled_moe,
    )(_test_init_moe_process_groups)()


@rerun_if_address_is_in_use()
def test_expert_parallelism():
    DP_SIZE = 2
    EP_SIZE = 2
    BS = 1
    SEQ_LEN = 8
    HIDDEN_SIZE = TINY_MOE_QWEN_CONFIG.hidden_size
    parallel_config = ParallelismArgs(
        tp=1,
        dp=DP_SIZE,
        pp=1,
        expert_parallel_size=EP_SIZE,
        expert_tensor_parallel_size=1,
        expert_data_parallel_size=1,
        enabled_moe=True,
    )
    inputs = torch.arange(BS * SEQ_LEN, dtype=torch.bfloat16).unsqueeze(-1).expand(-1, HIDDEN_SIZE)
    # NOTE: support top-k routing
    routing_indices = torch.tensor([2, 3, 1, 3, 1, 0, 2, 3], dtype=torch.int32)

    init_distributed(
        tp=1,
        dp=DP_SIZE,
        pp=1,
        expert_parallel_size=EP_SIZE,
        expert_tensor_parallel_size=1,
        expert_data_parallel_size=1,
        enabled_moe=True,
    )(_test_expert_parallelism)(
        list_input_batches=inputs, list_routing_indices=routing_indices, parallel_config=parallel_config
    )


def _test_expert_parallelism(
    parallel_context: ParallelContext,
    list_input_batches: torch.Tensor,
    list_routing_indices: torch.Tensor,
    parallel_config: ParallelismArgs,
):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    ep_rank = dist.get_rank(parallel_context.ep_pg)
    input_batches = (
        torch.chunk(list_input_batches, chunks=parallel_context.expert_parallel_size, dim=0)[ep_rank]
        .contiguous()
        .cuda()
    )
    list_input_batches = list_input_batches.contiguous().cuda()
    (
        torch.chunk(list_routing_indices, chunks=parallel_context.expert_parallel_size, dim=0)[ep_rank]
        .contiguous()
        .cuda()
    )
    list_routing_indices = list_routing_indices.contiguous().cuda()

    ref_parallel_context = copy(parallel_context)
    ref_parallel_context.expert_parallel_size = 1
    ref_parallel_context.expert_tensor_parallel_size = 1
    ref_parallel_context.expert_data_parallel_size = 1
    ref_parallel_context.ep_pg = parallel_context.tp_pg

    ref_parallel_config = copy(parallel_config)
    ref_parallel_config.expert_parallel_size = 1
    ref_parallel_config.expert_tensor_parallel_size = 1
    ref_parallel_config.expert_data_parallel_size = 1

    with init_on_device_and_dtype(device="cuda", dtype=torch.bfloat16):
        moe_layer = Qwen2MoELayer(
            config=TINY_MOE_QWEN_CONFIG, parallel_context=parallel_context, parallel_config=parallel_config
        )
        ref_moe_layer = Qwen2MoELayer(
            config=TINY_MOE_QWEN_CONFIG, parallel_context=ref_parallel_context, parallel_config=ref_parallel_config
        )
        # NOTE: make the parameters of all ranks in the ref_moe_layer the same
        for p in ref_moe_layer.parameters():
            dist.all_reduce(p, op=dist.ReduceOp.AVG)

    # NOTE: copy the parameter from ref moe to parallelized moe
    def is_expert_param(name):
        return any(x for x in ["experts.merged_gate_up_proj", "experts.merged_down_proj"] if x in name)

    for (n, p), (ref_n, ref_p) in zip(moe_layer.named_parameters(), ref_moe_layer.named_parameters()):
        assert n == ref_n
        if is_expert_param(n):
            # NOTE: expert parallel sharding
            num_local_experts = moe_layer.num_local_experts
            start_idx = ep_rank * num_local_experts
            end_idx = start_idx + num_local_experts
            p.data.copy_(ref_p.data[start_idx:end_idx, :, :])
        else:
            p.data.copy_(ref_p.data)

    for (name, param), (ref_name, ref_param) in zip(moe_layer.named_parameters(), ref_moe_layer.named_parameters()):
        if is_expert_param(name):
            continue

        assert name == ref_name
        assert torch.allclose(param, ref_param)

    outputs = moe_layer(input_batches)
    ref_outputs = ref_moe_layer(list_input_batches)

    assert torch.allclose(
        outputs["hidden_states"],
        torch.chunk(ref_outputs["hidden_states"], chunks=parallel_context.expert_parallel_size, dim=0)[ep_rank],
    )


@rerun_if_address_is_in_use()
def test_expert_parallelism_exclude_router():
    DP_SIZE = 2
    EP_SIZE = 2
    BS = 1
    SEQ_LEN = 8
    HIDDEN_SIZE = TINY_MOE_QWEN_CONFIG.hidden_size
    parallel_config = ParallelismArgs(
        tp=1,
        dp=DP_SIZE,
        pp=1,
        expert_parallel_size=EP_SIZE,
        expert_tensor_parallel_size=1,
        expert_data_parallel_size=1,
        enabled_moe=True,
    )
    inputs = torch.arange(BS * SEQ_LEN, dtype=torch.bfloat16).unsqueeze(-1).expand(-1, HIDDEN_SIZE)
    # NOTE: support top-k routing
    routing_indices = torch.tensor([2, 3, 1, 3, 1, 0, 2, 3], dtype=torch.int32)

    init_distributed(
        tp=1,
        dp=DP_SIZE,
        pp=1,
        expert_parallel_size=EP_SIZE,
        expert_tensor_parallel_size=1,
        expert_data_parallel_size=1,
        enabled_moe=True,
    )(_test_expert_parallelism_exclude_router)(
        list_input_batches=inputs, list_routing_indices=routing_indices, parallel_config=parallel_config
    )


def _test_expert_parallelism_exclude_router(
    parallel_context: ParallelContext,
    list_input_batches: torch.Tensor,
    list_routing_indices: torch.Tensor,
    parallel_config: ParallelismArgs,
):
    import os

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    ep_rank = dist.get_rank(parallel_context.ep_pg)
    input_batches = (
        torch.chunk(list_input_batches, chunks=parallel_context.expert_parallel_size, dim=0)[ep_rank]
        .contiguous()
        .cuda()
    )
    list_input_batches = list_input_batches.contiguous().cuda()
    routing_indices = (
        torch.chunk(list_routing_indices, chunks=parallel_context.expert_parallel_size, dim=0)[ep_rank]
        .contiguous()
        .cuda()
    )
    list_routing_indices = list_routing_indices.contiguous().cuda()

    ref_parallel_context = copy(parallel_context)
    ref_parallel_context.expert_parallel_size = 1
    ref_parallel_context.expert_tensor_parallel_size = 1
    ref_parallel_context.expert_data_parallel_size = 1
    ref_parallel_context.ep_pg = parallel_context.tp_pg

    ref_parallel_config = copy(parallel_config)
    ref_parallel_config.expert_parallel_size = 1
    ref_parallel_config.expert_tensor_parallel_size = 1
    ref_parallel_config.expert_data_parallel_size = 1

    with init_on_device_and_dtype(device="cuda", dtype=torch.bfloat16):
        moe_layer = Qwen2MoELayer(
            config=TINY_MOE_QWEN_CONFIG, parallel_context=parallel_context, parallel_config=parallel_config
        )
        ref_moe_layer = Qwen2MoELayer(
            config=TINY_MOE_QWEN_CONFIG, parallel_context=ref_parallel_context, parallel_config=ref_parallel_config
        )
        # # NOTE: make the parameters of all ranks in the ref_moe_layer the same
        for p in ref_moe_layer.parameters():
            dist.all_reduce(p, op=dist.ReduceOp.AVG)

    # NOTE: copy the parameter from ref moe to parallelized moe
    def is_expert_param(name):
        return any(x for x in ["experts.merged_gate_up_proj", "experts.merged_down_proj"] if x in name)

    for (n, p), (ref_n, ref_p) in zip(moe_layer.named_parameters(), ref_moe_layer.named_parameters()):
        assert n == ref_n
        if is_expert_param(n):
            # NOTE: expert parallel sharding
            num_local_experts = moe_layer.num_local_experts
            start_idx = ep_rank * num_local_experts
            end_idx = start_idx + num_local_experts
            p.data.copy_(ref_p.data[start_idx:end_idx, :, :])
        else:
            p.data.copy_(ref_p.data)

    for (name, param), (ref_name, ref_param) in zip(moe_layer.named_parameters(), ref_moe_layer.named_parameters()):
        if is_expert_param(name):
            continue

        assert name == ref_name
        assert torch.allclose(param, ref_param)

    outputs = moe_layer._compute_expert_outputs(
        input_batches, torch.ones_like(routing_indices, dtype=torch.float32), routing_indices, {}
    )
    ref_outputs = ref_moe_layer._compute_expert_outputs(
        list_input_batches, torch.ones_like(list_routing_indices, dtype=torch.float32), list_routing_indices, {}
    )
    assert torch.allclose(
        outputs, torch.chunk(ref_outputs, chunks=parallel_context.expert_parallel_size, dim=0)[ep_rank]
    )


if __name__ == "__main__":
    # test_grouped_mlp()
    # test_init_moe_process_groups(tp=1, dp=4, pp=1, expert_parallel_size=2, expert_tensor_parallel_size=1, expert_data_parallel_size=2, enabled_moe=True)
    # test_init_moe_process_groups(tp=2, dp=2, pp=2, expert_parallel_size=1, expert_tensor_parallel_size=1, expert_data_parallel_size=1, enabled_moe=False)

    # (1, 1, 1, 2, 1, 1) the test that fails
    # test_init_moe_process_groups(tp=1, dp=1, pp=1, expert_parallel_size=2, expert_tensor_parallel_size=1, expert_data_parallel_size=1)
    test_expert_parallelism()
    # test_expert_parallelism_exclude_router()
