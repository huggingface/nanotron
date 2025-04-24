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
from nanotron.nn.moe import GroupedMLP
from nanotron.parallel import ParallelContext
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
        pp_engine="1f1b",
        tp_mode="REDUCE_SCATTER",
        tp_linear_async_communication=True,
    )
    num_tokens_per_experts = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    NUM_TOKENS = num_tokens_per_experts.sum()
    NUM_EXPERTS = TINY_MOE_QWEN_CONFIG.moe_config.num_experts
    HIDDEN_SIZE = TINY_MOE_QWEN_CONFIG.hidden_size
    permuted_hidden_states = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    assert len(num_tokens_per_experts) == NUM_EXPERTS

    with init_on_device_and_dtype(device=torch.device("cuda"), dtype=torch.bfloat16):
        grouped_mlp = GroupedMLP(config=TINY_MOE_QWEN_CONFIG, parallel_config=parallel_config)

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

    assert np.all(expected_parallel_ranks["attn_groups"]["dp"] == parallel_context._group_to_ranks["dp_ranks"])
    assert np.all(expected_parallel_ranks["attn_groups"]["tp"] == parallel_context._group_to_ranks["tp_ranks"])
    assert np.all(expected_parallel_ranks["attn_groups"]["pp"] == parallel_context._group_to_ranks["pp_ranks"])
    assert np.all(expected_parallel_ranks["attn_groups"]["cp"] == parallel_context._group_to_ranks["cp_ranks"])

    assert np.all(expected_parallel_ranks["moe_groups"]["dp"] == parallel_context._group_to_ranks["ep_dp_ranks"])
    assert np.all(expected_parallel_ranks["moe_groups"]["tp"] == parallel_context._group_to_ranks["ep_tp_ranks"])
    assert np.all(expected_parallel_ranks["moe_groups"]["pp"] == parallel_context._group_to_ranks["ep_pp_ranks"])
    assert np.all(expected_parallel_ranks["moe_groups"]["ep"] == parallel_context._group_to_ranks["ep_ranks"])

    assert 1 == 1
    # world_rank = dist.get_rank(parallel_context.world_pg)

    # assert isinstance(parallel_context.world_rank_matrix, np.ndarray)
    # assert isinstance(parallel_context.world_ranks_to_pg, dict)

    # local_rank = tuple(i.item() for i in np.where(parallel_context.world_rank_matrix == world_rank))
    # global_rank = parallel_context.get_global_rank(*local_rank)
    # assert isinstance(global_rank, np.int64), f"The type of global_rank is {type(global_rank)}"

    # assert global_rank == dist.get_rank()

    # parallel_context.destroy()
    # assert dist.is_initialized() is False


# @pytest.mark.parametrize(
#     "tp,dp,pp",
#     [
#         pytest.param(*all_3d_configs)
#         for gpus in range(1, min(available_gpus(), 4) + 1)
#         for all_3d_configs in get_all_3d_configurations(gpus)
#     ],
# )
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


# if __name__ == "__main__":
#     # test_grouped_mlp()
#     # test_init_moe_process_groups(tp=1, dp=4, pp=1, expert_parallel_size=2, expert_tensor_parallel_size=1, expert_data_parallel_size=2, enabled_moe=True)
#     # test_init_moe_process_groups(tp=2, dp=2, pp=2, expert_parallel_size=1, expert_tensor_parallel_size=1, expert_data_parallel_size=1, enabled_moe=False)

#     # (1, 1, 1, 2, 1, 1) the test that fails
#     test_init_moe_process_groups(tp=1, dp=1, pp=1, expert_parallel_size=2, expert_tensor_parallel_size=1, expert_data_parallel_size=1)
