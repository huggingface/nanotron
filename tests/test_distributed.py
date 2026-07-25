import numpy as np
import pytest
import torch.distributed as dist
from helpers.utils import (
    available_gpus,
    get_all_3d_configurations,
    init_distributed,
    rerun_if_address_is_in_use,
)
from nanotron.parallel import ParallelContext
from torch.distributed import ProcessGroup


def _test_init_parallel_context(parallel_context: ParallelContext):
    assert dist.is_initialized() is True
    assert isinstance(parallel_context.world_pg, ProcessGroup)
    assert isinstance(parallel_context.tp_pg, ProcessGroup) if parallel_context.tensor_parallel_size > 1 else True
    assert isinstance(parallel_context.pp_pg, ProcessGroup) if parallel_context.pipeline_parallel_size > 1 else True
    assert isinstance(parallel_context.dp_pg, ProcessGroup) if parallel_context.data_parallel_size > 1 else True

    world_rank = dist.get_rank(parallel_context.world_pg)

    assert isinstance(parallel_context.world_rank_matrix, np.ndarray)
    assert isinstance(parallel_context.world_ranks_to_pg, dict)

    local_rank = tuple(i.item() for i in np.where(parallel_context.world_rank_matrix == world_rank))
    global_rank = parallel_context.get_global_rank(*local_rank)
    assert isinstance(global_rank, np.int64), f"The type of global_rank is {type(global_rank)}"

    assert global_rank == dist.get_rank()

    parallel_context.destroy()
    assert dist.is_initialized() is False


def test_parallel_context_get_global_rank():
    parallel_context = ParallelContext.__new__(ParallelContext)
    parallel_context.world_rank_matrix = np.arange(2 * 3 * 4 * 5 * 6).reshape((2, 3, 4, 5, 6))

    assert parallel_context.get_global_rank(1, 2, 3, 4, 5) == parallel_context.world_rank_matrix[1, 2, 3, 4, 5]
    assert (
        parallel_context.get_global_rank(
            expert_parallel_rank=1,
            pipeline_parallel_rank=2,
            data_parallel_rank=3,
            context_parallel_rank=4,
            tensor_parallel_rank=5,
        )
        == parallel_context.world_rank_matrix[1, 2, 3, 4, 5]
    )
    assert (
        parallel_context.get_global_rank(
            expert_parallel_rank=1,
            pipeline_parallel_rank=2,
            data_parallel_rank=3,
            tensor_parallel_rank=5,
        )
        == parallel_context.world_rank_matrix[1, 2, 3, 0, 5]
    )

    with pytest.raises(ValueError, match="Received conflicting values"):
        parallel_context.get_global_rank(
            ep_rank=0,
            expert_parallel_rank=1,
            pipeline_parallel_rank=2,
            data_parallel_rank=3,
            tensor_parallel_rank=5,
        )

    with pytest.raises(ValueError, match="expert_parallel_rank must be specified"):
        parallel_context.get_global_rank(
            pipeline_parallel_rank=2,
            data_parallel_rank=3,
            tensor_parallel_rank=5,
        )


@pytest.mark.parametrize(
    "tp,dp,pp",
    [
        pytest.param(*all_3d_configs)
        for gpus in range(1, min(available_gpus(), 4) + 1)
        for all_3d_configs in get_all_3d_configurations(gpus)
    ],
)
@rerun_if_address_is_in_use()
def test_init_parallel_context(tp: int, dp: int, pp: int):
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_init_parallel_context)()
