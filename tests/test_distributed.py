import pytest
from helpers.utils import (
    available_gpus,
    get_all_3d_configurations,
    init_distributed,
)
from nanotron.distributed import ParallelContext
from torch.distributed import ProcessGroup


def _test_init_parallel_context(parallel_context: ParallelContext):
    # parallel_modes = [
    #     ParallelMode.GLOBAL,
    #     ParallelMode.TENSOR,
    #     ParallelMode.PIPELINE,
    #     ParallelMode.DATA,
    # ]

    # assert isinstance(parallel_context.get_global_rank(), int)

    # for parallel_mode in parallel_modes:
    #     # local_rank = parallel_context.get_local_rank(parallel_mode)

    #     assert parallel_context.is_initialized(parallel_mode) is True
    #     # assert isinstance(parallel_context.get_group(parallel_mode), ProcessGroup)

    #     # assert type(parallel_context.get_local_rank(parallel_mode)) == int
    #     # assert type(parallel_context.get_world_size(parallel_mode)) == int

    #     # process_group = parallel_context.get_group(parallel_mode)
    #     # assert isinstance(process_group, ProcessGroup)
    #     # ranks_in_group = parallel_context.get_ranks_in_group(parallel_mode)
    #     # TODO(xrsrke): do an expected list of ranks
    #     # assert ranks_in_group == dist.get_process_group_ranks(process_group)
    #     # assert len(ranks_in_group) == parallel_context.get_world_size(parallel_mode)

    #     # assert parallel_context.is_first_rank(parallel_mode) == (local_rank == 0)
    #     # assert parallel_context.is_last_rank(parallel_mode) == (
    #     #     local_rank == parallel_context.get_world_size(parallel_mode) - 1
    #     # )

    assert isinstance(parallel_context.world_pg, ProcessGroup)
    assert isinstance(parallel_context.tp_pg, ProcessGroup) if parallel_context.tensor_parallel_size > 1 else True
    assert isinstance(parallel_context.pp_pg, ProcessGroup) if parallel_context.pipeline_parallel_size > 1 else True
    assert isinstance(parallel_context.dp_pg, ProcessGroup) if parallel_context.data_parallel_size > 1 else True

    # parallel_context.destroy()

    # for parallel_mode in parallel_modes:
    #     assert parallel_context.is_initialized(parallel_mode) is False


@pytest.mark.parametrize(
    "tp,dp,pp",
    [
        pytest.param(*all_3d_configs)
        for gpus in range(1, min(available_gpus(), 4) + 1)
        for all_3d_configs in get_all_3d_configurations(gpus)
    ],
)
def test_init_parallel_context(tp: int, dp: int, pp: int):
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_init_parallel_context)()
