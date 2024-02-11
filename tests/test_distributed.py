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
    ranks3d = parallel_context.get_3d_ranks(world_rank)
    assert isinstance(ranks3d, tuple) and len(ranks3d)

    assert isinstance(parallel_context.world_rank_matrix, np.ndarray)
    assert isinstance(parallel_context.world_ranks_to_pg, dict)

    parallel_context.destroy()
    assert dist.is_initialized() is False


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
