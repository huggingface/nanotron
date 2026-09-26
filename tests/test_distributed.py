import numpy as np
import pytest
import torch.distributed as dist
from helpers.utils import (
    available_gpus,
    get_all_3d_configurations,
    init_distributed,
    rerun_if_address_is_in_use,
)
import nanotron.distributed as nanotron_dist
from nanotron.parallel import ParallelContext
from torch.distributed import ProcessGroup



@pytest.mark.parametrize(
    "ranks",
    [
        np.array([0, 1], dtype=np.int64),
        [0, 1],
    ],
)
def test_new_group_normalizes_rank_types(monkeypatch, ranks):
    captured = {}
    expected_group = object()

    def fake_new_group(*, ranks, timeout, backend, pg_options):
        captured["ranks"] = ranks
        return expected_group

    monkeypatch.setattr(nanotron_dist.dist, "new_group", fake_new_group)

    result = nanotron_dist.new_group(ranks=ranks)

    assert result is expected_group
    assert captured["ranks"] == [0, 1]
    assert all(type(rank) is int for rank in captured["ranks"])


def test_new_group_rejects_empty_ranks():
    with pytest.raises(ValueError, match="Cannot create a group with not ranks inside it"):
        nanotron_dist.new_group(ranks=[])

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
