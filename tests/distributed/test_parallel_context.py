import pytest
import torch.distributed as dist
from nanotron.distributed import ParallelContext, ParallelMode
from nanotron.testing.utils import skip_if_no_cuda, spawn
from torch.distributed import ProcessGroup


def init_parallel_context(
    rank, world_size, seed, backend, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
):
    parallel_modes = [
        ParallelMode.GLOBAL,
        ParallelMode.TENSOR,
        ParallelMode.PIPELINE,
        ParallelMode.DATA,
    ]

    parallel_context = ParallelContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        # TODO(xrsrke): get host from env
        host="localhost",
        port=port,
        seed=seed,
        backend=backend,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )

    assert parallel_context.tensor_parallel_size == tensor_parallel_size
    assert parallel_context.pipeline_parallel_size == pipeline_parallel_size
    assert parallel_context.data_parallel_size == data_parallel_size

    assert parallel_context.get_global_rank() == rank

    for parallel_mode in parallel_modes:
        local_rank = parallel_context.get_local_rank(parallel_mode)

        assert parallel_context.is_initialized(parallel_mode) is True
        assert isinstance(parallel_context.get_group(parallel_mode), ProcessGroup)

        assert type(parallel_context.get_local_rank(parallel_mode)) == int
        assert type(parallel_context.get_world_size(parallel_mode)) == int

        process_group = parallel_context.get_group(parallel_mode)
        ranks_in_group = parallel_context.get_ranks_in_group(parallel_mode)
        # TODO(xrsrke): do an expected list of ranks
        assert ranks_in_group == dist.get_process_group_ranks(process_group)
        assert len(ranks_in_group) == parallel_context.get_world_size(parallel_mode)

        assert parallel_context.is_first_rank(parallel_mode) == (local_rank == 0)
        assert parallel_context.is_last_rank(parallel_mode) == (
            local_rank == parallel_context.get_world_size(parallel_mode) - 1
        )

    parallel_context.destroy()

    for parallel_mode in parallel_modes:
        assert parallel_context.is_initialized(parallel_mode) is False


@skip_if_no_cuda
@pytest.mark.parametrize("tensor_parallel_size", (2,))
@pytest.mark.parametrize("pipeline_parallel_size", (2,))
@pytest.mark.parametrize("data_parallel_size", (2,))
def test_init_parallel_context(tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    SEED = 69
    BACKEND = "nccl"
    WORLD_SIZE = tensor_parallel_size * pipeline_parallel_size * data_parallel_size

    spawn(
        init_parallel_context,
        world_size=WORLD_SIZE,
        seed=SEED,
        backend=BACKEND,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )
