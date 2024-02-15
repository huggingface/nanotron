import torch
from helpers.utils import (
    rerun_if_address_is_in_use,
    spawn,
)
from nanotron.parallel import ParallelContext


@rerun_if_address_is_in_use(max_try=2)
def test_rerun():
    spawn(_test_rerun, tp=2, dp=1, pp=1)


def _test_rerun(
    # rank: int, world_size: int,
    tp: int,
    pp: int,
    dp: int,
    # port: int,
):
    # setup_dist_env(rank, world_size, port)
    parallel_context = ParallelContext(data_parallel_size=dp, pipeline_parallel_size=pp, tensor_parallel_size=tp)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # if torch.randint(0, 6, (1,)).item() < 4:
    #     raise Exception("Address already in use")

    parallel_context.destroy()
