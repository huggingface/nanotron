import nanotron.fp8.distributed as fp8_dist
import pytest
import torch
import torch.distributed as dist
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.tensor import FP8Tensor
from nanotron.parallel import ParallelContext
from nanotron.testing.parallel import (
    init_distributed,
    rerun_if_address_is_in_use,
)


# @pytest.mark.skip("currently dont need all reduce for fp8 yet")
# @pytest.mark.parametrize("tp,dp,pp", [pytest.param(i, 1, 1) for i in range(1, min(4, available_gpus()) + 1)])
@pytest.mark.parametrize("tp,dp,pp", [(8, 1, 1)])
@pytest.mark.parametrize("is_fp8", [True])
@rerun_if_address_is_in_use()
def test_all_reduce(tp: int, dp: int, pp: int, is_fp8: bool):
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_all_reduce)(is_fp8=is_fp8)


def _test_all_reduce(parallel_context: ParallelContext, is_fp8: bool):
    pg = parallel_context.tp_pg
    tensor = torch.tensor(dist.get_rank(group=pg), device="cuda").float()
    ref_tensor = tensor.clone()

    dist.all_reduce(ref_tensor)

    assert 1 == 1

    tensor = FP8Tensor(tensor, dtype=DTypes.FP8E4M3, sync=True) if is_fp8 else tensor
    fp8_dist.all_reduce(
        tensor=tensor,
        op=dist.ReduceOp.SUM,
        group=pg,
    )

    assert tensor == sum(torch.arange(0, dist.get_world_size(pg)))
    parallel_context.destroy()
