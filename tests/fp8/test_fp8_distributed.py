import nanotron.fp8.distributed as fp8_dist
import pytest
import torch
import torch.distributed as dist
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.tensor import FP8Tensor
from nanotron.parallel import ParallelContext
from utils import set_system_path

set_system_path()

from tests.helpers.utils import init_distributed, rerun_if_address_is_in_use


@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
@pytest.mark.parametrize("op", [dist.ReduceOp.SUM, dist.ReduceOp.MIN])
@pytest.mark.parametrize("async_op", [True, False])
@pytest.mark.parametrize("world_size", [1, 2, 5])
@rerun_if_address_is_in_use()
def test_all_reduce(dtype, op, async_op, world_size):
    init_distributed(tp=world_size, dp=1, pp=1)(_test_all_reduce)(dtype=dtype, op=op, async_op=async_op)


def _test_all_reduce(parallel_context: ParallelContext, dtype: DTypes, op: dist.all_reduce, async_op):
    rank = dist.get_local_rank(parallel_context.tp_pg)
    tensor = torch.tensor(rank, dtype=torch.float32, device="cuda")
    tensor = FP8Tensor(tensor, dtype)

    fp8_dist.all_reduce(tensor, op, parallel_context.tp_pg, async_op)
