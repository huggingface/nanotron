import pytest

import torch
import torch.distributed as dist
import nanotron.fp8.distributed as fp8_dist
from nanotron.parallel import ParallelContext
from nanotron.fp8 import FP8Tensor, DTypes

from helpers.utils import init_distributed



@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
@pytest.mark.parametrize("op", [dist.ReduceOp.SUM, dist.ReduceOp.MIN])
@pytest.mark.parametrize("async_op", [True, False])
@pytest.mark.parametrize("world_size", [1, 2, 5])
def test_all_reduce(dtype, op, async_op, world_size):
    init_distributed(tp=world_size, dp=1, pp=1)(_test_all_reduce)(dtype=dtype, op=op, async_op=async_op)


def _test_all_reduce(parallel_context: ParallelContext, dtype: DTypes, op: dist.all_reduce, async_op):
    rank = dist.get_local_rank(parallel_context.tp_pg)
    tensor = torch.tensor(rank, dtype=torch.float32, device="cuda")
    tensor = FP8Tensor(tensor, dtype)
    
    fp8_dist.all_reduce(tensor, op, parallel_context.tp_pg, async_op)
