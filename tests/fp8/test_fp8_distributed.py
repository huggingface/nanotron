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
from nanotron.testing.utils import available_gpus


@pytest.mark.parametrize("tp,dp,pp", [pytest.param(i, 1, 1) for i in range(1, min(4, available_gpus()) + 1)])
@pytest.mark.parametrize("is_fp8", [False, True])
@rerun_if_address_is_in_use()
def test_all_gather(tp: int, dp: int, pp: int, is_fp8: bool):
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_all_gather)(is_fp8=is_fp8)


def _test_all_gather(parallel_context: ParallelContext, is_fp8: bool):
    pg = parallel_context.tp_pg
    tensor = torch.tensor(dist.get_rank(group=pg), device="cuda").float()
    tensor = FP8Tensor(tensor, dtype=DTypes.FP8E4M3) if is_fp8 else tensor
    expected_output = torch.arange(parallel_context.tensor_parallel_size, device="cuda").float()
    expected_output = FP8Tensor(expected_output, dtype=DTypes.FP8E4M3) if is_fp8 else expected_output

    tensor_list = [torch.empty_like(tensor) for _ in range(parallel_context.tensor_parallel_size)]
    fp8_dist.all_gather(tensor_list, tensor, pg)

    assert expected_output.tolist() == torch.cat([x.unsqueeze(dim=0) for x in tensor_list]).tolist()

    parallel_context.destroy()


@pytest.mark.skip("currently dont need all reduce for fp8 yet")
@pytest.mark.parametrize("tp,dp,pp", [pytest.param(i, 1, 1) for i in range(1, min(4, available_gpus()) + 1)])
@pytest.mark.parametrize("is_fp8", [False, True])
@rerun_if_address_is_in_use()
def test_all_reduce(tp: int, dp: int, pp: int, is_fp8: bool):
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_all_reduce)(is_fp8=is_fp8)


def _test_all_reduce(parallel_context: ParallelContext, is_fp8: bool):
    pg = parallel_context.tp_pg
    tensor = torch.tensor(dist.get_rank(group=pg), device="cuda").float()
    tensor = FP8Tensor(tensor, dtype=DTypes.FP8E4M3) if is_fp8 else tensor

    fp8_dist.all_reduce(
        tensor=tensor,
        op=dist.ReduceOp.SUM,
        group=pg,
    )

    assert tensor == sum(torch.arange(0, dist.get_world_size(pg)))
    # assert x.dtype == temp.dtype
    # assert x.requires_grad == temp.requires_grad

    parallel_context.destroy()
