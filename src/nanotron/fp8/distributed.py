import torch.distributed as dist

from nanotron.fp8.tensor import FP8Tensor


def all_reduce(tensor: FP8Tensor, op: dist.ReduceOp, group: dist.ProcessGroup, async_op: bool = False) -> FP8Tensor:
    pass
