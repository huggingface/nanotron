from typing import List, Union

import torch
import torch.distributed as dist
from torch.distributed import *  # noqa

from nanotron.distributed import *
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.tensor import FP8Tensor, convert_tensor_from_fp8
from nanotron.parallel.parameters import NanotronParameter, get_data_from_param


def all_reduce(
    tensor: Union[torch.Tensor, NanotronParameter],
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    assert tensor.__class__ in [torch.Tensor, NanotronParameter]
    data = get_data_from_param(tensor) if tensor.__class__ == NanotronParameter else tensor

    dist.all_reduce(data, op=op, group=group, async_op=async_op)


def all_gather(
    tensor_list: List[torch.Tensor],
    tensor: Union[FP8Tensor, NanotronParameter],
    group: dist.ProcessGroup,
    async_op: bool = False,
) -> torch.Tensor:
    tensor = tensor.data if tensor.__class__ == FP8Parameter else tensor
    # assert isinstance(tensor, FP8Tensor) if isinstance(tensor, FP8Tensor) else isinstance(tensor, torch.Tensor)

    if tensor.__class__ == FP8Tensor:
        # TODO(xrsrke): convert to the dtype of the first tensor in the list
        tensor = (
            convert_tensor_from_fp8(tensor, tensor.fp8_meta, torch.float32)
            if tensor_list[0].dtype != tensor.dtype
            else tensor
        )

    dist.all_gather(tensor_list, tensor, group, async_op)

    return tensor
