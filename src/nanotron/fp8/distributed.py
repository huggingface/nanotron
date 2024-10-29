from typing import List, Union

import torch
import torch.distributed as dist
from torch.distributed import *  # noqa

from nanotron.distributed import *
from nanotron.fp8.tensor import FP8Tensor, convert_tensor_from_fp8
from nanotron.parallel.parameters import NanotronParameter, get_data_from_param


def all_reduce(
    tensor: Union[torch.Tensor, NanotronParameter],
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    # NOTE: if keep nn.Parameter, got the following error:
    # File ".../.vscode-server/extensions/ms-python.debugpy-2024.12.0-linux-x64/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_vars.py", line 267, in eval_in_context
    #     result = eval(compiled, global_vars, local_vars)
    # File "<string>", line 1, in <module>
    # AttributeError: module 'torch.distributed.nn' has no attribute 'Parameter'
    assert tensor.__class__ in [torch.Tensor, torch.nn.Parameter, NanotronParameter]
    data = get_data_from_param(tensor) if tensor.__class__ == NanotronParameter else tensor

    # if data.__class__ == FP8Tensor:
    #     dist.all_reduce(data, op=op, group=group, async_op=async_op)
    # else:
    #     dist.all_reduce(data, op=op, group=group, async_op=async_op)
    dist.all_reduce(data, op=op, group=group, async_op=async_op)


def all_gather(
    tensor_list: List[torch.Tensor],
    tensor: Union[FP8Tensor, NanotronParameter],
    group: dist.ProcessGroup,
    async_op: bool = False,
) -> torch.Tensor:
    tensor = get_data_from_param(tensor) if tensor.__class__ == NanotronParameter else tensor

    if tensor.__class__ == FP8Tensor:
        # TODO(xrsrke): convert to the dtype of the first tensor in the list
        tensor = (
            convert_tensor_from_fp8(tensor, tensor.fp8_meta, torch.float32)
            if tensor_list[0].dtype != tensor.dtype
            else tensor
        )

    dist.all_gather(tensor_list=tensor_list, tensor=tensor, group=group, async_op=async_op)
