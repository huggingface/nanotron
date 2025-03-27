import functools
import operator

import torch
from torch import nn

from nanotron import distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.parallel.tied_parameters import get_tied_id_to_param
from nanotron.utils import Singleton


class MemoryBuffer(metaclass=Singleton):
    """
    Global memory buffer to store intermediate activations that need not to be cached for the backward pass.
    """

    def __init__(self):
        self.buffer = {}

    def get(self, name: str, shape: tuple[int], dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        required_numel = functools.reduce(operator.mul, shape, 1)
        if (name, dtype) not in self.buffer or self.buffer[name, dtype].numel() < required_numel:
            self.buffer[name, dtype] = torch.empty(
                required_numel, dtype=dtype, device=torch.cuda.current_device(), requires_grad=False
            )
        return self.buffer[name, dtype][:required_numel].view(shape)


def initial_sync(model: nn.Module, parallel_context: ParallelContext):
    # Synchronize across dp: basic assumption
    sorted_name_params = sorted(model.named_parameters(), key=lambda x: x[0])
    for name, param in sorted_name_params:
        dist.all_reduce(param, op=dist.ReduceOp.AVG, group=parallel_context.dp_pg)

    # Synchronize across tied weights: basic assumption
    for (_, group_ranks), param in sorted(
        get_tied_id_to_param(parameters=model.parameters(), root_module=model).items(), key=lambda x: x[0]
    ):
        group = parallel_context.world_ranks_to_pg[group_ranks]
        dist.all_reduce(param, op=dist.ReduceOp.AVG, group=group)
