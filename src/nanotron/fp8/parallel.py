from functools import partial

import torch
from torch import nn

from nanotron.parallel import ParallelContext


class DistributedDataParallel:
    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        self.parallel_context = parallel_context

        self._parallelize(module)

    @torch.no_grad()
    def _parallelize(self, module) -> nn.Module:
        if self.parallel_context.data_parallel_size > 1:
            self._register_grad_avg_hook(module)

        return module

    def _register_grad_avg_hook(self, module: nn.Module):
        for p in module.parameters():
            p.register_hook(partial(self._average_grad))

    def _average_grad(self, grad: torch.Tensor, is_expert: bool):
        # NOTE: (grad1 + grad2 + ... + gradn) / n = grad1/n + grad2/n + ... + gradn/n
        assert 1 == 1
        # grad.div_(self.parallel_context.data_parallel_size)

        # all_reduce(
        #     grad,
        #     op=dist.ReduceOp.SUM,
        #     parallel_context=self.parallel_context,
        #     parallel_mode=ParallelMode.EXPERT_DATA if is_expert else ParallelMode.DATA,
        # )
