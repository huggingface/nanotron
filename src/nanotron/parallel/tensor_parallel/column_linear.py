from typing import Optional

import torch
from torch.nn import functional as F

import nanotron.distributed as dist
from nanotron.parallel.utils import MemoryBuffer


class ColumnLinearContextParallel(torch.autograd.Function):
    """
    Column linear with memory_buffer for the allgather, context parallel
    enabled (i.e. tp_mode = TensorParallelLinearMode.REDUCE_SCATTER) and
    async communication disabled.
    """

    @staticmethod
    def forward(
        ctx, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], group: dist.ProcessGroup,
        tp_recompute_allgather: bool
    ):

        # Do allgather.
        sharded_batch_size, *rest_size = input.shape
        unsharded_batch_size = sharded_batch_size * group.size()
        if tp_recompute_allgather:
            total_input = MemoryBuffer().get("allgather", (unsharded_batch_size, *rest_size), dtype=input.dtype)
        else:
            total_input = torch.empty(unsharded_batch_size, *rest_size, dtype=input.dtype, device=input.device)
        dist.all_gather_into_tensor(total_input, input.contiguous(), group=group)

        # Prepare context.
        ctx.group = group
        ctx.tp_recompute_allgather = tp_recompute_allgather
        ctx.input_size = input.shape
        if tp_recompute_allgather:
            ctx.save_for_backward(input, weight, bias)
        else:
            ctx.save_for_backward(total_input, weight, bias)

        # Get linear output.
        out = F.linear(total_input, weight, bias)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Either allgather the inputs again or get them from context.
        group = ctx.group
        tp_recompute_allgather = ctx.tp_recompute_allgather
        input_size = ctx.input_size
        if tp_recompute_allgather:
            input, weight, bias = ctx.saved_tensors
            sharded_batch_size, *rest_size = input.shape
            total_input = sharded_batch_size * group.size()
            unsharded_batch_size = sharded_batch_size * group.size()
            total_input = MemoryBuffer().get("allgather", (unsharded_batch_size, *rest_size), dtype=input.dtype)
            dist.all_gather_into_tensor(total_input, input.contiguous(), group=group)
        else:
            total_input, weight, bias = ctx.saved_tensors

        # Get the grad_output and total_input on the correct views to be able to transpose them below.
        grad_output = grad_output.contiguous()
        assert grad_output.dim() == 3
        grad_output = grad_output.view(grad_output.size(0) * grad_output.size(1), grad_output.size(2))
        total_input = total_input.view(total_input.size(0) * total_input.size(1), total_input.size(2))

        # Compute gradients.
        grad_weight = grad_output.T @ total_input
        grad_input = grad_output @ weight
        sub_grad_input = torch.empty(input_size, dtype=total_input.dtype, device=total_input.device, requires_grad=False)
        dist.reduce_scatter_tensor(sub_grad_input, grad_input, group=group, op=dist.ReduceOp.SUM)
        grad_bias = torch.sum(grad_output, dim=0) if bias is not None else None

        return sub_grad_input, grad_weight, grad_bias, None, None


def column_linear_context_parallel(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], group: dist.ProcessGroup,
    tp_recompute_allgather: bool = False
):
    return ColumnLinearContextParallel.apply(input, weight, bias, group, tp_recompute_allgather)
