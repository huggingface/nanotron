from contextlib import contextmanager
from typing import Optional

import torch
from nanotron import distributed as dist
from nanotron.optim.gradient_accumulator import GradientAccumulator
from torch import nn


@contextmanager
def ddp_trigger_sync_in_bwd(model_ddp):
    """Trigger the sync of the gradients in the next backward pass of DDP model."""
    assert isinstance(model_ddp, torch.nn.parallel.DistributedDataParallel)
    old_require_backward_grad_sync = model_ddp.require_backward_grad_sync
    old_require_forward_param_sync = model_ddp.require_forward_param_sync

    model_ddp.require_backward_grad_sync = True
    model_ddp.require_forward_param_sync = True
    # https://github.com/pytorch/pytorch/blob/master/torch/csrc/distributed/c10d/reducer.cpp#L1325-L1356
    model_ddp.reducer.prepare_for_backward([])
    try:
        yield
    finally:
        model_ddp.require_backward_grad_sync = old_require_backward_grad_sync
        model_ddp.require_forward_param_sync = old_require_forward_param_sync


def sync_gradients_across_dp(
    module: nn.Module,
    dp_pg: dist.ProcessGroup,
    reduce_op: dist.ReduceOp,
    grad_accumulator: Optional[GradientAccumulator],
    **sync_options,
):
    """Sync gradients across data parallelism.

    Args:
        module: The module to sync gradients for.
        dp_pg: The data parallelism process group.
        reduce_op: The reduce operation to use.
        grad_accumulator: The gradient accumulator to use.
        sync_options: Additional options given when using `grad_accumulator`. Please look at `GradientAccumulator.sync_gradients_across_dp` for documentation
    """
    if grad_accumulator is not None:
        # This is an optimized path that
        grad_accumulator.sync_gradients_across_dp(dp_pg=dp_pg, reduce_op=reduce_op, **sync_options)
        return

    # Sync gradients
    for name, param in module.named_parameters():
        dist.all_reduce(param.grad, op=reduce_op, group=dp_pg)
