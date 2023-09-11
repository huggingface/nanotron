from contextlib import nullcontext

import pytest
import torch
from helpers.exception import assert_fail_except_rank_with
from helpers.utils import available_gpus, init_distributed
from torch import nn
from torch.distributed import GradBucket

from brrr.core import distributed as dist
from brrr.core.parallelism.data_parallelism.utils import ddp_trigger_sync_in_bwd
from brrr.core.parallelism.parameters import BRRRParameter
from brrr.core.process_groups_initializer import DistributedProcessGroups
from brrr.core.utils import assert_tensor_synced_across_pg


@pytest.mark.skipif(available_gpus() < 2, reason="Testing test_ddp_with_afab requires at least 2 gpus")
@pytest.mark.parametrize("accumulation_steps", [1, 3])
def test_ddp_with_afab(accumulation_steps):
    init_distributed(tp=1, dp=2, pp=1)(_test_ddp_with_afab)(accumulation_steps=accumulation_steps)


def _test_ddp_with_afab(dpg: DistributedProcessGroups, accumulation_steps: int):
    dist.get_rank(dpg.dp_pg)
    half_precision = torch.float16

    def allreduce_hook(process_group: dist.ProcessGroup, bucket: GradBucket):
        # DDP groups grads in GradBuckets. This hook is called throughout the bwd pass, once each bucket is ready to overlap communication with computation.
        # See https://pytorch.org/docs/stable/ddp_comm_hooks.html#what-does-a-communication-hook-operate-on for more details.
        half_flat_bucket_buffer = bucket.buffer()
        group_to_use = process_group if process_group is not None else dpg.dp_pg

        return (
            dist.all_reduce(half_flat_bucket_buffer, group=group_to_use, async_op=True, op=dist.ReduceOp.AVG)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

    model_hook = nn.Linear(3, 2, bias=False, dtype=half_precision, device="cuda")

    # Create BRRR Parameter
    model_hook.weight = BRRRParameter(model_hook.weight)

    model_ddp_hook = torch.nn.parallel.DistributedDataParallel(
        model_hook,
        process_group=dpg.dp_pg,
    )

    # Register DDP hook
    model_ddp_hook.register_comm_hook(state=None, hook=allreduce_hook)

    activations = []
    # All forward
    for i in range(accumulation_steps):
        input = torch.randn(5, 3, dtype=half_precision, device="cuda")

        with model_ddp_hook.no_sync():
            loss_hook = model_ddp_hook(input).sum()

        activations.append(loss_hook)

    # All backward
    for i in range(accumulation_steps):
        context = nullcontext()
        if i == accumulation_steps - 1:
            context = ddp_trigger_sync_in_bwd(model_ddp_hook)  # triggers a sync for the final backward
        loss_hook = activations[i]
        with context:
            loss_hook.backward()

        grad_hook = model_ddp_hook.module.weight.grad.clone()

        # Check that the gradients are synchronized across DP
        if i == accumulation_steps - 1:
            assert_tensor_synced_across_pg(grad_hook, dpg.dp_pg)
        else:
            with assert_fail_except_rank_with(AssertionError, rank_exception=0, pg=dpg.dp_pg):
                assert_tensor_synced_across_pg(grad_hook, dpg.dp_pg)
