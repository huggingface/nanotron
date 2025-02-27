# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import torch
from torch import distributed as torch_dist

from nanotron import distributed as dist
from nanotron.distributed import ProcessGroup
from nanotron.parallel.comm import AsyncCommBucket, is_async_comm


class DifferentiableIdentity(torch.autograd.Function):
    """All-reduce gradients in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup], async_all_reduce: bool, handle_idx=None, comm_stream=None):
        ctx.async_all_reduce = async_all_reduce
        ctx.handle_idx = handle_idx
        ctx.group = group
        ctx.comm_stream = comm_stream
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)

        group = ctx.group
        handle_idx = ctx.handle_idx.replace("fwd.", "bwd.") if ctx.handle_idx is not None else None
        async_all_reduce = is_async_comm(handle_idx) if handle_idx is not None else ctx.async_all_reduce

        if handle_idx is not None and "bwd.layer_mlp_" in handle_idx and "batch_1" in handle_idx:
            assert 1 == 1

        return (
            DifferentiableAllReduceSum.apply(grad_output, group, async_all_reduce, handle_idx, ctx.comm_stream),
            None,
            None,
            None,
            None,
        )


class DifferentiableAllReduceSum(torch.autograd.Function):
    """All-reduce in a differentiable fashion"""

    @staticmethod
    def forward(
        ctx,
        tensor,
        group: Optional[ProcessGroup],
        async_all_reduce: bool,
        handle_idx: Optional[int] = None,
        comm_stream=None,
    ) -> Tuple[torch.Tensor, Optional["dist.Work"]]:
        from contextlib import nullcontext

        ctx.async_all_reduce = async_all_reduce

        if group.size() == 1:
            return tensor

        if comm_stream is not None:
            comm_stream.wait_stream(torch.cuda.default_stream())
            comm_context = torch.cuda.stream(comm_stream)
        else:
            comm_context = nullcontext()

        with comm_context:
            if async_all_reduce is True:
                assert comm_stream is not None
                handle = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=True)
                AsyncCommBucket.add(handle_idx, handle)
            else:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)

        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


class DifferentiableAllGather(torch.autograd.Function):
    """All gather in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        ctx.group = group

        if group.size() == 1:
            return tensor

        # TODO @thomasw21: gather along another dimension
        sharded_batch_size, *rest_size = tensor.shape
        if group is None:
            group = torch_dist.distributed_c10d._get_default_group()
        unsharded_batch_size = sharded_batch_size * group.size()

        unsharded_tensor = torch.empty(
            unsharded_batch_size,
            *rest_size,
            device=tensor.device,
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
        )

        # `tensor` can sometimes not be contiguous
        # https://cs.github.com/pytorch/pytorch/blob/2b267fa7f28e18ca6ea1de4201d2541a40411457/torch/distributed/nn/functional.py#L317
        tensor = tensor.contiguous()

        dist.all_gather_into_tensor(unsharded_tensor, tensor, group=group)
        return unsharded_tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        out = DifferentiableReduceScatterSum.apply(grad_output, group)
        return out, None


class DifferentiableReduceScatterSum(torch.autograd.Function):
    """Reduce scatter in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        ctx.group = group

        if group.size() == 1:
            return tensor

        # TODO @thomasw21: shard along another dimension
        unsharded_batch_size, *rest_size = tensor.shape
        if group is None:
            group = torch_dist.distributed_c10d._get_default_group()
        assert unsharded_batch_size % group.size() == 0

        # TODO @thomasw21: Collectives seem to require tensors to be contiguous
        # https://cs.github.com/pytorch/pytorch/blob/2b267fa7f28e18ca6ea1de4201d2541a40411457/torch/distributed/nn/functional.py#L305
        tensor = tensor.contiguous()

        sharded_tensor = torch.empty(
            unsharded_batch_size // group.size(),
            *rest_size,
            device=tensor.device,
            dtype=tensor.dtype,
            requires_grad=False,
        )
        dist.reduce_scatter_tensor(sharded_tensor, tensor, group=group, op=dist.ReduceOp.SUM)
        return sharded_tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        return DifferentiableAllGather.apply(grad_output, group), None


# -----------------
# Helper functions.
# -----------------


def differentiable_identity(
    tensor, group: Optional[ProcessGroup] = None, async_all_reduce: bool = False, handle_idx=None, comm_stream=None
):
    return DifferentiableIdentity.apply(tensor, group, async_all_reduce, handle_idx, comm_stream)


def differentiable_all_reduce_sum(
    tensor, group: Optional[ProcessGroup] = None, async_all_reduce: bool = False, handle_idx=None, comm_stream=None
):
    return DifferentiableAllReduceSum.apply(tensor, group, async_all_reduce, handle_idx, comm_stream)


def differentiable_all_gather(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableAllGather.apply(tensor, group)


def differentiable_reduce_scatter_sum(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableReduceScatterSum.apply(tensor, group)
