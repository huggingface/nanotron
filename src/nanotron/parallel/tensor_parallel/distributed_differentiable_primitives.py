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
from nanotron.parallel.comm import AsyncCommBucket
from nanotron.parallel.tensor_parallel.domino import is_async_comm


class DifferentiableIdentity(torch.autograd.Function):
    """All-reduce gradients in a differentiable fashion"""

    @staticmethod
    def forward(
        ctx,
        tensor,
        group: Optional[ProcessGroup],
        op_name: str = None,
        comm_stream: torch.cuda.Stream = None,
    ):
        ctx.group = group
        ctx.op_name = op_name
        ctx.comm_stream = comm_stream
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        op_name = ctx.op_name.replace("fwd.", "bwd.") if ctx.op_name is not None else ctx.op_name
        return (
            DifferentiableAllReduceSum.apply(grad_output, group, op_name, ctx.comm_stream),
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
        op_name: str = None,
        comm_stream: torch.cuda.Stream = None,
    ) -> Tuple[torch.Tensor, Optional["dist.Work"]]:
        ctx.op_name = op_name
        ctx.comm_stream = comm_stream

        if group.size() == 1:
            return tensor

        async_all_reduce = is_async_comm(op_name) if op_name is not None else False
        with torch.cuda.stream(comm_stream):
            if async_all_reduce:
                handle = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=True)
                AsyncCommBucket.add(op_name, handle)
            else:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)

        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


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


def differentiable_identity(tensor, group: Optional[ProcessGroup] = None, op_name: str = None):
    return DifferentiableIdentity.apply(tensor, group, op_name)


def differentiable_all_reduce_sum(tensor, group: Optional[ProcessGroup] = None, op_name: str = None):
    return DifferentiableAllReduceSum.apply(tensor, group, op_name)


def differentiable_all_gather(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableAllGather.apply(tensor, group)


def differentiable_reduce_scatter_sum(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableReduceScatterSum.apply(tensor, group)
