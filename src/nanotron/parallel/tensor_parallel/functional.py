# coding=utf-8
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
import math
from typing import Optional

import torch
from torch.nn import functional as F

import nanotron.distributed as dist
from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import (
    differentiable_all_gather,
    differentiable_all_reduce_sum,
    differentiable_identity,
    differentiable_reduce_scatter_sum,
)
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.utils import assert_cuda_max_connections_set_to_1


class _ShardedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        sharded_logits,  # (batch_size, length, sharded_hidden_size)
        target,  # (batch_size, length)
        group: dist.ProcessGroup,
    ):
        # Maximum value along last dimension across all GPUs.
        logits_max = torch.max(sharded_logits, dim=-1)[0]
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
        # Subtract the maximum value.
        sharded_logits = sharded_logits - logits_max.unsqueeze(dim=-1)

        # Get the shard's indices
        sharded_hidden_size = sharded_logits.shape[-1]
        rank = dist.get_rank(group)
        start_index = rank * sharded_hidden_size
        end_index = start_index + sharded_hidden_size

        # Create a mask of valid ids (1 means it needs to be masked).
        target_mask = (target < start_index) | (target >= end_index)
        masked_target = target.clone() - start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, shard-size] and target to a 1-D tensor of size [*].
        logits_2d = sharded_logits.view(-1, sharded_hidden_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.shape[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        if predicted_logits_1d.is_contiguous():
            predicted_logits_1d = predicted_logits_1d.clone()
        else:
            predicted_logits_1d = predicted_logits_1d.contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=group)

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = sharded_logits
        torch.exp(sharded_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=group)

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss.view_as(target)

    @staticmethod
    def backward(ctx, grad_output):
        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        sharded_hidden_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, sharded_hidden_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None


def sharded_cross_entropy(sharded_logits, target, group: dist.ProcessGroup, dtype: torch.dtype = None):
    """Helper function for the cross entropy."""
    if dtype is not None:
        # Cast input to specific dtype.
        sharded_logits = sharded_logits.to(dtype=dtype)
    return _ShardedCrossEntropy.apply(sharded_logits, target, group)


class _ColumnLinearAsyncCommunication(torch.autograd.Function):
    """Adapted from https://github.com/NVIDIA/Megatron-LM/blob/e6d7e09845590d0a36bc7f29eb28db974fb8da4e/megatron/core/tensor_parallel/layers.py#L215"""

    @staticmethod
    @assert_cuda_max_connections_set_to_1
    def forward(ctx, tensor, weight, bias, group, tp_mode):
        ctx.use_bias = bias is not None
        ctx.tp_mode = tp_mode
        ctx.group = group

        if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
            gathered_tensor = tensor
            ctx.save_for_backward(tensor, weight)
            return F.linear(gathered_tensor, weight, bias)
        elif tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
            group_size = group.size()
            current_rank = dist.get_rank(group)
            if group_size == 1:
                gathered_tensor = tensor
                ctx.save_for_backward(tensor, weight)
                return F.linear(gathered_tensor, weight, bias)
            else:
                # `tensor` can sometimes not be contiguous
                # https://cs.github.com/pytorch/pytorch/blob/2b267fa7f28e18ca6ea1de4201d2541a40411457/torch/distributed/nn/functional.py#L317
                tensor = tensor.contiguous()
                ctx.save_for_backward(tensor, weight)

                # TODO @thomasw21: gather along another dimension
                sharded_batch_size, *intermediate_size, hidden_size = tensor.shape
                if group is None:
                    group = dist.distributed_c10d._get_default_group()
                gathered_batch_size = sharded_batch_size * group.size()

                gathered_tensor = torch.empty(
                    gathered_batch_size,
                    *intermediate_size,
                    hidden_size,
                    device=tensor.device,
                    dtype=tensor.dtype,
                    requires_grad=tensor.requires_grad,
                )

                handle = dist.all_gather_into_tensor(gathered_tensor, tensor, group=group, async_op=True)

                # Compute a shard of column_linear in the same time of AllGather
                # We could compute the matmul of current holding shard and the current rank's weight
                # We assume that rank 0 holds w0, rank 1 holds w1, etc.
                # weights: w0 w1 w2 w3
                # rank 0:  X  -  -  -
                # rank 1:  -  X  -  -
                # rank 2:  -  -  X  -
                # rank 3:  -  -  -  X
                # We call the corresponding shard of output "same_device_shard"
                output_size = weight.shape[0]
                gathered_output = torch.empty(
                    gathered_batch_size,
                    *intermediate_size,
                    output_size,
                    device=tensor.device,
                    dtype=tensor.dtype,
                    requires_grad=tensor.requires_grad,
                )
                before_shard, same_device_shard, after_shard = torch.split(
                    gathered_output,
                    split_size_or_sections=[
                        sharded_batch_size * current_rank,
                        sharded_batch_size,
                        sharded_batch_size * (group_size - current_rank - 1),
                    ],
                    dim=0,
                )
                first_dims = math.prod([sharded_batch_size, *intermediate_size])
                if bias is None:
                    torch.mm(
                        input=tensor.view(first_dims, hidden_size),
                        mat2=weight.t(),
                        out=same_device_shard.view(first_dims, output_size),
                    )
                else:
                    torch.addmm(
                        input=bias[None, :],
                        mat1=tensor.view(first_dims, hidden_size),
                        mat2=weight.t(),
                        out=same_device_shard.view(first_dims, output_size),
                    )

                # Wait communication
                handle.wait()

                # Compute all the other shards that are obtained from AllGather
                # weights: w0 w1 w2 w3
                # rank 0:  -  X  X  X
                # rank 1:  X  -  X  X
                # rank 2:  X  X  -  X
                # rank 3:  X  X  X  -
                # As they could be not contiguous (r1 and r2) vertically as they are separated by "same_device_shard"
                # We need to compute them separately, i.e. "before_shard" and "after_shard"
                # For r0, "before_shard" is empty. For r3, "after_shard" is empty.
                if before_shard.numel() > 0:
                    first_dims = math.prod(before_shard.shape[:-1])
                    if bias is None:
                        torch.mm(
                            input=gathered_tensor[: sharded_batch_size * current_rank].view(first_dims, hidden_size),
                            mat2=weight.t(),
                            out=before_shard.view(first_dims, output_size),
                        )
                    else:
                        torch.addmm(
                            input=bias[None, :],
                            mat1=gathered_tensor[: sharded_batch_size * current_rank].view(first_dims, hidden_size),
                            mat2=weight.t(),
                            out=before_shard.view(first_dims, output_size),
                        )
                if after_shard.numel() > 0:
                    first_dims = math.prod(after_shard.shape[:-1])
                    if bias is None:
                        torch.mm(
                            input=gathered_tensor[sharded_batch_size * (current_rank + 1) :].view(
                                first_dims, hidden_size
                            ),
                            mat2=weight.t(),
                            out=after_shard.view(first_dims, output_size),
                        )
                    else:
                        torch.addmm(
                            input=bias[None, :],
                            mat1=gathered_tensor[sharded_batch_size * (current_rank + 1) :].view(
                                first_dims, hidden_size
                            ),
                            mat2=weight.t(),
                            out=after_shard.view(first_dims, output_size),
                        )

                return gathered_output
        else:
            raise ValueError(f"Got unexpected mode: {tp_mode}.")

    @staticmethod
    @assert_cuda_max_connections_set_to_1
    def backward(ctx, grad_output):
        tensor, weight = ctx.saved_tensors
        group = ctx.group
        use_bias = ctx.use_bias
        tp_mode = ctx.tp_mode

        handle: Optional[dist.Work] = None
        if tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
            # TODO @thomasw21: gather along another dimension
            sharded_batch_size, *rest_size = tensor.shape
            if group is None:
                group = dist.distributed_c10d._get_default_group()

            if group.size() == 1:
                total_tensor = tensor
            else:
                unsharded_batch_size = sharded_batch_size * group.size()

                unsharded_tensor = torch.empty(
                    unsharded_batch_size,
                    *rest_size,
                    device=tensor.device,
                    dtype=tensor.dtype,
                    requires_grad=False,
                )
                handle = dist.all_gather_into_tensor(unsharded_tensor, tensor, group=group, async_op=True)
                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # gather is scheduled before the tensor gradient computation
                total_tensor = unsharded_tensor
        else:
            total_tensor = tensor

        grad_tensor = grad_output.matmul(weight)

        if handle is not None:
            handle.wait()

        # Doing gather + slicing during the NeMo forward pass can make this tensor
        # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
        # clones it if it's not contiguous:
        # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        grad_output_first_dims, grad_output_last_dim = grad_output.shape[:-1], grad_output.shape[-1]
        total_tensor_first_dims, total_tensor_last_dim = total_tensor.shape[:-1], total_tensor.shape[-1]
        grad_output = grad_output.view(math.prod(grad_output_first_dims), grad_output_last_dim)
        total_tensor = total_tensor.view(math.prod(total_tensor_first_dims), total_tensor_last_dim)

        handle: Optional[dist.Work] = None
        if tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
            if group.size() == 1:
                sub_grad_tensor = grad_tensor
            else:
                sub_grad_tensor = torch.empty(
                    tensor.shape, dtype=grad_tensor.dtype, device=grad_tensor.device, requires_grad=False
                )
                # reduce_scatter
                handle = dist.reduce_scatter_tensor(sub_grad_tensor, grad_tensor, group=group, async_op=True)
                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # reduce scatter is scheduled before the weight gradient computation
        elif tp_mode is TensorParallelLinearMode.ALL_REDUCE:
            # Asynchronous all-reduce
            handle = dist.all_reduce(grad_tensor, group=group, async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation
        else:
            raise ValueError()

        # TODO @thomasw21: This sounds like we don't have the optimal physical layout
        grad_weight = grad_output.t().matmul(total_tensor)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if handle is not None:
            handle.wait()

        if tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
            return sub_grad_tensor, grad_weight, grad_bias, None, None
        elif tp_mode is TensorParallelLinearMode.ALL_REDUCE:
            return grad_tensor, grad_weight, grad_bias, None, None
        else:
            raise ValueError(f"Got unexpected mode: {tp_mode}.")


def column_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    group: dist.ProcessGroup,
    tp_mode: TensorParallelLinearMode,
    async_communication: bool,
):
    if async_communication:
        return _ColumnLinearAsyncCommunication.apply(input, weight, bias, group, tp_mode)

    if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        input = differentiable_identity(input, group=group)
    elif tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
        input = differentiable_all_gather(input, group=group)
    else:
        raise ValueError(f"Got unexpected mode: {tp_mode}.")

    return F.linear(input, weight, bias)


class _RowLinearAsyncCommunication(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, weight, bias, group, tp_mode):
        assert (
            tp_mode is TensorParallelLinearMode.REDUCE_SCATTER
        ), f"async communication in RowLinear only supports REDUCE_SCATTER, got {tp_mode}"

        if group is None:
            group = dist.distributed_c10d._get_default_group()

        ctx.use_bias = bias is not None
        ctx.group = group

        out = F.linear(tensor, weight, bias)

        if group.size() > 1:
            out = differentiable_reduce_scatter_sum(out, group=group)

        ctx.save_for_backward(tensor, weight)
        return out

    @staticmethod
    @assert_cuda_max_connections_set_to_1
    def backward(ctx, grad_output):
        tensor, weight = ctx.saved_tensors
        group = ctx.group
        use_bias = ctx.use_bias

        handle_0: Optional[dist.Work] = None
        handle_1: Optional[dist.Work] = None

        # TODO @thomasw21: gather along another dimension
        sharded_batch_size, *rest_size = grad_output.shape

        if group.size() == 1:
            total_grad_output = grad_output
        else:
            unsharded_batch_size = sharded_batch_size * group.size()

            total_grad_output = torch.empty(
                unsharded_batch_size,
                *rest_size,
                device=grad_output.device,
                dtype=grad_output.dtype,
                requires_grad=False,
            )

            # Doing gather + slicing during the NeMo forward pass can make this tensor
            # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
            # clones it if it's not contiguous:
            # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
            grad_output = grad_output.contiguous()

            handle_0 = dist.all_gather_into_tensor(total_grad_output, grad_output, group=group, async_op=True)

        grad_tensor = grad_output.matmul(weight)

        # wait for the first all_gather to finish before starting the second all_gather
        if handle_0 is not None:
            handle_0.wait()

        # TODO @thomasw21: gather along another dimension
        sharded_batch_size, *rest_size = grad_tensor.shape

        if group.size() == 1:
            total_grad_tensor = grad_tensor
        else:
            unsharded_batch_size = sharded_batch_size * group.size()

            total_grad_tensor = torch.empty(
                unsharded_batch_size,
                *rest_size,
                device=grad_tensor.device,
                dtype=grad_tensor.dtype,
                requires_grad=False,
            )

            handle_1 = dist.all_gather_into_tensor(total_grad_tensor, grad_tensor, group=group, async_op=True)

        # Convert the tensor shapes to 2D for execution compatibility
        tensor = tensor.contiguous()
        tensor_first_dims, tensor_last_dim = tensor.shape[:-1], tensor.shape[-1]
        tensor = tensor.view(math.prod(tensor_first_dims), tensor_last_dim)

        # Convert the tensor shapes to 2D for execution compatibility
        total_grad_output_first_dims, total_grad_output_last_dim = (
            total_grad_output.shape[:-1],
            total_grad_output.shape[-1],
        )
        total_grad_output = total_grad_output.view(math.prod(total_grad_output_first_dims), total_grad_output_last_dim)

        # TODO @thomasw21: This sounds like we don't have the optimal physical layout
        grad_weight = total_grad_output.t().matmul(tensor)
        grad_bias = total_grad_output.sum(dim=0) if use_bias else None

        if handle_1 is not None:
            handle_1.wait()

        return total_grad_tensor, grad_weight, grad_bias, None, None


def row_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    group: dist.ProcessGroup,
    tp_mode: TensorParallelLinearMode,
    async_communication: bool,
):
    if async_communication:
        return _RowLinearAsyncCommunication.apply(input, weight, bias, group, tp_mode)

    out = F.linear(input, weight, bias)

    if tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        out = differentiable_all_reduce_sum(out, group=group)
    elif tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
        out = differentiable_reduce_scatter_sum(out, group=group)
    else:
        raise ValueError(f"Got unexpected mode: {tp_mode}.")

    return out
