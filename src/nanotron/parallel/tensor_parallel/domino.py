import re

import torch

from nanotron.parallel.comm import AsyncCommBucket

FWD_MLP_HANDLE_IDX = "fwd.layer_mlp_{}_batch_{}"
FWD_ATTN_HANDLE_IDX = "fwd.layer_attn_{}_batch_{}"
BWD_ATTN_HANDLE_IDX = "bwd.layer_attn_{}_batch_{}"
BWD_MLP_HANDLE_IDX = "bwd.layer_mlp_{}_batch_{}"


def is_async_comm(op_name: str):
    """
    There are two operations that we can't overlap
    for the forward pass: the last micro-batch of the mlp layer
    for the backward pass: the first micro-batch of the attention layer
    """
    NON_ASYNC_HANDLE_IDX = [
        # "fwd.layer_attn_{}_batch_0",
        # "fwd.layer_mlp_{}_batch_0",
        "fwd.layer_mlp_{}_batch_1",
        # "bwd.layer_mlp_{}_batch_1",
        "bwd.layer_attn_{}_batch_0",
    ]

    patterns = [p.replace("{}", r"\d+") for p in NON_ASYNC_HANDLE_IDX]  # Replace {} with regex for numbers
    regex = re.compile("^(" + "|".join(patterns) + ")$")  # Combine patterns into a single regex
    not_async = bool(regex.match(op_name))
    return not not_async


class WaitComm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, wait_handle_idx, comm_stream):
        ctx.wait_handle_idx = wait_handle_idx
        ctx.comm_stream = comm_stream
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)

        if "bwd.layer_mlp_1_batch_0" == ctx.wait_handle_idx:
            assert 1 == 1

        if "bwd.layer_mlp_0_batch_1" == ctx.wait_handle_idx:
            assert 1 == 1

        if is_async_comm(ctx.wait_handle_idx):
            from nanotron.constants import _AUTOGRAD_RUNS

            _AUTOGRAD_RUNS.append(f"wait_{ctx.wait_handle_idx}")
            handle = AsyncCommBucket.pop(ctx.wait_handle_idx)
            assert handle is not None
            handle.wait()
            torch.cuda.default_stream().wait_stream(ctx.comm_stream)
        else:
            from nanotron import constants

            constants._NOT_BWD_ASYNC_OPS.append(ctx.wait_handle_idx)

        # if "bwd.layer_mlp_0_batch_1" == ctx.wait_handle_idx:
        #     assert AsyncCommBucket._copy_async_op.get(ctx.wait_handle_idx).is_completed() is True

        return grad_output, None, None
