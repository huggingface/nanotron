import re
import threading
from contextlib import contextmanager
from typing import Optional

import torch

from nanotron.parallel.comm import AsyncCommBucket, CudaStreamManager

FWD_MLP_OP_NAME = "fwd.layer_mlp_{}_batch_{}"
FWD_ATTN_OP_NAME = "fwd.layer_attn_{}_batch_{}"
BWD_ATTN_OP_NAME = "bwd.layer_attn_{}_batch_{}"
BWD_MLP_OP_NAME = "bwd.layer_mlp_{}_batch_{}"

_operation_context = threading.local()


def is_domino_async_comm(x: str) -> bool:
    """
    Determine whether a module (e.g., mlp, attention)
    performs all-reduce asynchronously in tensor parallelism
    """
    NON_ASYNC_HANDLE_IDX = [
        # "fwd.layer_mlp_{}_batch_1",
        "bwd.layer_attn_{}_batch_0",
    ]

    patterns = [p.replace("{}", r"\d+") for p in NON_ASYNC_HANDLE_IDX]  # Replace {} with regex for numbers
    regex = re.compile("^(" + "|".join(patterns) + ")$")  # Combine patterns into a single regex
    not_async = bool(regex.match(x))
    return not not_async


class OperationContext:
    """
    Context manager that sets both operation name and stream manager in thread-local context.

    Args:
        op_name: Name of the current operation
        stream_manager: Associated CUDA stream manager
    """

    def __init__(self, op_name: str, stream_manager: CudaStreamManager):
        self.op_name = op_name
        self.stream_manager = stream_manager
        self._previous_op_name: Optional[str] = None
        self._previous_stream_manager: Optional[CudaStreamManager] = None

    def __enter__(self):
        """Store current context and set new values"""
        # Handle operation name
        if not hasattr(_operation_context, "_current_op_name"):
            _operation_context._current_op_name = None
        self._previous_op_name = _operation_context._current_op_name
        _operation_context._current_op_name = self.op_name

        # Handle stream manager
        if not hasattr(_operation_context, "_current_stream_manager"):
            _operation_context._current_stream_manager = None
        self._previous_stream_manager = _operation_context._current_stream_manager
        _operation_context._current_stream_manager = self.stream_manager

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous context values"""
        _operation_context._current_op_name = self._previous_op_name
        _operation_context._current_stream_manager = self._previous_stream_manager


class WaitComm(torch.autograd.Function):
    """
    Enforce a tensor to wait for the communication operation to finish
    in torch's autograd graph.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, op_name: str, comm_stream: torch.cuda.Stream, comm_bucket: AsyncCommBucket):
        assert isinstance(comm_stream, torch.cuda.Stream)
        ctx.op_name = op_name
        ctx.comm_stream = comm_stream
        ctx.comm_bucket = comm_bucket
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        NOTE: because the communication operation is already being executed
        so the communication stream don't have to wait for the compute stream here
        but the compute stream waits for the communication stream
        before proceeding
        """
        if is_domino_async_comm(ctx.op_name):
            handle = ctx.comm_bucket.pop(ctx.op_name)
            handle.wait()

            ctx.comm_stream.synchronize()
            torch.cuda.default_stream().wait_stream(ctx.comm_stream)

        return grad_output, None, None, None


@contextmanager
def set_operation_context(name: str, stream_manager: CudaStreamManager):
    with OperationContext(name, stream_manager):
        yield


def get_op_name() -> Optional[str]:
    """
    Get the name of the current operation.
    """
    return getattr(_operation_context, "_current_op_name", None)


def get_stream_manager() -> Optional[CudaStreamManager]:
    """
    Get the stream manager for the current operation.
    """
    return getattr(_operation_context, "_current_stream_manager", None)
