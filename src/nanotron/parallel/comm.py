from contextlib import contextmanager
from typing import Dict

import torch


class CudaStreamManager:
    _streams: Dict[str, "torch.cuda.Stream"] = {}

    @staticmethod
    def create(name: str):
        assert name not in CudaStreamManager._streams
        CudaStreamManager._streams[name] = torch.cuda.Stream()

    @staticmethod
    def get(name: str):
        return CudaStreamManager._streams.get(name)

    @contextmanager
    def run_on_stream(name: str):
        stream = CudaStreamManager.get(name)
        with torch.cuda.stream(stream):
            yield stream


class AsyncCommBucket:
    """

    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    RuntimeError: expected Variable or None (got tuple)
        Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    RuntimeError: expected Variable or None (got tuple)
    """

    _async_op: Dict[int, "dist.Work"] = {}

    @staticmethod
    def add(tensor_id: int, work: "dist.Work"):
        assert (
            tensor_id not in AsyncCommBucket._async_op
        ), f"tensor_id: {tensor_id}, keys: {AsyncCommBucket._async_op.keys()}"
        AsyncCommBucket._async_op[tensor_id] = work

    @staticmethod
    def get(tensor_id: int):
        return AsyncCommBucket._async_op.get(tensor_id)

    @staticmethod
    def pop(tensor_id: int):
        assert tensor_id in AsyncCommBucket._async_op, f"tensor_id: {tensor_id}"
        return AsyncCommBucket._async_op.pop(tensor_id)

    @staticmethod
    def wait(tensor_id: int):
        work = AsyncCommBucket._async_op.pop(tensor_id)
        work.wait()

    @staticmethod
    def clear_all():
        AsyncCommBucket._async_op.clear()


def is_async_comm(x):
    import re

    NON_ASYNC_HANDLE_IDX = [
        # "fwd.layer_attn_{}_batch_0",
        # "fwd.layer_mlp_{}_batch_0",
        # "fwd.layer_mlp_{}_batch_1",
        "bwd.layer_mlp_{}_batch_1",
        "bwd.layer_attn_{}_batch_0",
    ]

    patterns = [p.replace("{}", r"\d+") for p in NON_ASYNC_HANDLE_IDX]  # Replace {} with regex for numbers
    regex = re.compile("^(" + "|".join(patterns) + ")$")  # Combine patterns into a single regex
    not_async = bool(regex.match(x))
    return not not_async


class WaitComm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, wait_handle_idx):
        ctx.wait_handle_idx = wait_handle_idx
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)

        if "bwd.layer_mlp_1_batch_0" == ctx.wait_handle_idx:
            assert 1 == 1

        if is_async_comm(ctx.wait_handle_idx):
            handle = AsyncCommBucket.pop(ctx.wait_handle_idx)
            assert handle is not None
            handle.wait()
            # assert handle.is_completed() is True

        return grad_output, None
