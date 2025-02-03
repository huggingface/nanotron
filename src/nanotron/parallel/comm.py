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
        return AsyncCommBucket._async_op.pop(tensor_id)

    @staticmethod
    def wait(tensor_id: int):
        work = AsyncCommBucket._async_op.pop(tensor_id)
        work.wait()

    @staticmethod
    def clear_all():
        AsyncCommBucket._async_op.clear()


class WaitComm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, wait_handle_idx):
        ctx.wait_handle_idx = wait_handle_idx
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # import pydevd

        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        # if ctx.wait_handle_idx != "layer_1_batch_1":
        if ctx.wait_handle_idx != "layer_30_batch_1":
            handle = AsyncCommBucket.pop(ctx.wait_handle_idx)
            handle.wait()
            # assert 1 == 1
        return grad_output, None
