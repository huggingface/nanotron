from contextlib import contextmanager
from typing import Dict

import torch

from nanotron.parallel.tensor_parallel.domino import is_async_comm


class CudaStreamManager:
    _streams: Dict[str, "torch.cuda.Stream"] = {}

    @staticmethod
    def create(name: str, device: torch.device = None):
        assert name not in CudaStreamManager._streams
        CudaStreamManager._streams[name] = torch.cuda.Stream(device=device)

    @staticmethod
    def get(name: str):
        if name not in CudaStreamManager._streams:
            CudaStreamManager.create(name)
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
    _copy_async_op: Dict[int, "dist.Work"] = {}

    @staticmethod
    def add(op_name: int, work: "dist.Work"):
        assert op_name not in AsyncCommBucket._async_op, f"Operation with name: {op_name} already exists"
        AsyncCommBucket._async_op[op_name] = work
        AsyncCommBucket._copy_async_op[op_name] = work

    @staticmethod
    def get(op_name: int):
        if op_name not in AsyncCommBucket._async_op:
            raise KeyError(f"Operation with name: {op_name} doesn't exist")

        return AsyncCommBucket._async_op.get(op_name)

    @staticmethod
    def pop(op_name: int):
        if op_name not in AsyncCommBucket._async_op:
            raise KeyError(f"Operation with name: {op_name} doesn't exist")

        return AsyncCommBucket._async_op.pop(op_name)

    @staticmethod
    def wait(op_name: int):
        """Wait and remove the operation from the bucket"""
        work = AsyncCommBucket.pop(op_name)
        work.wait()

    @staticmethod
    def is_all_completed() -> bool:
        if not len(AsyncCommBucket._async_op) == 0:
            return False

        not_finished = []
        for k, v in AsyncCommBucket._copy_async_op.items():
            if v.is_completed() is not True:
                not_finished.append((k, v))
        return len(not_finished) == 0

    @staticmethod
    def clear_all():
        AsyncCommBucket._async_op.clear()
        AsyncCommBucket._copy_async_op.clear()


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
        if is_async_comm(ctx.wait_handle_idx):
            if "bwd.layer_mlp_27_batch_1" == ctx.wait_handle_idx:
                assert 1 == 1

            handle = AsyncCommBucket.pop(ctx.wait_handle_idx)
            assert handle is not None
            handle.wait()
            assert 1 == 1

            torch.cuda.current_stream().wait_stream(ctx.comm_stream)

        return grad_output, None, None
