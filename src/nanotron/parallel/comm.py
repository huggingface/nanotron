from contextlib import contextmanager
from typing import Dict

import torch

from nanotron.parallel.tensor_parallel.domino import is_async_comm


class CudaStreamManager:
    def __init__(self):
        self._streams: Dict[str, "torch.cuda.Stream"] = {}

    def create(self, name: str, device: torch.device):
        assert name not in self._streams
        self._streams[name] = torch.cuda.Stream(device=device)

    def get(self, name: str):
        if name not in self._streams:
            self.create(name)
        return self._streams.get(name)

    @contextmanager
    def run_on_stream(self, name: str):
        stream = self.get(name)
        with torch.cuda.stream(stream):
            yield stream


class AsyncCommBucket:
    _async_op: Dict[int, "dist.Work"] = {}
    _copy_async_op: Dict[int, "dist.Work"] = {}

    @staticmethod
    def add(op_name: int, work: "dist.Work"):
        assert op_name not in AsyncCommBucket._async_op, f"Operation with name: {op_name} already exists"
        assert work is not None
        AsyncCommBucket._async_op[op_name] = work
        AsyncCommBucket._copy_async_op[op_name] = work

    @staticmethod
    def get(op_name: str) -> "dist.Work":
        if op_name not in AsyncCommBucket._async_op:
            raise KeyError(f"Operation with name: {op_name} doesn't exist")

        return AsyncCommBucket._async_op.get(op_name)

    @staticmethod
    def pop(op_name: str) -> "dist.Work":
        if op_name not in AsyncCommBucket._async_op:
            raise KeyError(f"Operation with name: {op_name} doesn't exist")

        return AsyncCommBucket._async_op.pop(op_name)

    @staticmethod
    def wait(op_name: str):
        """Wait and remove the operation from the bucket"""
        work = AsyncCommBucket.pop(op_name)
        work.wait()

    @staticmethod
    def is_all_completed() -> bool:
        if not len(AsyncCommBucket._async_op) == 0:
            return False

        not_finished = []
        for k, v in AsyncCommBucket._copy_async_op.items():
            assert is_async_comm(k) is True, f"Operation with name {k} wasn't executed asynchronously!"
            if v.is_completed() is not True:
                not_finished.append((k, v))
        return len(not_finished) == 0

    @staticmethod
    def clear_all():
        AsyncCommBucket._async_op.clear()
        AsyncCommBucket._copy_async_op.clear()


class WaitComm(torch.autograd.Function):
    """
    Enforce a tensor to wait for the communication operation to finish
    in torch's autograd graph
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, op_name: str, comm_stream: torch.cuda.Stream):
        ctx.op_name = op_name
        ctx.comm_stream = comm_stream
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        NOTE: because the communication operation is already being executed
        so the communication stream don't have to wait for the compute stream here
        but the compute stream waits for the communication stream
        before proceeding
        """
        if is_async_comm(ctx.op_name):
            handle = AsyncCommBucket.pop(ctx.op_name)
            handle.wait()

            ctx.comm_stream.synchronize()
            torch.cuda.default_stream().wait_stream(ctx.comm_stream)

        return grad_output, None, None
