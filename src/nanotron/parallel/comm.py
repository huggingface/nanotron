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
    _copy_async_op: Dict[int, "dist.Work"] = {}

    @staticmethod
    def add(tensor_id: int, work: "dist.Work"):
        assert (
            tensor_id not in AsyncCommBucket._async_op
        ), f"tensor_id: {tensor_id}, keys: {AsyncCommBucket._async_op.keys()}"
        AsyncCommBucket._async_op[tensor_id] = work
        AsyncCommBucket._copy_async_op[tensor_id] = work

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
    def is_all_completed() -> bool:
        assert len(AsyncCommBucket._async_op) == 0, "there are still some async ops haven't executed"

        not_finished = []
        for k, v in AsyncCommBucket._copy_async_op.items():
            if v.is_completed() is not True:
                not_finished.append((k, v))
        return len(not_finished) == 0

    @staticmethod
    def clear_all():
        AsyncCommBucket._async_op.clear()
        AsyncCommBucket._copy_async_op.clear()
