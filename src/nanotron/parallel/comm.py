from contextlib import contextmanager
from typing import Dict

import torch

from nanotron.constants import CUDA_STREAM_COMM_NAME


class AsyncCommBucket:
    """
    Store aynchronous communication operations.
    """

    def __init__(self):
        self._async_op: Dict[int, "dist.Work"] = {}
        self._copy_async_op: Dict[int, "dist.Work"] = {}

    def add(self, op_name: int, work: "dist.Work"):
        assert op_name not in self._async_op, f"Operation with name: {op_name} already exists"
        assert work is not None
        self._async_op[op_name] = work
        self._copy_async_op[op_name] = work

    def get(self, op_name: str) -> "dist.Work":
        if op_name not in self._async_op:
            raise KeyError(f"Operation with name: {op_name} doesn't exist")

        return self._async_op.get(op_name)

    def pop(self, op_name: str) -> "dist.Work":
        if op_name not in self._async_op:
            raise KeyError(f"Operation with name: {op_name} doesn't exist")

        return self._async_op.pop(op_name)

    def wait(self, op_name: str):
        """Wait and remove the operation from the bucket"""
        work = self.pop(op_name)
        work.wait()

    def is_all_completed(self) -> bool:
        if not len(self._async_op) == 0:
            return False

        not_finished = []
        for k, v in self._copy_async_op.items():
            # assert is_domino_async_comm(k) is True, f"Operation with name {k} wasn't executed asynchronously!"
            if v.is_completed() is not True:
                not_finished.append((k, v))
        return len(not_finished) == 0

    def clear_all(self):
        self._async_op.clear()
        self._copy_async_op.clear()


class CudaStreamManager:
    def __init__(self):
        self._streams: Dict[str, "torch.cuda.Stream"] = {}
        self.comm_bucket = AsyncCommBucket()

    def init_default_comm_stream(self):
        """
        Initialize the default communication stream for the current cuda device.
        """
        self.create(CUDA_STREAM_COMM_NAME.format(torch.cuda.current_device()), torch.cuda.current_device())

    def create(self, name: str, device: torch.device):
        assert name not in self._streams
        self._streams[name] = torch.cuda.Stream(device=device)

    def get(self, name: str):
        if name not in self._streams:
            self.create(name)
        return self._streams.get(name)

    def get_default_comm_stream(self) -> torch.cuda.Stream:
        """
        Return the default communication stream for the current cuda device.
        """
        return self.get(CUDA_STREAM_COMM_NAME.format(torch.cuda.current_device()))

    @contextmanager
    def run_on_stream(self, name: str):
        stream = self.get(name)
        with torch.cuda.stream(stream):
            yield stream


def insert_backward_sync_to_tensor(
    tensor: torch.Tensor, op_name: str, stream_manager: CudaStreamManager
) -> torch.Tensor:
    """
    Insert a wait communication operation of a given op_name to the autograd graph
    of a tensor.
    """
    from nanotron.parallel.tensor_parallel.domino import WaitComm

    assert isinstance(stream_manager, CudaStreamManager)
    comm_stream = stream_manager.get(CUDA_STREAM_COMM_NAME.format(torch.cuda.current_device()))
    return WaitComm.apply(tensor, op_name, comm_stream, stream_manager.comm_bucket)
