import pytest
import torch
import torch.distributed as dist
from helpers.utils import (
    init_distributed,
    rerun_if_address_is_in_use,
)
from nanotron.parallel import ParallelContext
from nanotron.parallel.comm import AsyncCommBucket, CudaStreamManager, insert_backward_sync_to_tensor


class MockWork:
    def __init__(self):
        self.completed = False
        self.wait_called = False

    def wait(self):
        self.wait_called = True
        self.completed = True

    def is_completed(self):
        return self.completed


def test_cuda_stream_manager():
    manager = CudaStreamManager()
    manager.create("test", torch.device("cuda"))

    stream = manager.get("test")
    assert isinstance(stream, torch.cuda.Stream)


@rerun_if_address_is_in_use()
def test_add_async_op_to_bucket():
    init_distributed(tp=2, dp=1, pp=1)(_test_add_async_op_to_bucket)()


def _test_add_async_op_to_bucket(parallel_context: ParallelContext):
    OP_NAME = "test"
    tensor = torch.randn(1, device="cuda")
    work = dist.all_reduce(tensor, async_op=True)

    comm_bucket = AsyncCommBucket()
    comm_bucket.add(OP_NAME, work)

    assert comm_bucket.get(OP_NAME) is work


@rerun_if_address_is_in_use()
def test_wait_async_op_to_bucket():
    init_distributed(tp=2, dp=1, pp=1)(_test_wait_async_op_to_bucket)()


def _test_wait_async_op_to_bucket(parallel_context: ParallelContext):
    OP_NAME = "test"
    work = MockWork()
    comm_bucket = AsyncCommBucket()

    comm_bucket.add(OP_NAME, work)
    assert work.is_completed() is False

    comm_bucket.wait(OP_NAME)
    assert work.is_completed()
    with pytest.raises(KeyError):
        comm_bucket.get(OP_NAME)


@rerun_if_address_is_in_use()
def test_is_all_completed_in_async_bucket():
    init_distributed(tp=2, dp=1, pp=1)(_test_wait_async_op_to_bucket)()


def _test_wait_async_op_to_bucket(parallel_context: ParallelContext):
    OP_NAME = "test"
    work = MockWork()
    comm_bucket = AsyncCommBucket()

    comm_bucket.add(OP_NAME, work)
    assert comm_bucket.is_all_completed() is False

    comm_bucket.wait(OP_NAME)
    assert comm_bucket.is_all_completed() is True


@rerun_if_address_is_in_use()
def test_clear_ops_in_async_bucket():
    init_distributed(tp=2, dp=1, pp=1)(_test_clear_ops_in_async_bucket)()


def _test_clear_ops_in_async_bucket(parallel_context: ParallelContext):
    comm_bucket = AsyncCommBucket()

    comm_bucket.add("test1", MockWork())
    comm_bucket.add("test2", MockWork())
    comm_bucket.add("test3", MockWork())

    assert comm_bucket.is_all_completed() is False

    comm_bucket.clear_all()
    assert comm_bucket.is_all_completed() is True
    with pytest.raises(KeyError):
        comm_bucket.get("test1")


@rerun_if_address_is_in_use()
def test_wait_comm():
    init_distributed(tp=2, dp=1, pp=1)(_test_wait_comm)()


def _test_wait_comm(parallel_context: ParallelContext):
    OP_NAME = "test"
    tensor = torch.randn(1, device="cuda", requires_grad=True)
    stream_manager = CudaStreamManager()

    comm_stream = torch.cuda.Stream()

    with torch.cuda.stream(comm_stream):
        work = MockWork()
        stream_manager.comm_bucket.add(OP_NAME, work)

    output = insert_backward_sync_to_tensor(tensor, OP_NAME, stream_manager)
    assert work.is_completed() is False

    # NOTE: we test that it waits for the async op to complete
    # automatically in autograd
    (output + 1).backward()
    assert work.is_completed()
