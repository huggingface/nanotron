import pytest
import torch
import torch.distributed as dist
from helpers.utils import (
    init_distributed,
    rerun_if_address_is_in_use,
)
from nanotron.parallel import ParallelContext
from nanotron.parallel.comm import AsyncCommBucket, WaitComm


class MockWork:
    def __init__(self):
        self.completed = False
        self.wait_called = False

    def wait(self):
        self.wait_called = True
        self.completed = True

    def is_completed(self):
        return self.completed


@rerun_if_address_is_in_use()
def test_add_async_op_to_bucket():
    init_distributed(tp=2, dp=1, pp=1)(_test_add_async_op_to_bucket)()


def _test_add_async_op_to_bucket(parallel_context: ParallelContext):
    OP_NAME = "test"
    tensor = torch.randn(1, device="cuda")
    work = dist.all_reduce(tensor, async_op=True)

    AsyncCommBucket.add(OP_NAME, work)

    assert AsyncCommBucket.get(OP_NAME) is work


@rerun_if_address_is_in_use()
def test_wait_async_op_to_bucket():
    init_distributed(tp=2, dp=1, pp=1)(_test_wait_async_op_to_bucket)()


def _test_wait_async_op_to_bucket(parallel_context: ParallelContext):
    OP_NAME = "test"
    work = MockWork()

    AsyncCommBucket.add(OP_NAME, work)
    assert work.is_completed() is False

    AsyncCommBucket.wait(OP_NAME)
    assert work.is_completed()
    with pytest.raises(KeyError):
        AsyncCommBucket.get(OP_NAME)


@rerun_if_address_is_in_use()
def test_is_all_completed_in_async_bucket():
    init_distributed(tp=2, dp=1, pp=1)(_test_wait_async_op_to_bucket)()


def _test_wait_async_op_to_bucket(parallel_context: ParallelContext):
    OP_NAME = "test"
    work = MockWork()

    AsyncCommBucket.add(OP_NAME, work)
    assert AsyncCommBucket.is_all_completed() is False

    AsyncCommBucket.wait(OP_NAME)
    assert AsyncCommBucket.is_all_completed() is True


@rerun_if_address_is_in_use()
def test_clear_ops_in_async_bucket():
    init_distributed(tp=2, dp=1, pp=1)(_test_clear_ops_in_async_bucket)()


def _test_clear_ops_in_async_bucket(parallel_context: ParallelContext):
    tensor1 = torch.randn(1, device="cuda")
    tensor2 = torch.randn(1, device="cuda")
    tensor3 = torch.randn(1, device="cuda")

    AsyncCommBucket.add("test1", dist.all_reduce(tensor1, async_op=True))
    AsyncCommBucket.add("test2", dist.all_reduce(tensor2, async_op=True))
    AsyncCommBucket.add("test3", dist.all_reduce(tensor3, async_op=True))

    assert AsyncCommBucket.is_all_completed() is False

    AsyncCommBucket.clear_all()
    assert AsyncCommBucket.is_all_completed() is True
    with pytest.raises(KeyError):
        AsyncCommBucket.get("test1")


@rerun_if_address_is_in_use()
def test_wait_comm():
    init_distributed(tp=2, dp=1, pp=1)(_test_wait_comm)()


def _test_wait_comm(parallel_context: ParallelContext):
    tensor = torch.randn(1, device="cuda", requires_grad=True)
    OP_NAME = "test"

    comm_stream = torch.cuda.Stream()

    with torch.cuda.stream(comm_stream):
        work = MockWork()
        AsyncCommBucket.add(OP_NAME, work)

    output = WaitComm.apply(tensor, OP_NAME, comm_stream)
    assert work.is_completed() is False

    # NOTE: we test that it waits for the async op to complete
    # automatically in autograd
    (output + 1).backward()
    assert work.is_completed()
