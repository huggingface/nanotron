import contextlib

import pytest
import torch
from helpers.exception import assert_fail_with
from helpers.utils import available_gpus, init_distributed, rerun_if_address_is_in_use
from nanotron import distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.p2p import P2P


@pytest.mark.skipif(available_gpus() < 2, reason="Testing test_ddp_with_afab requires at least 2 gpus")
@pytest.mark.parametrize("send_contiguous", [True, False])
@pytest.mark.parametrize("full", [True, False])
@rerun_if_address_is_in_use()
def test_check_send_recv_tensor(send_contiguous: bool, full: bool):
    init_distributed(tp=1, dp=1, pp=2)(_test_check_send_recv_tensor)(send_contiguous=send_contiguous, full=full)


def _test_check_send_recv_tensor(parallel_context: ParallelContext, send_contiguous: bool, full: bool):
    p2p = P2P(pg=parallel_context.pp_pg, device=torch.device("cuda"))
    if dist.get_rank(p2p.pg) == 0:
        tensor_to_send = torch.randn(3, 5, dtype=torch.float, device=torch.device("cuda"))
        if send_contiguous is True:
            assert tensor_to_send.is_contiguous()
        else:
            tensor_to_send = tensor_to_send.transpose(0, 1)
            assert not tensor_to_send.is_contiguous()

        # `full` defines if we take a non trivial slice of the tensor
        if full is False:
            tensor_to_send = tensor_to_send[1:3]

    if send_contiguous is False and full is False:
        # This is supposed to return a ValueError mentioning that you should have sent a smaller model by running `contiguous` before.
        send_first_context = assert_fail_with(
            AssertionError,
            error_msg="Expect storage_size to be smaller than tensor size. It might not be true, when you use slicing for example though. We probably don't want to support it in our P2P system",
        )
        fail_at_first_send = True
    else:
        send_first_context = contextlib.nullcontext()
        fail_at_first_send = False

    # Send tensor back and forth through p2p protocol and check that we get the same thing.
    if dist.get_rank(p2p.pg) == 0:
        with send_first_context:
            handles = p2p.isend_tensors([tensor_to_send], to_rank=1)
        if fail_at_first_send is True:
            # We early return if we caught an error
            return
        for handle in handles:
            handle.wait()
        tensor_travelled_back_and_forth = p2p.recv_tensors(1, from_rank=1)[0]
        torch.testing.assert_close(tensor_to_send, tensor_travelled_back_and_forth, atol=0, rtol=0)
    elif dist.get_rank(p2p.pg) == 1:
        #  Instead of letting first rank hang since sending won't be possible, we early return
        tensors, handles = p2p.irecv_tensors(1, from_rank=0)
        if fail_at_first_send is True:
            return
        for handle in handles:
            handle.wait()
        tensor_to_recv = tensors[0]
        p2p.send_tensors([tensor_to_recv], to_rank=0)
    else:
        raise ValueError()

    if full is False and send_contiguous is True:
        # We can actually check that we haven't sent the entire storage as storage not accessed by the tensor are not sent
        if dist.get_rank(p2p.pg) == 0:
            # Check that the first element in the storages don't correspond (because they are not support to be communicated when the tensor is not full).

            print(tensor_to_send.untyped_storage()[:4], tensor_travelled_back_and_forth.untyped_storage()[:4])
            print(tensor_to_send.as_strided(size=(1,), stride=(1,), storage_offset=0))
            print(tensor_travelled_back_and_forth.as_strided(size=(1,), stride=(1,), storage_offset=0))
            assert not torch.allclose(
                tensor_to_send.as_strided(size=(1,), stride=(1,), storage_offset=0),
                tensor_travelled_back_and_forth.as_strided(size=(1,), stride=(1,), storage_offset=0),
            )

    parallel_context.destroy()
