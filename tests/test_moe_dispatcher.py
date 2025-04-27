import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from helpers.utils import (
    rerun_if_address_is_in_use,
)
from nanotron.distributed import initialize_torch_distributed
from nanotron.nn.moe import AllToAllDispatcher
from nanotron.utils import find_free_port


def setup_dist_env(rank, world_size, port):
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    # NOTE: since we do unit tests in a
    # single node => this is fine!
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def _test_all_to_all_dispatcher(rank, world_size, port, inputs, routing_indices, expected_outputs, num_experts):
    setup_dist_env(rank, world_size, port)
    initialize_torch_distributed()

    ep_pg = dist.new_group(ranks=list(range(world_size)))
    ep_rank = dist.get_rank(ep_pg)

    # NOTE: each ep rank holds a chunk of the inputs
    input = torch.chunk(inputs, world_size, dim=0)[ep_rank].contiguous().cuda()
    expected_output = expected_outputs[ep_rank].contiguous().cuda()
    routing_indices = torch.chunk(routing_indices, world_size, dim=0)[ep_rank].contiguous().cuda()
    num_local_experts = num_experts // world_size

    # Create dispatcher
    dispatcher = AllToAllDispatcher(num_local_experts=num_local_experts, num_experts=num_experts, ep_pg=ep_pg)

    # Test permute/unpermute round trip
    # (dispatched_inputs,
    #  inverse_permute_mapping,
    # #  num_tokens_per_expert) = dispatcher.permute(inputs, routing_indices)
    dispatched_input = dispatcher.permute(input, routing_indices)

    assert torch.allclose(dispatched_input, expected_output)

    dist.destroy_process_group()


@rerun_if_address_is_in_use()
def test_all_to_all_dispatcher():
    world_size = 2
    port = find_free_port()
    BS = 1
    SEQ_LEN = 8
    HIDDEN_SIZE = 4
    NUM_EXPERTS = 4

    inputs = torch.arange(BS * SEQ_LEN, dtype=torch.bfloat16).unsqueeze(-1).expand(-1, HIDDEN_SIZE)
    routing_indices = torch.tensor([[2], [3], [1], [3], [1], [0], [2], [3]], dtype=torch.int32)

    # expected_outputs = [
    #     torch.tensor([2, 5, 4], dtype=torch.bfloat16).unsqueeze(-1).expand(-1, HIDDEN_SIZE),
    #     torch.tensor([0, 1, 3, 6, 7], dtype=torch.bfloat16).unsqueeze(-1).expand(-1, HIDDEN_SIZE),
    # ] # all-to-all but without permutation

    expected_outputs = [
        torch.tensor([5, 2, 4], dtype=torch.bfloat16).unsqueeze(-1).expand(-1, HIDDEN_SIZE),
        torch.tensor([0, 6, 1, 3, 7], dtype=torch.bfloat16).unsqueeze(-1).expand(-1, HIDDEN_SIZE),
    ]  # all-to-all but without permutation

    mp.spawn(
        _test_all_to_all_dispatcher,
        args=(world_size, port, inputs, routing_indices, expected_outputs, NUM_EXPERTS),
        nprocs=world_size,
    )


if __name__ == "__main__":
    test_all_to_all_dispatcher()
