import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from helpers.utils import (
    rerun_if_address_is_in_use,
)
from nanotron.distributed import initialize_torch_distributed
from nanotron.nn.moe import AllToAllDispatcher
from nanotron.utils import find_free_port

HIDDEN_SIZE = 4


def setup_dist_env(rank, world_size, port):
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    # NOTE: since we do unit tests in a
    # single node => this is fine!
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def _test_all_to_all_dispatcher(
    rank,
    world_size,
    port,
    inputs,
    routing_indices,
    expected_permuted_outputs,
    expected_num_local_dispatched_tokens_per_expert,
    num_experts,
):
    setup_dist_env(rank, world_size, port)
    initialize_torch_distributed()

    ep_pg = dist.new_group(ranks=list(range(world_size)))
    ep_rank = dist.get_rank(ep_pg)
    expected_num_local_dispatched_tokens_per_expert = expected_num_local_dispatched_tokens_per_expert[ep_rank].cuda()

    # NOTE: each ep rank holds a chunk of the inputs
    input = torch.chunk(inputs, world_size, dim=0)[ep_rank].contiguous().cuda()
    expected_output = expected_permuted_outputs[ep_rank].contiguous().cuda()
    routing_indices = torch.chunk(routing_indices, world_size, dim=0)[ep_rank].contiguous().cuda()
    num_local_experts = num_experts // world_size

    dispatcher = AllToAllDispatcher(num_local_experts=num_local_experts, num_experts=num_experts, ep_pg=ep_pg)

    (
        dispatched_input,
        inverse_permute_mapping,
        inverse_expert_sorting_index,
        num_local_dispatched_tokens_per_expert,
    ) = dispatcher.permute(input, routing_indices, {})

    assert torch.allclose(dispatched_input, expected_output)
    assert torch.equal(expected_num_local_dispatched_tokens_per_expert, num_local_dispatched_tokens_per_expert)

    # NOTE: assume topk=1
    routing_weights = torch.ones_like(inverse_permute_mapping).unsqueeze(-1)
    undispatched_input = dispatcher.unpermute(
        dispatched_input, inverse_permute_mapping, routing_weights, inverse_expert_sorting_index
    )

    list_undispatched_inputs = [torch.empty_like(undispatched_input) for _ in range(world_size)]
    dist.all_gather(list_undispatched_inputs, undispatched_input)

    assert torch.allclose(undispatched_input, input)

    dist.destroy_process_group()


@rerun_if_address_is_in_use()
@pytest.mark.parametrize(
    "routing_indices, expected_permuted_outputs, expected_num_local_dispatched_tokens_per_expert",
    [
        # torch.tensor([2, 3, 1, 3, 1, 0, 2, 3], dtype=torch.int32), # top-k=1
        [
            torch.tensor([[2], [3], [1], [3], [1], [0], [2], [3]], dtype=torch.int32),
            [
                torch.tensor([5, 2, 4], dtype=torch.bfloat16).unsqueeze(-1).expand(-1, HIDDEN_SIZE),
                torch.tensor([0, 6, 1, 3, 7], dtype=torch.bfloat16).unsqueeze(-1).expand(-1, HIDDEN_SIZE),
            ],
            torch.tensor([[1, 2], [2, 3]], dtype=torch.bfloat16),
        ],  # top-k=1
        [
            torch.tensor([[2, 1], [3, 0], [1, 2], [3, 1], [1, 2], [0, 1], [2, 1], [1, 2]], dtype=torch.int32),
            [
                # NOTE: this isn't include expert sorting index
                # torch.tensor([1, 0, 2, 3, 5, 4, 5, 6, 7], dtype=torch.bfloat16).unsqueeze(-1).expand(-1, HIDDEN_SIZE),
                # torch.tensor([0, 2, 1, 3, 4, 6, 7], dtype=torch.bfloat16).unsqueeze(-1).expand(-1, HIDDEN_SIZE),
                torch.tensor([1, 5, 0, 2, 3, 4, 5, 6, 7], dtype=torch.bfloat16).unsqueeze(-1).expand(-1, HIDDEN_SIZE),
                torch.tensor([0, 2, 4, 6, 7, 1, 3], dtype=torch.bfloat16).unsqueeze(-1).expand(-1, HIDDEN_SIZE),
            ],  # top-k=2
            torch.tensor([[2, 7], [5, 2]], dtype=torch.bfloat16),
        ],
    ],
)
def test_all_to_all_dispatcher(
    routing_indices, expected_permuted_outputs, expected_num_local_dispatched_tokens_per_expert
):
    port = find_free_port()
    WORLD_SIZE = 2
    BS = 1
    SEQ_LEN = 8
    # HIDDEN_SIZE = 4
    NUM_EXPERTS = 4

    # NOTE: input.shape = [bs*seq_len, hidden_size]
    # routing_indices.shape = [bs*seq_len]
    # routing_weights.shape = [bs*seq_len, 1]
    inputs = torch.arange(BS * SEQ_LEN, dtype=torch.bfloat16).unsqueeze(-1).expand(-1, HIDDEN_SIZE)
    # NOTE: support top-k routing
    # routing_indices = torch.tensor([2, 3, 1, 3, 1, 0, 2, 3], dtype=torch.int32)
    # expected_num_local_dispatched_tokens_per_expert = torch.tensor([[1, 2], [2, 3]], dtype=torch.bfloat16)

    # expected_permuted_outputs = [
    #     torch.tensor([5, 2, 4], dtype=torch.bfloat16).unsqueeze(-1).expand(-1, HIDDEN_SIZE),
    #     torch.tensor([0, 6, 1, 3, 7], dtype=torch.bfloat16).unsqueeze(-1).expand(-1, HIDDEN_SIZE),
    # ]  # all-to-all but without permutation

    mp.spawn(
        _test_all_to_all_dispatcher,
        args=(
            WORLD_SIZE,
            port,
            inputs,
            routing_indices,
            expected_permuted_outputs,
            expected_num_local_dispatched_tokens_per_expert,
            NUM_EXPERTS,
        ),
        nprocs=WORLD_SIZE,
    )


if __name__ == "__main__":

    test_all_to_all_dispatcher(
        # routing_indices=torch.tensor([[2], [3], [1], [3], [1], [0], [2], [3]], dtype=torch.int32)
        routing_indices=torch.tensor(
            [[2, 1], [3, 0], [1, 2], [3, 1], [1, 2], [0, 1], [2, 1], [1, 2]], dtype=torch.int32
        ),
        expected_permuted_outputs=[
            torch.tensor([1, 5, 0, 2, 3, 4, 5, 6, 7], dtype=torch.bfloat16).unsqueeze(-1).expand(-1, HIDDEN_SIZE),
            torch.tensor([0, 2, 4, 6, 7, 1, 3], dtype=torch.bfloat16).unsqueeze(-1).expand(-1, HIDDEN_SIZE),
        ],
        expected_num_local_dispatched_tokens_per_expert=torch.tensor([[2, 7], [5, 2]], dtype=torch.bfloat16),
    )
