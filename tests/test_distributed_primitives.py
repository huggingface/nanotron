import pytest
import torch
import torch.distributed as dist
from helpers.utils import (
    init_distributed,
    rerun_if_address_is_in_use,
)
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import all_to_all

# def _test_all_to_all(parallel_context: ParallelContext, input_split_sizes=None, output_split_sizes=None):
#     rank = dist.get_rank(parallel_context.tp_pg)
#     world_size = dist.get_world_size(parallel_context.tp_pg)

#     if input_split_sizes is None and output_split_sizes is None:
#         # Default case: uniform sizes
#         input = torch.arange(4, device="cuda") + rank * 4
#         output = all_to_all(input, group=parallel_context.tp_pg)
#         expected_output = torch.tensor([rank + i * world_size for i in range(world_size)], device="cuda")
#         assert torch.allclose(output, expected_output)
#     else:
#         # Custom split sizes case
#         # input_size = input_split_sizes[rank]
#         # input = torch.arange(input_size, device="cuda") + sum(input_split_sizes[:rank])

#         input = torch.arange(8, device="cuda") + rank * 4
#         input_list = [torch.zeros_like(input) for _ in range(world_size)]
#         dist.all_gather(input_list, input)

#         output = all_to_all(
#             input,
#             output_split_sizes=output_split_sizes,
#             input_split_sizes=input_split_sizes,
#             group=parallel_context.tp_pg,
#         )

#         dist.barrier()
#         output_list = [torch.zeros_like(output) for _ in range(world_size)]
#         dist.all_gather(output_list, output)

#         assert isinstance(output, torch.Tensor)
#         assert 1 == 1

#         # # Calculate expected output
#         # expected_size = output_split_sizes[rank]
#         # offsets = [sum(input_split_sizes[:r]) for r in range(world_size)]

#         # expected_elements = []
#         # for i in range(world_size):
#         #     # Get the elements that rank i sends to current rank
#         #     start_idx = offsets[i] + sum(output_split_sizes[:rank])
#         #     end_idx = start_idx + output_split_sizes[rank]
#         #     expected_elements.extend(range(start_idx, end_idx))

#         # expected_output = torch.tensor(expected_elements, device="cuda")
#         # assert torch.allclose(output, expected_output)
#         # assert output.size(0) == output_split_sizes[rank]


def _test_all_to_all(
    parallel_context: ParallelContext, inputs, expected_outputs, input_split_sizes, output_split_sizes
):
    rank = dist.get_rank(parallel_context.tp_pg)

    input = inputs[rank].to("cuda")
    expected_output = expected_outputs[rank].to("cuda")
    input_split_sizes = input_split_sizes[rank]
    output_split_sizes = output_split_sizes[rank]

    output = all_to_all(
        input, group=parallel_context.tp_pg, input_split_sizes=input_split_sizes, output_split_sizes=output_split_sizes
    )
    assert torch.allclose(output, expected_output)


@pytest.mark.parametrize(
    "world_size, inputs, expected_outputs, input_split_sizes, output_split_sizes",
    [
        (
            4,
            # NOTE: range(4) is range(world_size)
            [torch.arange(4) + rank * 4 for rank in range(4)],
            [torch.tensor([rank + i * 4 for i in range(4)]) for rank in range(4)],
            [None, None, None, None],
            [None, None, None, None],
        ),  # Default case: uniform sizes
        (
            2,
            [
                torch.tensor([2, 0, 1, 3]).unsqueeze(-1).expand(-1, 4),
                torch.tensor([5, 4, 6, 7]).unsqueeze(-1).expand(-1, 4),
            ],
            # [
            #     torch.tensor([5, 2, 4]).unsqueeze(-1).expand(-1, 4),
            #     torch.tensor([0, 6, 1, 3, 7]).unsqueeze(-1).expand(-1, 4),
            # ], # my expected output
            [
                torch.tensor([2, 5, 4]).unsqueeze(-1).expand(-1, 4),
                torch.tensor([0, 1, 3, 6, 7]).unsqueeze(-1).expand(-1, 4),
            ],
            [[1, 3], [2, 2]],
            [[1, 2], [3, 2]],
        ),  # Custom split sizes
    ],
)
@rerun_if_address_is_in_use()
def test_all_to_all(world_size, inputs, expected_outputs, input_split_sizes, output_split_sizes):
    # WORLD_SIZE = 4
    # inputs = [torch.arange(4) + rank * 4 for rank in range(WORLD_SIZE)]
    # expected_outputs = [torch.tensor([rank + i * WORLD_SIZE for i in range(WORLD_SIZE)]) for rank in range(WORLD_SIZE)]

    init_distributed(
        tp=world_size,
        dp=1,
        pp=1,
        expert_parallel_size=1,
        expert_tensor_parallel_size=1,
        expert_data_parallel_size=1,
        enabled_moe=False,
    )(_test_all_to_all)(
        inputs=inputs,
        expected_outputs=expected_outputs,
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
    )


if __name__ == "__main__":
    # test_all_to_all(4,
    #     # NOTE: range(4) is range(world_size)
    #     [torch.arange(4) + rank * 4 for rank in range(4)],
    #     [torch.tensor([rank + i * 4 for i in range(4)]) for rank in range(4)],
    #     [None, None, None, None],
    #     [None, None, None, None],
    # )
    test_all_to_all(
        2,
        [
            torch.tensor([2, 0, 1, 3]).unsqueeze(-1).expand(-1, 4),
            torch.tensor([5, 4, 6, 7]).unsqueeze(-1).expand(-1, 4),
        ],
        # [
        #     torch.tensor([5, 2, 4]).unsqueeze(-1).expand(-1, 4),
        #     torch.tensor([0, 6, 1, 3, 7]).unsqueeze(-1).expand(-1, 4),
        # ], # my expected output
        [
            torch.tensor([2, 5, 4]).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0, 1, 3, 6, 7]).unsqueeze(-1).expand(-1, 4),
        ],  # my expected output
        [[1, 3], [2, 2]],
        [[1, 2], [3, 2]],
    )
