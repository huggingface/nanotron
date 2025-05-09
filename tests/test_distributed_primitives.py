import pytest
import torch
import torch.distributed as dist
from helpers.utils import (
    init_distributed,
    rerun_if_address_is_in_use,
)
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import all_to_all


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
    test_all_to_all(
        2,
        [
            torch.tensor([2, 0, 1, 3]).unsqueeze(-1).expand(-1, 4),
            torch.tensor([5, 4, 6, 7]).unsqueeze(-1).expand(-1, 4),
        ],
        [
            torch.tensor([2, 5, 4]).unsqueeze(-1).expand(-1, 4),
            torch.tensor([0, 1, 3, 6, 7]).unsqueeze(-1).expand(-1, 4),
        ],
        [[1, 3], [2, 2]],
        [[1, 2], [3, 2]],
    )
