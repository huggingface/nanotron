import numpy as np
import pytest
from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron.data.clm_collator import DataCollatorForCLMWithPositionIds
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer


@pytest.mark.parametrize(
    "input_pp_rank,output_pp_rank",
    [
        (0, 0),  # current rank has both inputs and labels
        (1, 0),  # current rank only has labels
        (1, 2),  # current rank has no data
    ],
)
@rerun_if_address_is_in_use()
def test_clm_collator_with_position_ids_pp_ranks(input_pp_rank: int, output_pp_rank: int):
    init_distributed(tp=1, dp=1, pp=1)(_test_clm_collator_with_position_ids_pp_ranks)(
        input_pp_rank=input_pp_rank, output_pp_rank=output_pp_rank
    )


def _test_clm_collator_with_position_ids_pp_ranks(
    parallel_context: ParallelContext, input_pp_rank: int, output_pp_rank: int
):
    sequence_length = 8
    current_pp_rank = 0
    collator = DataCollatorForCLMWithPositionIds(
        sequence_length=sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        parallel_context=parallel_context,
    )

    if current_pp_rank in [input_pp_rank, output_pp_rank]:
        examples = [
            {"input_ids": np.arange(sequence_length + 1), "positions": np.array([0, 1, 2, 3, 0, 1, 2, 3, 4])},
            {"input_ids": np.arange(sequence_length + 1) + 100, "positions": np.arange(sequence_length + 1)},
        ]
    else:
        examples = [{}, {}]

    batch = collator(examples)

    assert set(batch.keys()) == {"input_ids", "position_ids", "label_ids", "label_mask"}

    for key in ["input_ids", "position_ids"]:
        if current_pp_rank == input_pp_rank:
            assert batch[key].shape == (2, sequence_length)
        else:
            assert batch[key] == TensorPointer(group_rank=input_pp_rank)

    for key in ["label_ids", "label_mask"]:
        if current_pp_rank == output_pp_rank:
            assert batch[key].shape == (2, sequence_length)
        else:
            assert batch[key] == TensorPointer(group_rank=output_pp_rank)

    if current_pp_rank == output_pp_rank:
        np.testing.assert_array_equal(batch["label_ids"][0], np.arange(1, sequence_length + 1))
        # The label that starts a new document (position 0) is masked
        np.testing.assert_array_equal(batch["label_mask"][0], [True, True, True, False, True, True, True, True])

    parallel_context.destroy()
