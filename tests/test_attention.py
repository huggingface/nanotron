import pytest
import torch
from helpers.utils import available_gpus, init_distributed, rerun_if_address_is_in_use
from nanotron.models.attention import InfiniAttention
from nanotron.parallel import ParallelContext


@pytest.mark.parametrize("tp,dp,pp", [pytest.param(i, 1, 1) for i in range(1, min(4, available_gpus()) + 1)])
@rerun_if_address_is_in_use()
def test_infini_attention(tp: int, dp: int, pp: int):
    SEQ_LEN = 10
    BATCH_SIZE = 5
    HIDDEN_SIZE = 16

    hidden_states = torch.randn(SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE)
    sequence_mask = torch.randint(0, 2, (BATCH_SIZE, SEQ_LEN)).bool()

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_infini_attention)(hidden_states, sequence_mask)


def _test_infini_attention(
    parallel_context: ParallelContext, hidden_states: torch.Tensor, sequence_mask: torch.Tensor
):
    attn_output = InfiniAttention()

    assert attn_output.shape == hidden_states.shape

    parallel_context.destroy()
