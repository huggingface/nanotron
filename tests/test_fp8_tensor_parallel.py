
import pytest
from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.linear import FP8LinearMeta
from nanotron.fp8.tensor import FP8Tensor
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import (
    FP8TensorParallelColumnLinear,
)


@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@rerun_if_address_is_in_use()
def test_fp8_column_linear_metadata(
    tp: int,
    dp: int,
    pp: int,
):
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_fp8_column_linear_metadata)()


def _test_fp8_column_linear_metadata(
    parallel_context: ParallelContext,
):
    # NOTE: divisible by 16 for TP
    in_features = 32
    out_features_per_tp_rank = 16

    out_features = parallel_context.tp_pg.size() * out_features_per_tp_rank

    column_linear = FP8TensorParallelColumnLinear(
        in_features=in_features,
        out_features=out_features,
        pg=parallel_context.tp_pg,
        mode=TensorParallelLinearMode.ALL_REDUCE,
        device="cuda",
        async_communication=False,
        bias=False,
    )

    assert isinstance(column_linear.weight, NanotronParameter)
    assert isinstance(column_linear.weight.data, FP8Tensor)
    assert isinstance(column_linear.accum_qtype, DTypes)
    assert isinstance(column_linear.metadatas, FP8LinearMeta)

    parallel_context.destroy()
