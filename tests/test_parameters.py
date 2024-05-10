import pytest
import torch
from helpers.context import TestContext
from helpers.utils import (
    init_distributed,
    rerun_if_address_is_in_use,
)
from nanotron.constants import CHECKPOINT_VERSION
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.parameter import FP8Parameter
from nanotron.parallel import ParallelContext
from nanotron.parallel.sharded_parameters import SplitConfig, create_sharded_parameter_from_config
from nanotron.serialize.metadata import TensorMetadata


@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
@rerun_if_address_is_in_use()
def test_create_sharded_fp8_parameter(dtype):
    test_context = TestContext()
    init_distributed(tp=2, dp=1, pp=1)(_test_create_sharded_fp8_parameter)(test_context=test_context, dtype=dtype)


def _test_create_sharded_fp8_parameter(parallel_context: ParallelContext, test_context: TestContext, dtype: DTypes):
    # param = torch.nn.Parameter(torch.randn(16, 64))
    data = torch.randn(16, 64)
    param = FP8Parameter(data, dtype)

    split_config = SplitConfig(
        split_dim=0,
        contiguous_chunks=(8, 8),
    )
    param = create_sharded_parameter_from_config(parameter=param, pg=parallel_context.tp_pg, split_config=split_config)
    sharded_info = param.get_sharded_info()
    metadata = TensorMetadata(
        version=CHECKPOINT_VERSION,
        local_global_slices_pairs=sharded_info.local_global_slices_pairs,
        unsharded_shape=sharded_info.unsharded_shape,
    )
    metadata_str_dict = metadata.to_str_dict()
    # Assert metadata_str_dict is Dict[str, str]
    assert isinstance(metadata_str_dict, dict)
    assert all(isinstance(key, str) for key in metadata_str_dict.keys())
    assert all(isinstance(value, str) for value in metadata_str_dict.values())

    metadata_from_str_dict = TensorMetadata.from_str_dict(metadata_str_dict)
    assert metadata == metadata_from_str_dict

    parallel_context.destroy()
