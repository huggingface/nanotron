import torch
from helpers.exception import assert_fail_with
from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron.models.base import DTypeInvariantTensor, init_on_device_and_dtype

# from nanotron.testing.utils import TestContext
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.sharded_parameters import SplitConfig, create_sharded_parameter_from_config
from torch import nn


@rerun_if_address_is_in_use()
def test_random_hash_nanotron_parameter():
    init_distributed(tp=2, dp=1, pp=1)(_test_random_hash_nanotron_parameter)()


def _test_random_hash_nanotron_parameter(parallel_context: ParallelContext):
    param = torch.nn.Parameter(torch.randn(16, 64))
    split_config = SplitConfig(
        split_dim=0,
        contiguous_chunks=(8, 8),
    )
    param = create_sharded_parameter_from_config(parameter=param, pg=parallel_context.tp_pg, split_config=split_config)
    hash = getattr(param, NanotronParameter.NANOTRON_PARAMETER_HASH_ATTRIBUTE_NAME)

    assert type(hash) == str


def test_nanotron_parameter_does_not_override_some_parameter_variable():
    param = nn.Parameter(torch.empty(3))
    assert not hasattr(param, NanotronParameter.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME)


def test_uncastable_tensor():
    # Test that we can create an DTypeInvariantTensor
    x = DTypeInvariantTensor(torch.randn(3, 3))
    assert isinstance(x, torch.Tensor)
    assert isinstance(x, DTypeInvariantTensor)

    # Test that we cannot modify the type of an DTypeInvariantTensor
    with assert_fail_with(RuntimeError, error_msg="Cannot convert the type of an DTypeInvariantTensor to float"):
        x = x.float()

    with assert_fail_with(RuntimeError, error_msg="Cannot convert the type of an DTypeInvariantTensor to half"):
        x = x.half()

    with assert_fail_with(RuntimeError, error_msg="Cannot change the type of an DTypeInvariantTensor"):
        x = x.to(torch.float32)

    with assert_fail_with(RuntimeError, error_msg="Cannot change the type of an DTypeInvariantTensor"):
        x = x.to(dtype=torch.float32)

    # Test that we can modify the value of an DTypeInvariantTensor
    x[0, 0] = 1
    assert x[0, 0] == 1

    # Test that we can modify the device of an DTypeInvariantTensor
    x = x.to("cuda")
    assert x.device.type == "cuda"


def test_register_buffer_does_not_update_uncastable_tensor():
    old_device = torch.device("cuda")
    old_dtype = torch.float32
    new_device = torch.device("cpu")
    new_dtype = torch.bfloat16
    with init_on_device_and_dtype(device=new_device, dtype=new_dtype):
        module = torch.nn.Module()
        # Test that we can register an DTypeInvariantTensor as a buffer
        tensor = DTypeInvariantTensor(torch.randn(3, 4, dtype=old_dtype, device=old_device))
        module.register_buffer("buffer", tensor)

        # Test that we can modify the buffer
        module.buffer[0, 0] = 1
        assert module.buffer[0, 0] == 1

        # Test that device has been updated
        assert module.buffer.device.type == new_device.type

        # Test that dtype has not been modified
        assert module.buffer.dtype is old_dtype


@rerun_if_address_is_in_use()
def test_create_param_that_share_metadata():
    init_distributed(tp=2, dp=1, pp=1)(_test_create_param_that_share_metadata)()


def _test_create_param_that_share_metadata(parallel_context: ParallelContext):
    param = torch.nn.Parameter(torch.randn(16, 64))
    split_config = SplitConfig(
        split_dim=0,
        contiguous_chunks=(8, 8),
    )
    orig_param = create_sharded_parameter_from_config(
        parameter=param, pg=parallel_context.tp_pg, split_config=split_config
    )
    new_param = NanotronParameter.create_param_that_share_metadata(torch.randn(16, 64), param=orig_param)

    new_param_metadata = getattr(new_param, NanotronParameter.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME)
    orig_param_metadata = getattr(orig_param, NanotronParameter.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME)

    for (p1_k, p1_v), (p2_k, p2_v) in zip(new_param_metadata.items(), orig_param_metadata.items()):
        assert p1_k == p2_k
        assert p1_v == p2_v

    orig_hash = getattr(orig_param, NanotronParameter.NANOTRON_PARAMETER_HASH_ATTRIBUTE_NAME)
    new_hash = getattr(new_param, NanotronParameter.NANOTRON_PARAMETER_HASH_ATTRIBUTE_NAME)

    assert new_hash == orig_hash

    parallel_context.destroy()
