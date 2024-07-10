import pytest
import torch
from helpers.exception import assert_fail_with
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.tensor import FP8Tensor, convert_tensor_from_fp8
from nanotron.models.base import DTypeInvariantTensor, init_on_device_and_dtype
from nanotron.parallel.parameters import NanotronParameter
from torch import nn


def test_nanotron_parameter_does_not_override_some_parameter_variable():
    param = nn.Parameter(torch.empty(3))
    assert not hasattr(param, NanotronParameter.NANOTRON_PARAMETER_METADATA_ATTRIBUTE_NAME)


def test_create_nanotron_parameter():
    data = torch.randn(3)
    param = nn.Parameter(data)
    param = NanotronParameter(param)

    assert isinstance(param.data, torch.Tensor)
    assert torch.allclose(param, data)
    assert torch.allclose(param.data, data)


@pytest.mark.parametrize("is_fp8", [False, True])
def test_gradients_flow_to_nanotron_parameter(is_fp8):
    data = torch.randn(3, device="cuda")
    data = FP8Tensor(data, dtype=DTypes.FP8E4M3) if is_fp8 else data
    param = nn.Parameter(data)
    param = NanotronParameter(param)

    x = torch.randn(3, device="cuda")
    data = param.data
    fp32_data = convert_tensor_from_fp8(data, data.fp8_meta, torch.float32) if is_fp8 else data
    output = torch.sum(fp32_data * x)
    output.backward()

    assert torch.allclose(param.grad, x)
    # assert torch.allclose(param.data.grad, x)


def test_set_new_value_for_nanotron_parameter():
    param = nn.Parameter(torch.randn(3))
    param = NanotronParameter(param)

    before_data = param.data.clone()
    new_data = torch.randn_like(param.data)
    param.data = new_data.clone()

    assert isinstance(param, NanotronParameter)
    assert not torch.allclose(before_data, param.data)
    assert torch.allclose(param.data, new_data)


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
