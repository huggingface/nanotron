import pytest
import torch
from nanotron.constants import CHECKPOINT_VERSION
from nanotron.fp8.constants import FP8_DTYPES
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.tensor import FP8Tensor
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.sharded_parameters import SplitConfig, create_sharded_parameter_from_config
from nanotron.serialize.metadata import TensorMetadata
from nanotron.testing.parallel import init_distributed, rerun_if_address_is_in_use
from torch import nn


def create_sharded_fp8_parameter(param: nn.Parameter, parallel_context: ParallelContext):
    split_config = SplitConfig(
        split_dim=0,
        contiguous_chunks=(8, 8),
    )
    param = create_sharded_parameter_from_config(parameter=param, pg=parallel_context.tp_pg, split_config=split_config)
    return param


@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
def test_create_fp8_parameter(dtype):
    tensor = torch.randn(16, 16, device="cuda", dtype=torch.float32)

    fp8_parameter = FP8Parameter(tensor, dtype)

    assert isinstance(fp8_parameter.data, FP8Tensor)
    assert fp8_parameter.requires_grad is True
    assert fp8_parameter.grad is None
    assert fp8_parameter.dtype in FP8_DTYPES

    assert isinstance(fp8_parameter.fp8_meta, FP8Meta)
    assert isinstance(fp8_parameter.data.fp8_meta, FP8Meta)
    assert fp8_parameter.data.fp8_meta is fp8_parameter.fp8_meta


def test_fp8_parameter_grad_metadata():
    GRAD_META = ["input_grad", "weight_grad", "output_grad"]
    tensor = torch.randn(16, 16, device="cuda", dtype=torch.float32)
    fp8_parameter = FP8Parameter(tensor, DTypes.FP8E4M3)

    assert all(hasattr(fp8_parameter.fp8_grad_meta, attr) for attr in GRAD_META)
    assert all(isinstance(getattr(fp8_parameter.fp8_grad_meta, attr), FP8Meta) for attr in GRAD_META)


@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
@pytest.mark.parametrize("grad_dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
def test_setting_fp8_gradient_to_fp8_parameter(dtype, grad_dtype):
    fp8_parameter = FP8Parameter(torch.randn(16, 16, device="cuda"), dtype)
    fp8_grad = FP8Tensor(torch.randn(16, 16, device="cuda"), dtype=grad_dtype)

    fp8_parameter.grad = fp8_grad

    assert torch.equal(fp8_parameter.grad, fp8_parameter.data.grad)
    assert id(fp8_parameter.grad) == id(fp8_parameter.data.grad)
    assert fp8_parameter.grad.data_ptr() == fp8_parameter.data.grad.data_ptr()


@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
def test_fp8_parameter_storage_memory(dtype):
    data = torch.randn(16, 16, device="cuda", dtype=torch.float32)
    fp8_parameter = FP8Parameter(data, dtype)

    assert id(fp8_parameter.data) != id(data)
    assert fp8_parameter.data_ptr() == data.data_ptr()
    assert fp8_parameter.data.data_ptr() != data.data_ptr()


@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
def test_set_data_in_fp8_parameter(dtype):
    data = torch.randn(16, 16, device="cuda", dtype=torch.float32)
    fp8_parameter = FP8Parameter(data, dtype)

    new_data = torch.randn(16, 16, device="cuda", dtype=torch.float32)
    new_fp8_data = FP8Tensor(new_data, dtype=dtype)

    fp8_parameter.data = new_fp8_data

    assert fp8_parameter.data is new_fp8_data
    assert torch.equal(fp8_parameter.data, new_fp8_data)
    assert fp8_parameter.data.data_ptr() == new_fp8_data.data_ptr()

    assert fp8_parameter.fp8_meta is new_fp8_data.fp8_meta


@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
def test_set_gradient_in_fp8_parameter(dtype):
    data = torch.randn(16, 16, device="cuda", dtype=torch.float32)
    fp8_parameter = FP8Parameter(data, dtype)

    grad = torch.randn(16, 16, device="cuda", dtype=torch.float32)
    fp8_grad = FP8Tensor(grad, dtype=dtype)

    fp8_parameter.grad = fp8_grad

    assert fp8_parameter.grad is fp8_grad
    assert torch.equal(fp8_parameter.grad, fp8_grad)
    assert fp8_parameter.grad.data_ptr() == fp8_grad.data_ptr()

    assert fp8_parameter.data.grad is fp8_parameter.grad


@pytest.mark.parametrize("dtype", [DTypes.FP8E4M3, DTypes.FP8E5M2])
@rerun_if_address_is_in_use()
def test_create_sharded_fp8_parameter(dtype):
    init_distributed(tp=2, dp=1, pp=1)(_test_create_sharded_fp8_parameter)(dtype=dtype)


def _test_create_sharded_fp8_parameter(parallel_context: ParallelContext, dtype: DTypes):
    data = torch.randn(16, 64, device="cuda")
    param = FP8Parameter(data, dtype)

    param = create_sharded_fp8_parameter(param, parallel_context)
    sharded_info = param.get_sharded_info()

    assert isinstance(param, NanotronParameter)
    assert isinstance(param.data, FP8Tensor)
    assert isinstance(param.data.fp8_meta, FP8Meta)

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


# TODO(xrsrke): add test for preventing torch autograd do the backward pass
# on a FP8Parameter

# TODO(xrsrke): test CPU parameter


# TODO(xrsrke): test convert model to FP8
# include the FP8's NanotronParameter's dtype and requires_grad

# TODO(xrsrke): test set FP8 gradients to FP8 NanotronParameter
