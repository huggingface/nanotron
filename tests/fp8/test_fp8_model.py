import pytest
import torch
from nanotron.fp8.tensor import FP8Tensor
from nanotron.fp8.utils import get_leaf_modules
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.nn import FP8TensorParallelColumnLinear, FP8TensorParallelRowLinear
from nanotron.testing.parallel import init_distributed, rerun_if_address_is_in_use
from nanotron.testing.utils import create_nanotron_model

MODULE_NAME_TO_FP8_MODULE = {
    "qkv_proj": FP8TensorParallelColumnLinear,
    "o_proj": FP8TensorParallelRowLinear,
    "gate_up_proj": FP8TensorParallelColumnLinear,
    "down_proj": FP8TensorParallelRowLinear,
    "lm_head": FP8TensorParallelColumnLinear,
}


@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@rerun_if_address_is_in_use()
def test_initialize_fp8_model(tp: int, dp: int, pp: int):
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_initialize_fp8_model)()


def _test_initialize_fp8_model(parallel_context: ParallelContext):
    model = create_nanotron_model(parallel_context, dtype=torch.int8)
    modules = get_leaf_modules(model)

    for module_name, module in modules:
        is_fp8_module = any(name in module_name for name in MODULE_NAME_TO_FP8_MODULE.keys())

        if is_fp8_module:
            fp8_module_name = [name for name in MODULE_NAME_TO_FP8_MODULE.keys() if name in module_name][0]
            assert module.__class__ == MODULE_NAME_TO_FP8_MODULE[fp8_module_name]
        else:
            assert any(module.__class__ == fp8_module for fp8_module in MODULE_NAME_TO_FP8_MODULE.values()) is False

    # expect the number of fp8 parameters and non-fp8 parameters
    # expect the logits to be in accumulation precisio


@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@rerun_if_address_is_in_use()
def test_forward_pass_of_fp8_model(tp: int, dp: int, pp: int):
    input_ids = torch.randint(0, 100, size=(16, 64))
    input_mask = torch.ones_like(input_ids)
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_forward_pass_of_fp8_model)(input_ids=input_ids, input_mask=input_mask)


def _test_forward_pass_of_fp8_model(
    parallel_context: ParallelContext, input_ids: torch.Tensor, input_mask: torch.Tensor
):
    input_ids = input_ids.to("cuda")
    input_mask = input_mask.to("cuda")
    nanotron_model = create_nanotron_model(parallel_context, dtype=torch.int8)

    logits = nanotron_model.model(input_ids, input_mask)

    # NOTE: the last layer in the model casts the logits to FP32
    assert logits.dtype == torch.float32


@pytest.mark.parametrize("tp,dp,pp", [[1, 1, 1], [2, 1, 1]])
@rerun_if_address_is_in_use()
def test_fp8_model_has_gradients(tp: int, dp: int, pp: int):
    input_ids = torch.randint(0, 100, size=(16, 64))
    input_mask = torch.ones_like(input_ids)
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_fp8_model_has_gradients)(input_ids=input_ids, input_mask=input_mask)


def _test_fp8_model_has_gradients(
    parallel_context: ParallelContext, input_ids: torch.Tensor, input_mask: torch.Tensor
):
    input_ids = input_ids.to("cuda")
    input_mask = input_mask.to("cuda")
    nanotron_model = create_nanotron_model(parallel_context, dtype=torch.int8)

    logits = nanotron_model.model(input_ids, input_mask)

    logits.sum().backward()

    modules = get_leaf_modules(nanotron_model)
    for module_name, module in modules:
        is_fp8_module = any(name in module_name for name in MODULE_NAME_TO_FP8_MODULE.keys())
        for param in module.parameters():
            if is_fp8_module:
                assert param.data._temp_grad.__class__ == FP8Tensor
            else:
                assert param.grad.__class__ == torch.Tensor
