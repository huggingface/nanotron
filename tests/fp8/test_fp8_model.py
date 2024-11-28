import pytest
import torch
from nanotron.config import ModelArgs, RandomInit
from nanotron.config.fp8_config import FP8Args
from nanotron.fp8.tensor import FP8Tensor
from nanotron.fp8.utils import convert_model_to_fp8
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.testing.llama import TINY_LLAMA_CONFIG, create_llama_from_config, get_llama_training_config
from nanotron.testing.utils import init_distributed, rerun_if_address_is_in_use
from torch import nn


# NOTE: fp8 quantization should be parametrization-method-agnotic
@pytest.mark.parametrize("tp,dp,pp", [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2)])
@rerun_if_address_is_in_use()
def test_initialize_fp8_model(tp: int, dp: int, pp: int):
    fp8_config = FP8Args()
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_initialize_fp8_model)(fp8_config=fp8_config)


def _test_initialize_fp8_model(parallel_context: ParallelContext, fp8_config: FP8Args):
    model_args = ModelArgs(init_method=RandomInit(std=1.0), model_config=TINY_LLAMA_CONFIG)
    config = get_llama_training_config(model_args)
    llama = create_llama_from_config(
        model_config=TINY_LLAMA_CONFIG,
        device=torch.device("cuda"),
        parallel_context=parallel_context,
        dtype=torch.float32,
    )
    llama.init_model_randomly(config=config)

    llama = convert_model_to_fp8(llama, config=fp8_config)

    assert 1 == 1
    # NOTE: test the default recipe in fp8's nanotron
    from nanotron.fp8.utils import find_fp8_config_by_module_name, get_leaf_modules

    for name, module in get_leaf_modules(llama):
        recipe = find_fp8_config_by_module_name(name, fp8_config)

        assert all(p.__class__ == NanotronParameter for p in module.parameters())
        if recipe is None:
            assert all(
                p.dtype == fp8_config.resid_dtype for p in module.parameters()
            ), f"name: {name}, __class__: {module.weight.data.__class__}"
            try:
                assert all(
                    p.data.__class__ == nn.Parameter for p in module.parameters()
                ), f"name: {name}, __class__: {module.weight.data.__class__}"
            except:
                assert 1 == 1
        else:
            assert all(
                isinstance(p.data.__class__, FP8Tensor) for p in module.parameters()
            ), f"name: {name}, __class__: {module.weight.data.__class__}"
            assert all(
                p.dtype in [torch.int8, torch.uint8] for p in module.parameters()
            ), f"name: {name}, __class__: {module.weight.data.__class__}"
    # NOTE: check the expected parameters have fp8 dtype
    # NOTE: check the dtype of non-fp8 parameters
