import math
from typing import Union

import pytest
import torch
from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron.config import ModelArgs, RandomInit, SpectralMupInit
from nanotron.parallel import ParallelContext
from nanotron.scaling.parametrization import ParametrizationMethod

from tests.helpers.llama_helper import TINY_LLAMA_CONFIG, create_llama_from_config, get_llama_training_config


@pytest.mark.parametrize("tp,dp,pp", [(2, 1, 1)])
@pytest.mark.parametrize("parametrization_method", [ParametrizationMethod.SPECTRAL_MUP])
@pytest.mark.skip
@rerun_if_address_is_in_use()
def test_parametrization(tp: int, dp: int, pp: int, parametrization_method: ParametrizationMethod):
    if parametrization_method == ParametrizationMethod.STANDARD:
        init_method = RandomInit(std=1.0)
    elif parametrization_method == ParametrizationMethod.SPECTRAL_MUP:
        init_method = SpectralMupInit(use_mup=True)

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_parametrization)(
        init_method=init_method,
        parametrization_method=parametrization_method,
    )


def _test_parametrization(
    parallel_context: ParallelContext,
    init_method: Union[RandomInit, SpectralMupInit],
    parametrization_method: ParametrizationMethod,
):
    def spectral_std(fan_in: int, fan_out: int):
        return torch.tensor((1.0 / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in)))

    model_args = ModelArgs(init_method=init_method, model_config=TINY_LLAMA_CONFIG)
    config = get_llama_training_config(model_args)

    llama = create_llama_from_config(
        model_config=TINY_LLAMA_CONFIG,
        device=torch.device("cuda"),
        parallel_context=parallel_context,
    )
    llama.init_model_randomly(config=config, init_method=parametrization_method)

    hidden_size = TINY_LLAMA_CONFIG.hidden_size
    interdimte_size = TINY_LLAMA_CONFIG.intermediate_size

    o_proj_infeatures = llama.model.decoder[0].pp_block.attn.o_proj.in_features * parallel_context.tensor_parallel_size
    NAME_TO_EXPECTED_STD = {
        "input_layernorm": torch.tensor(0.0),
        "post_attention_layernorm": torch.tensor(0.0),
        "final_layer_norm": torch.tensor(0.0),
        "token_embedding": torch.tensor(1.0),
        # "lm_head": torch.tensor(1.0),
        "qkv_proj": spectral_std(fan_in=hidden_size, fan_out=interdimte_size),
        "o_proj": spectral_std(fan_in=o_proj_infeatures, fan_out=hidden_size),
        "gate_up_proj": spectral_std(fan_in=hidden_size, fan_out=interdimte_size),
        "down_proj": spectral_std(fan_in=interdimte_size, fan_out=hidden_size),
    }

    def find_expected_std(param_name):
        for name in NAME_TO_EXPECTED_STD:
            if name in param_name:
                return NAME_TO_EXPECTED_STD[name]

    for name, param in llama.model.named_parameters():
        if "lm_head" in name:
            continue

        expected_std = find_expected_std(name)
        assert expected_std is not None, f"Could not find expected std for {name}"
        assert torch.allclose(
            param.std().float(), expected_std, atol=0.05
        ), f"name: {name}, expected: {expected_std}, actual: {param.std()}"
