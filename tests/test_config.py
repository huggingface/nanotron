import pytest
from nanotron.config import ModelArgs, RandomInit, SpectralMupInit
from nanotron.config.models_config import LlamaConfig, MoEConfig, Qwen2Config, Starcoder2Config


@pytest.mark.parametrize(
    ("model_config", "expected_type"),
    [
        ({"is_llama_config": True}, LlamaConfig),
        ({"is_qwen2_config": True}, Qwen2Config),
        ({"is_starcoder2_config": True}, Starcoder2Config),
    ],
)
def test_model_args_restores_serialized_model_config(model_config, expected_type):
    model_args = ModelArgs(model_config=model_config, init_method=RandomInit(std=0.02))

    assert isinstance(model_args.model_config, expected_type)
    assert model_args.model_config._is_using_mup is False


def test_model_args_restores_nested_moe_config():
    model_args = ModelArgs(
        model_config={"is_qwen2_config": True, "moe_config": {"num_experts": 4, "top_k": 2}},
        init_method=RandomInit(std=0.02),
    )

    assert isinstance(model_args.model_config.moe_config, MoEConfig)
    assert model_args.model_config.moe_config.num_experts == 4


def test_model_args_sets_mup_on_restored_config():
    model_args = ModelArgs(
        model_config={"is_llama_config": True},
        init_method=SpectralMupInit(use_mup=True),
    )

    assert isinstance(model_args.model_config, LlamaConfig)
    assert model_args.model_config.is_using_mup is True
