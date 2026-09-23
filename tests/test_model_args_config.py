"""Tests for ModelArgs dict deserialization (regression for #198).

When a checkpoint is loaded from YAML, dacite passes model_config as a raw
dict.  ModelArgs.__post_init__ must convert it to the correct config object
before setting _is_using_mup, otherwise an AttributeError is raised.
"""

import dataclasses
import pytest
from nanotron.config.config import ModelArgs
from nanotron.config.models_config import LlamaConfig, Qwen2Config, Starcoder2Config
from nanotron.config import RandomInit


LLAMA_DICT = {
    "bos_token_id": 1, "eos_token_id": 2, "hidden_act": "silu",
    "hidden_size": 128, "initializer_range": 0.02, "intermediate_size": 512,
    "is_llama_config": True, "max_position_embeddings": 128,
    "num_attention_heads": 4, "num_hidden_layers": 2, "num_key_value_heads": 2,
    "pad_token_id": None, "pretraining_tp": 1, "rms_norm_eps": 1e-6,
    "rope_scaling": None, "tie_word_embeddings": False, "use_cache": True,
    "vocab_size": 256,
}

QWEN2_DICT = {
    "bos_token_id": 1, "eos_token_id": 2, "hidden_act": "silu",
    "hidden_size": 128, "initializer_range": 0.02, "intermediate_size": 512,
    "is_qwen2_config": True, "max_position_embeddings": 128,
    "num_attention_heads": 4, "num_hidden_layers": 2, "num_key_value_heads": 2,
    "pad_token_id": None, "pretraining_tp": 1, "rms_norm_eps": 1e-6,
    "rope_scaling": None, "tie_word_embeddings": False, "use_cache": True,
    "vocab_size": 256, "max_window_layers": 2, "use_sliding_window": False,
    "sliding_window": None, "rope_theta": 10000.0,
}


@pytest.mark.parametrize("config_dict,expected_cls", [
    (LLAMA_DICT, LlamaConfig),
    (QWEN2_DICT, Qwen2Config),
])
def test_model_args_dict_deserialization(config_dict, expected_cls):
    """ModelArgs must deserialize dict model_config without raising AttributeError (regression #198)."""
    args = ModelArgs(init_method=RandomInit(std=0.02), model_config=dict(config_dict))
    assert isinstance(args.model_config, expected_cls)
    assert hasattr(args.model_config, "_is_using_mup")
    assert args.model_config._is_using_mup is False


def test_model_args_unknown_dict_raises():
    """A dict with no recognised discriminator field should raise ValueError, not AttributeError."""
    with pytest.raises(ValueError, match="no recognised discriminator field"):
        ModelArgs(init_method=RandomInit(std=0.02), model_config={"hidden_size": 128})
