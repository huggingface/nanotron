import pytest
import torch
from llama.convert_nanotron_to_hf import convert_nanotron_to_hf, hf_config_from_nanotron_config, load_nanotron_model
from nanotron.config import LlamaConfig as NanotronLlamaConfig
from nanotron.models.base import init_on_device_and_dtype
from transformers import LlamaForCausalLM

CONFIG = NanotronLlamaConfig(
    {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "is_llama_config": True,
        "max_position_embeddings": 128,
        "num_attention_heads": 16,
        "num_hidden_layers": 16,
        "num_key_value_heads": 16,
        "pad_token_id": None,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 32000,
    }
)

DEVICE = torch.device("cuda")
DTYPE = getattr(torch, "bfloat16")

BATCH_SIZE = 3
SEQUENCE_LENGTH = 5


@pytest.fixture
def nanotron_model():
    model = load_nanotron_model(
        CONFIG,
        DEVICE,
        DTYPE,
    )
    return model


@pytest.fixture
def hf_model():
    model_config_hf = hf_config_from_nanotron_config(CONFIG)
    with init_on_device_and_dtype(DEVICE, DTYPE):
        hf_model = LlamaForCausalLM._from_config(model_config_hf)
    return hf_model


@pytest.fixture
def dummy_inputs():
    return torch.rand(BATCH_SIZE, SEQUENCE_LENGTH, CONFIG.hidden_size)


def get_nanotron_attention(nanotron_model):
    nanotron_first_decoder = nanotron_model.model.decoder[0].pp_block.attn
    return nanotron_first_decoder


def get_hf_attention(hf_model):
    hf_first_decoder = hf_model.model.layers[0].self_attn
    return hf_first_decoder


def test_attention_layers(nanotron_model, hf_model, dummy_inputs):
    updated_hf_model = convert_nanotron_to_hf(nanotron_model, hf_model)
    nanotron_attention = get_nanotron_attention(nanotron_model)
    hf_attention = get_hf_attention(updated_hf_model)
    x_nanotron = dummy_inputs
    x_hf = dummy_inputs.permute(1, 0, 2)
    mask = torch.ones_like(x_hf[..., 0])
    # llama.py @ L. 391
    position_ids = torch.cumsum(mask, dim=-1, dtype=torch.int32) - 1
    y_nanotron = nanotron_attention.forward(x_nanotron)["attention_state"]
    y_hf = hf_attention(x_hf, position_ids=position_ids)[0]
    assert torch.allclose(y_hf, y_nanotron)
