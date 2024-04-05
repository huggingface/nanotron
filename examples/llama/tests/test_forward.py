# ruff: noqa: E402
import pytest
import torch
from nanotron.config import LlamaConfig as NanotronLlamaConfig
from nanotron.models.base import init_on_device_and_dtype
from transformers import LlamaForCausalLM
from utils import set_system_path

from examples.llama.convert_nanotron_to_hf import (
    convert_nanotron_to_hf,
    hf_config_from_nanotron_config,
    load_nanotron_model,
)

set_system_path()
from tests.helpers.utils import init_distributed

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


BATCH_SIZE = 3
SEQUENCE_LENGTH = 5


def create_nanotron_model():
    model = load_nanotron_model(
        CONFIG,
        torch.device("cpu"),
        torch.bfloat16,
    )
    return model


def create_hf_model():
    model_config_hf = hf_config_from_nanotron_config(CONFIG)
    with init_on_device_and_dtype(torch.device("cuda"), torch.bfloat16):
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


def test_attention_layers(dummy_inputs):
    init_distributed(tp=1, dp=1, pp=1)(_test_attention_layers)(dummy_inputs=dummy_inputs)


def _test_attention_layers(parallel_context, dummy_inputs):
    nanotron_model = create_nanotron_model()
    hf_model = create_hf_model()
    updated_hf_model = convert_nanotron_to_hf(nanotron_model, hf_model, CONFIG)
    nanotron_attention = get_nanotron_attention(nanotron_model)
    hf_attention = get_hf_attention(updated_hf_model)
    x_nanotron = dummy_inputs.permute(1, 0, 2)
    x_hf = dummy_inputs
    mask = torch.repeat_interleave(torch.ones_like(x_hf[..., 0])[..., None], SEQUENCE_LENGTH, dim=-1)
    # llama.py @ L. 391
    position_ids = torch.cumsum(mask[..., 0], dim=-1, dtype=torch.int32) - 1
    y_nanotron = nanotron_attention.to(device="cuda").forward(
        x_nanotron.cuda().bfloat16(), mask[..., 0].cuda().bfloat16()
    )["hidden_states"]
    y_hf = hf_attention(
        x_hf.cuda().bfloat16(),
        attention_mask=mask[:, None].cuda().bfloat16(),
        position_ids=position_ids.cuda().bfloat16(),
    )[0]
    assert y_hf.permute(1, 0, 2).shape == y_nanotron.shape
    assert torch.allclose(y_hf, y_nanotron.permute(1, 0, 2))
