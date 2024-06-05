# ruff: noqa: E402
import dataclasses
import json
from pathlib import Path

import pytest
import torch
from transformers import LlamaForCausalLM
from utils import set_system_path

set_system_path()

import nanotron
from nanotron.config import LlamaConfig as NanotronLlamaConfig
from nanotron.models.base import init_on_device_and_dtype
from nanotron.models.llama import LlamaForTraining
from nanotron.parallel import ParallelContext
from nanotron.trainer import mark_tied_parameters

from examples.llama.convert_hf_to_nanotron import convert_checkpoint_and_save as convert_hf_to_nt_and_save
from examples.llama.convert_hf_to_nanotron import convert_hf_to_nt
from examples.llama.convert_nanotron_to_hf import convert_checkpoint_and_save as convert_nt_to_hf_and_save
from examples.llama.convert_nanotron_to_hf import convert_nt_to_hf, get_hf_config
from examples.llama.convert_weights import load_nanotron_model, make_parallel_config
from tests.helpers.context import TestContext
from tests.helpers.utils import init_distributed

CONFIG = NanotronLlamaConfig(
    **{
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 512,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "is_llama_config": True,
        "max_position_embeddings": 128,
        "num_attention_heads": 8,
        "num_hidden_layers": 4,
        "num_key_value_heads": 4,
        "pad_token_id": None,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 4096,
    }
)


BATCH_SIZE = 3
SEQUENCE_LENGTH = 5
ATOL = 0.03


def create_nanotron_model(parallel_context: ParallelContext) -> LlamaForTraining:
    parallel_config = make_parallel_config(
        tp=parallel_context.tensor_parallel_size,
        dp=parallel_context.data_parallel_size,
        pp=parallel_context.pipeline_parallel_size,
    )
    nanotron_model = nanotron.models.build_model(
        model_builder=lambda: LlamaForTraining(
            config=CONFIG,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=None,
        ),
        parallel_context=parallel_context,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    mark_tied_parameters(model=nanotron_model, parallel_context=parallel_context)
    return nanotron_model


def create_huggingface_model() -> LlamaForCausalLM:
    config_hf = get_hf_config(CONFIG)
    with init_on_device_and_dtype(torch.device("cuda"), torch.bfloat16):
        model_hf = LlamaForCausalLM._from_config(config_hf)
    return model_hf


@pytest.fixture(autouse=True, scope="module")
def fix_seed():
    torch.manual_seed(0)
    yield


@pytest.fixture
def input_ids() -> torch.Tensor:
    return torch.randint(0, CONFIG.vocab_size, size=(BATCH_SIZE, SEQUENCE_LENGTH), device="cuda")


def _test_nt_to_hf(parallel_context: ParallelContext, input_ids: torch.Tensor):
    model_nt = create_nanotron_model(parallel_context)
    model_hf = create_huggingface_model()
    convert_nt_to_hf(model_nt, model_hf, CONFIG)
    input_mask = torch.ones_like(input_ids)
    logits_nt = model_nt.model(input_ids, input_mask).permute(1, 0, 2)
    logits_hf = model_hf(input_ids).logits
    assert logits_nt.size() == logits_hf.size()
    assert torch.allclose(logits_nt, logits_hf, atol=ATOL), torch.mean(torch.abs(logits_nt - logits_hf))


def test_nt_to_hf(input_ids: torch.Tensor):
    init_distributed(tp=1, dp=1, pp=1)(_test_nt_to_hf)(input_ids=input_ids)


def _test_nt_to_hf_with_files(parallel_context: ParallelContext, input_ids: torch.Tensor, test_context: TestContext):
    # Create and save nanotron model.
    model_nt = create_nanotron_model(parallel_context)
    root = test_context.get_auto_remove_tmp_dir()
    nt_path = root / "nanotron"
    hf_path = root / "hf"
    nanotron.serialize.save_weights(model=model_nt, parallel_context=parallel_context, root_folder=nt_path)
    with open(nt_path / "model_config.json", "w+") as f:
        json.dump(dataclasses.asdict(CONFIG), f)
    input_mask = torch.ones_like(input_ids)
    logits_nt = model_nt.model(input_ids, input_mask).permute(1, 0, 2)
    del model_nt
    # Perform conversion.
    convert_nt_to_hf_and_save(nt_path, hf_path)
    # Load huggingface and get logits.
    model_hf = LlamaForCausalLM.from_pretrained(hf_path).cuda()
    logits_hf = model_hf(input_ids).logits
    assert logits_nt.size() == logits_hf.size()
    torch.testing.assert_allclose(logits_nt, logits_hf, atol=ATOL)


def test_nt_to_hf_with_files(input_ids: torch.Tensor):
    init_distributed(tp=1, dp=1, pp=1)(_test_nt_to_hf_with_files)(input_ids=input_ids, test_context=TestContext())


def _test_hf_to_nt(parallel_context: ParallelContext, input_ids: torch.Tensor):
    model_nt = create_nanotron_model(parallel_context)
    model_hf = create_huggingface_model()
    convert_hf_to_nt(model_hf, model_nt, CONFIG)
    input_mask = torch.ones_like(input_ids)
    logits_nt = model_nt.model(input_ids, input_mask).permute(1, 0, 2)
    logits_hf = model_hf(input_ids).logits
    assert logits_nt.size() == logits_hf.size()
    torch.testing.assert_allclose(logits_hf, logits_nt, atol=ATOL)  


def test_hf_to_nt(input_ids: torch.Tensor):
    init_distributed(tp=1, dp=1, pp=1)(_test_hf_to_nt)(input_ids=input_ids)


def _test_hf_to_nt_with_files(parallel_context: ParallelContext, input_ids: torch.Tensor, test_context: TestContext):
    # Create and save hf model.
    model_hf = create_huggingface_model()
    root = test_context.get_auto_remove_tmp_dir()
    nt_path = root / "nanotron"
    hf_path = root / "hf"
    model_hf.save_pretrained(hf_path)
    logits_hf = model_hf(input_ids).logits
    del model_hf
    # Perform conversion.
    convert_hf_to_nt_and_save(hf_path, nt_path)
    # Load nanotron and get logits.
    input_mask = torch.ones_like(input_ids)
    model_nt = load_nanotron_model(checkpoint_path=nt_path)
    logits_nt = model_nt.model(input_ids, input_mask).permute(1, 0, 2)
    assert logits_nt.size() == logits_hf.size()
    assert torch.allclose(logits_nt, logits_hf, atol=ATOL)


def test_hf_to_nt_with_files(input_ids: torch.Tensor):
    init_distributed(tp=1, dp=1, pp=1)(_test_hf_to_nt_with_files)(input_ids=input_ids, test_context=TestContext())


def _test_composed_conversion(parallel_context: ParallelContext):
    # Get HF statedict.
    model_hf = create_huggingface_model()
    hf_sd = {key: val.clone() for key, val in model_hf.state_dict().items()}
    # Convert once to nanotron, save its statedict.
    model_nt = create_nanotron_model(parallel_context)
    convert_hf_to_nt(model_hf, model_nt, CONFIG)
    nt_sd = {key: val.clone() for key, val in model_nt.state_dict().items()}
    # Convert back to HF, compare statedicts.
    del model_hf
    model_hf = create_huggingface_model()
    convert_nt_to_hf(model_nt, model_hf, CONFIG)
    hf_sd_new = model_hf.state_dict()
    assert set(hf_sd_new) == set(hf_sd)
    assert all(torch.all(hf_sd[key] == hf_sd_new[key]) for key in hf_sd_new)
    # Convert to nanotron one more time, compare statedicts.
    del model_nt
    model_nt = create_nanotron_model(parallel_context)
    convert_hf_to_nt(model_hf, model_nt, CONFIG)
    nt_sd_new = model_nt.state_dict()
    assert set(nt_sd_new) == set(nt_sd)
    assert all(torch.all(nt_sd[key] == nt_sd_new[key]) for key in nt_sd_new)


def test_composed_conversion():
    init_distributed(tp=1, dp=1, pp=1)(_test_composed_conversion)()


def _save_parallel_nanotron(parallel_context: ParallelContext, input_ids: torch.Tensor, nt_path: Path):
    # Create and save a parallel model.
    model_nt = create_nanotron_model(parallel_context)
    nanotron.serialize.save_weights(model=model_nt, parallel_context=parallel_context, root_folder=nt_path)
    with open(nt_path / "model_config.json", "w+") as f:
        json.dump(dataclasses.asdict(CONFIG), f)

    # Get parallel predictions.
    input_ids = input_ids.cuda()  # Move them to the current device index.
    input_mask = torch.ones_like(input_ids)
    logits_nt = model_nt.model(input_ids, input_mask).permute(1, 0, 2)
    if torch.distributed.get_rank() == 0:
        torch.save(logits_nt.detach().cpu(), nt_path / "logits.pt")

    # Convert nanotron to hf, load it and compare logits.
    # hf_path = root/"hf"
    # convert_nt_to_hf_and_save(nt_path, hf_path)
    # model_hf = LlamaForCausalLM.from_pretrained(hf_path).cuda()
    # logits_hf = model_hf(input_ids).logits

    # assert logits_nt.size() == logits_hf.size()
    # assert torch.allclose(logits_nt, logits_hf, atol=ATOL), torch.mean(torch.abs(logits_nt - logits_hf))


def _convert_from_parallel(parallel_context: ParallelContext, input_ids: torch.Tensor, nt_path: Path, hf_path: Path):
    # Convert parallel nanotron to hf, get and save huggingface predictions.
    convert_nt_to_hf_and_save(nt_path, hf_path)
    model_hf = LlamaForCausalLM.from_pretrained(hf_path).cuda()
    logits_hf = model_hf(input_ids).logits
    torch.save(logits_hf.detach().cpu(), hf_path / "logits.pt")


def test_tensor_parallel_conversion(input_ids: torch.Tensor):
    # Set up test.
    test_context = TestContext()
    root = test_context.get_auto_remove_tmp_dir()
    nt_path = root / "nanotron"
    hf_path = root / "nanotron"

    # Launch both parts.
    init_distributed(tp=2, dp=1, pp=1)(_save_parallel_nanotron)(input_ids=input_ids, nt_path=nt_path)
    assert (nt_path / "logits.pt").exists()
    init_distributed(tp=1, dp=1, pp=1)(_convert_from_parallel)(input_ids=input_ids, nt_path=nt_path, hf_path=hf_path)
    assert (hf_path / "logits.pt").exists()

    # Load logits and verify they match.
    logits_nt = torch.load(nt_path / "logits.pt")
    logits_hf = torch.load(hf_path / "logits.pt")
    assert logits_nt.size() == logits_hf.size()
    assert torch.allclose(logits_nt, logits_hf, atol=ATOL), torch.mean(torch.abs(logits_nt - logits_hf))
