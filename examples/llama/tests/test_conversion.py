# ruff: noqa: E402
import dataclasses
import json
from pathlib import Path
from typing import Optional

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from utils import set_system_path

set_system_path()

import nanotron
from nanotron import distributed as dist
from nanotron.config import LlamaConfig as NanotronLlamaConfig
from nanotron.config import NanotronConfigs
from nanotron.config import Qwen2Config as NanotronQwen2Config
from nanotron.models.base import init_on_device_and_dtype
from nanotron.models.llama import LlamaForTraining
from nanotron.models.qwen import Qwen2ForTraining
from nanotron.parallel import ParallelContext
from nanotron.random import set_random_seed
from nanotron.trainer import mark_tied_parameters

from examples.llama.convert_hf_to_nanotron import convert_checkpoint_and_save as convert_hf_to_nt_and_save
from examples.llama.convert_hf_to_nanotron import convert_hf_to_nt, get_nanotron_config
from examples.llama.convert_nanotron_to_hf import convert_checkpoint_and_save as convert_nt_to_hf_and_save
from examples.llama.convert_nanotron_to_hf import convert_nt_to_hf, get_hf_config
from examples.llama.convert_weights import load_nanotron_model, make_parallel_config
from tests.helpers.context import TestContext
from tests.helpers.utils import init_distributed

set_random_seed(0)
torch.backends.cudnn.deterministic = True

BATCH_SIZE = 6
SEQUENCE_LENGTH = 5
ATOL = 0.03

NT_MODEL_INSTANCE = Qwen2ForTraining

if NT_MODEL_INSTANCE == LlamaForTraining:
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
            "attention_bias": False,
        }
    )
elif NT_MODEL_INSTANCE == Qwen2ForTraining:
    CONFIG = NanotronQwen2Config(
        **{
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "num_key_value_heads": 8,
            "pad_token_id": None,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": 4096,
            "_attn_implementation": "sdpa",
            "attention_bias": False,
            "rope_interleaved": False,
        }
    )


def create_nanotron_model(parallel_context: ParallelContext, config: NanotronConfigs = CONFIG) -> NT_MODEL_INSTANCE:
    parallel_config = make_parallel_config(
        tp=parallel_context.tensor_parallel_size,
        dp=parallel_context.data_parallel_size,
        pp=parallel_context.pipeline_parallel_size,
    )
    nanotron_model = nanotron.models.build_model(
        model_builder=lambda: NT_MODEL_INSTANCE(
            config=config,
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


def create_huggingface_model(model_name: Optional[str] = None) -> LlamaForCausalLM:
    if model_name is None:
        with init_on_device_and_dtype(torch.device("cuda"), torch.bfloat16):
            model_hf = LlamaForCausalLM._from_config(get_hf_config(CONFIG))
    else:
        with init_on_device_and_dtype(torch.device("cuda"), torch.bfloat16):
            model_hf = AutoModelForCausalLM.from_pretrained(model_name)
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
    if NT_MODEL_INSTANCE == Qwen2ForTraining:
        position_ids = (
            torch.arange(0, input_ids.shape[1], device=input_ids.device, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(BATCH_SIZE, 1)
        )
        logits_nt = model_nt.model(input_ids, position_ids).view(BATCH_SIZE, SEQUENCE_LENGTH, -1)
    else:
        logits_nt = model_nt.model(input_ids, input_mask).permute(1, 0, 2)
    logits_hf = model_hf(input_ids).logits
    torch.testing.assert_allclose(logits_nt, logits_hf.to(logits_nt.dtype), atol=ATOL, rtol=ATOL)


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
    if NT_MODEL_INSTANCE == Qwen2ForTraining:
        position_ids = (
            torch.arange(0, input_ids.shape[1], device=input_ids.device, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(BATCH_SIZE, 1)
        )
        logits_nt = model_nt.model(input_ids, position_ids).view(BATCH_SIZE, SEQUENCE_LENGTH, -1)
    else:
        logits_nt = model_nt.model(input_ids, input_mask).permute(1, 0, 2)
    del model_nt
    # Perform conversion.
    convert_nt_to_hf_and_save(nt_path, hf_path, config_cls=NanotronQwen2Config)
    # Load huggingface and get logits.
    model_hf = LlamaForCausalLM.from_pretrained(hf_path).cuda()
    logits_hf = model_hf(input_ids).logits
    torch.testing.assert_allclose(logits_nt, logits_hf.to(logits_nt.dtype), atol=ATOL, rtol=ATOL)


def test_nt_to_hf_with_files(input_ids: torch.Tensor):
    init_distributed(tp=1, dp=1, pp=1)(_test_nt_to_hf_with_files)(input_ids=input_ids, test_context=TestContext())


def _test_hf_to_nt(parallel_context: ParallelContext, input_ids: torch.Tensor, model_name: Optional[str]):
    model_hf = create_huggingface_model(model_name)
    model_nt = create_nanotron_model(parallel_context, get_nanotron_config(model_hf.config))
    convert_hf_to_nt(model_hf, model_nt, model_hf.config)
    input_mask = torch.ones_like(input_ids)
    if NT_MODEL_INSTANCE == Qwen2ForTraining:
        position_ids = (
            torch.arange(0, input_ids.shape[1], device=input_ids.device, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(BATCH_SIZE, 1)
        )
        logits_nt = model_nt.model(input_ids, position_ids).view(BATCH_SIZE, SEQUENCE_LENGTH, -1)
    else:
        logits_nt = model_nt.model(input_ids, input_mask).permute(1, 0, 2)
    logits_hf = model_hf(input_ids).logits
    torch.testing.assert_allclose(logits_hf.to(logits_nt.dtype), logits_nt, atol=ATOL, rtol=ATOL)


@pytest.mark.parametrize("model_name", [None, "HuggingFaceTB/SmolLM2-135M"])
def test_hf_to_nt(input_ids: torch.Tensor, model_name: Optional[str]):
    init_distributed(tp=1, dp=1, pp=1)(_test_hf_to_nt)(input_ids=input_ids, model_name=model_name)


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
    model_nt = load_nanotron_model(checkpoint_path=nt_path, config_cls=NanotronQwen2Config)
    if NT_MODEL_INSTANCE == Qwen2ForTraining:
        position_ids = (
            torch.arange(0, input_ids.shape[1], device=input_ids.device, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(BATCH_SIZE, 1)
        )
        logits_nt = model_nt.model(input_ids, position_ids).view(BATCH_SIZE, SEQUENCE_LENGTH, -1)
    else:
        logits_nt = model_nt.model(input_ids, input_mask).permute(1, 0, 2)
    torch.testing.assert_allclose(logits_nt, logits_hf.to(logits_nt.dtype), atol=ATOL, rtol=ATOL)


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
    if NT_MODEL_INSTANCE == Qwen2ForTraining:
        position_ids = (
            torch.arange(0, input_ids.shape[1], device=input_ids.device, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(BATCH_SIZE, 1)
        )
        logits_nt = model_nt.model(input_ids, position_ids).view(BATCH_SIZE, SEQUENCE_LENGTH, -1)
    else:
        logits_nt = model_nt.model(input_ids, input_mask).permute(1, 0, 2)

    tp_pg = parallel_context.tp_pg
    tp_rank = dist.get_rank(tp_pg)
    sharded_vocab_size = CONFIG.vocab_size // tp_pg.size()
    logits_nt = logits_nt.view(BATCH_SIZE, SEQUENCE_LENGTH, -1).contiguous()
    assert logits_nt.shape[-1] == sharded_vocab_size
    logits_nt_full = torch.empty(
        tp_pg.size(), BATCH_SIZE, SEQUENCE_LENGTH, sharded_vocab_size, device="cuda", dtype=logits_nt.dtype
    )
    dist.all_gather_into_tensor(logits_nt_full, logits_nt, group=tp_pg)
    logits_nt_full = logits_nt_full.permute(1, 2, 0, 3).reshape(BATCH_SIZE, SEQUENCE_LENGTH, -1)
    torch.testing.assert_close(
        logits_nt, logits_nt_full[:, :, tp_rank * sharded_vocab_size : (tp_rank + 1) * sharded_vocab_size]
    )
    if tp_rank == 0:
        torch.save(logits_nt_full.detach().cpu(), nt_path / "logits_nt.pt")


def _convert_from_parallel(parallel_context: ParallelContext, input_ids: torch.Tensor, nt_path: Path, hf_path: Path):
    # Convert parallel nanotron to hf, get and save huggingface predictions.
    convert_nt_to_hf_and_save(nt_path, hf_path, config_cls=NanotronQwen2Config)
    model_hf = LlamaForCausalLM.from_pretrained(hf_path).cuda()
    logits_hf = model_hf(input_ids).logits
    torch.save(logits_hf.detach().cpu(), hf_path / "logits_hf.pt")


def test_tensor_parallel_conversion(input_ids: torch.Tensor):
    # Set up test.
    test_context = TestContext()
    root = test_context.get_auto_remove_tmp_dir()
    nt_path = root / "nanotron"
    hf_path = root / "nanotron"

    # Launch both parts.
    init_distributed(tp=2, dp=1, pp=1)(_save_parallel_nanotron)(input_ids=input_ids, nt_path=nt_path)
    assert (nt_path / "logits_nt.pt").exists()
    init_distributed(tp=1, dp=1, pp=1)(_convert_from_parallel)(input_ids=input_ids, nt_path=nt_path, hf_path=hf_path)
    assert (hf_path / "logits_hf.pt").exists()

    # Load logits and verify they match.
    logits_nt = torch.load(nt_path / "logits_nt.pt")
    logits_hf = torch.load(hf_path / "logits_hf.pt")
    torch.testing.assert_allclose(logits_nt, logits_hf, atol=ATOL, rtol=ATOL)


if __name__ == "__main__":
    # run all tests
    # test_nt_to_hf(input_ids=torch.randint(0, CONFIG.vocab_size, size=(BATCH_SIZE, SEQUENCE_LENGTH), device="cuda"))
    # test_hf_to_nt(input_ids=torch.randint(0, CONFIG.vocab_size, size=(BATCH_SIZE, SEQUENCE_LENGTH), device="cuda"))
    test_tensor_parallel_conversion(
        input_ids=torch.randint(0, CONFIG.vocab_size, size=(BATCH_SIZE, SEQUENCE_LENGTH), device="cuda")
    )

    # Warning: Converting from HF to Nanotron is a better test because we don't initialize weights in standard way. (e.g. Layernorms)
    # Test SmolLM2-135M
    # test_hf_to_nt(
    #     input_ids=torch.randint(0, CONFIG.vocab_size, size=(BATCH_SIZE, SEQUENCE_LENGTH), device="cuda"),
    #     model_name="HuggingFaceTB/SmolLM2-135M",
    # )
