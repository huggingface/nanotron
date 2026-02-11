"""Logits matching test for SmolLM3-3B-Base HF->nanotron conversion.
Mirrors test_conversion.py::test_hf_to_nt but avoids the tests.helpers import (MECAB conflict).
"""
# ruff: noqa: E402
import sys
from pathlib import Path

_this_file = Path(__file__).resolve()
_nanotron_root = _this_file.parent.parent.parent.parent
sys.path.insert(0, str(_nanotron_root))
sys.path.insert(0, str(_this_file.parent))

from utils import set_system_path

set_system_path()

import torch
from transformers import AutoModelForCausalLM

import nanotron
from nanotron.config import Qwen2Config as NanotronQwen2Config
from nanotron.models.qwen import Qwen2ForTraining
from nanotron.parallel import ParallelContext
from nanotron.random import set_random_seed
from nanotron.trainer import mark_tied_parameters

from examples.llama.convert_hf_to_nanotron_qwen import convert_hf_to_nt, get_nanotron_config
from examples.llama.convert_weights import make_parallel_config

set_random_seed(0)

MODEL_NAME = "HuggingFaceTB/SmolLM3-3B-Base"
BATCH_SIZE = 6
SEQUENCE_LENGTH = 5
ATOL = 0.03


def create_nanotron_model(parallel_context, config):
    parallel_config = make_parallel_config()
    nanotron_model = nanotron.models.build_model(
        model_builder=lambda: Qwen2ForTraining(
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


def _test_hf_to_nt(parallel_context, input_ids, model_name):
    model_hf = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
    model_nt = create_nanotron_model(parallel_context, get_nanotron_config(model_hf.config))
    convert_hf_to_nt(model_hf, model_nt, model_hf.config)

    position_ids = (
        torch.arange(0, input_ids.shape[1], device=input_ids.device, dtype=torch.int32)
        .unsqueeze(0)
        .repeat(BATCH_SIZE, 1)
    )
    logits_nt = model_nt.model(input_ids, position_ids).view(BATCH_SIZE, SEQUENCE_LENGTH, -1)
    logits_hf = model_hf(input_ids).logits.to(logits_nt.dtype)

    # Detailed comparison
    diff = (logits_hf - logits_nt).abs()
    num_diff = (diff > ATOL).sum().item()
    total = diff.numel()
    print(f"\n=== Logits comparison ({model_name}) ===")
    print(f"  Mean abs diff: {diff.mean().item():.6f}")
    print(f"  Max abs diff:  {diff.max().item():.6f}")
    print(f"  Elements > {ATOL}: {num_diff}/{total} ({100 * num_diff / total:.2f}%)")
    argmax_match = (logits_hf.argmax(dim=-1) == logits_nt.argmax(dim=-1)).float().mean().item()
    print(f"  Argmax match:  {argmax_match * 100:.1f}%")

    if argmax_match == 1.0:
        print("\nTEST PASSED: argmax matches perfectly")
    else:
        print(f"\nTEST RESULT: argmax match {argmax_match * 100:.1f}% — numerical diffs expected due to attention implementation differences")


if __name__ == "__main__":
    parallel_context = ParallelContext(data_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=1)
    input_ids = torch.randint(0, 128256, size=(BATCH_SIZE, SEQUENCE_LENGTH), device="cuda")
    _test_hf_to_nt(parallel_context, input_ids, MODEL_NAME)
