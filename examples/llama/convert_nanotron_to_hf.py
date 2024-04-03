# ruff: noqa: E402
"""
Converts a nanotron model to HF format
Command:
    torchrun --nproc_per_node=1 convert_nanotron_to_hf.py --checkpoint_path=weights-tp1 --save_path=HF_130M
"""

import argparse
import json
from pathlib import Path
from typing import Literal

import torch
from nanotron import logging
from nanotron.config import (
    AllForwardAllBackwardPipelineEngine,
    ParallelismArgs,
    TensorParallelLinearMode,
)
from nanotron.config import LlamaConfig as NanotronLlamaConfig
from nanotron.models import build_model, init_on_device_and_dtype
from nanotron.models.llama import LlamaForTraining
from nanotron.parallel import ParallelContext
from nanotron.serialize import load_weights
from nanotron.trainer import mark_tied_parameters
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import LlamaConfig as HFLlamaConfig

logger = logging.get_logger(__name__)

HARCODED_PROMPT = "what is the meaning of the word chutzpah?"


def convert_nanotron_to_hf(
    nanotron_model: LlamaForTraining, hf_model: LlamaForCausalLM, model_config: NanotronLlamaConfig
) -> LlamaForCausalLM:
    nanotron_model_state_dict = nanotron_model.state_dict()
    # Get mapping of Nanotron layer and HF layer
    hf_to_nanotron = {}
    # Static mappings
    hf_to_nanotron["lm_head.weight"] = "lm_head.pp_block.weight"
    hf_to_nanotron["model.embed_tokens.weight"] = "token_position_embeddings.pp_block.token_embedding.weight"
    hf_to_nanotron["model.norm.weight"] = "final_layer_norm.pp_block.weight"
    hf_to_nanotron["model.embed_tokens.weight"] = "token_position_embeddings.pp_block.token_embedding.weight"
    # Dynamic mappings within a loop
    for i in range(model_config.num_hidden_layers):
        hf_to_nanotron[f"model.layers.{i}.self_attn.q_proj.weight"] = f"decoder.{i}.pp_block.attn.qkv_proj.weight"
        hf_to_nanotron[f"model.layers.{i}.self_attn.k_proj.weight"] = f"decoder.{i}.pp_block.attn.qkv_proj.weight"
        hf_to_nanotron[f"model.layers.{i}.self_attn.v_proj.weight"] = f"decoder.{i}.pp_block.attn.qkv_proj.weight"
        hf_to_nanotron[f"model.layers.{i}.self_attn.o_proj.weight"] = f"decoder.{i}.pp_block.attn.o_proj.weight"
        hf_to_nanotron[f"model.layers.{i}.mlp.gate_proj.weight"] = f"decoder.{i}.pp_block.mlp.gate_up_proj.weight"
        hf_to_nanotron[f"model.layers.{i}.mlp.gate_proj.bias"] = f"decoder.{i}.pp_block.mlp.gate_up_proj.bias"
        hf_to_nanotron[f"model.layers.{i}.mlp.up_proj.weight"] = f"decoder.{i}.pp_block.mlp.gate_up_proj.weight"
        hf_to_nanotron[f"model.layers.{i}.mlp.up_proj.bias"] = f"decoder.{i}.pp_block.mlp.gate_up_proj.bias"
        hf_to_nanotron[f"model.layers.{i}.mlp.down_proj.weight"] = f"decoder.{i}.pp_block.mlp.down_proj.weight"
        hf_to_nanotron[f"model.layers.{i}.mlp.down_proj.bias"] = f"decoder.{i}.pp_block.mlp.down_proj.bias"
        hf_to_nanotron[f"model.layers.{i}.input_layernorm.weight"] = f"decoder.{i}.pp_block.input_layernorm.weight"
        hf_to_nanotron[
            f"model.layers.{i}.post_attention_layernorm.weight"
        ] = f"decoder.{i}.pp_block.post_attention_layernorm.weight"
    # Loop over the state dict and convert the keys to HF format
    for module_name_hf, module_hf in hf_model.named_modules():
        for param_name_hf, param_hf in module_hf.named_parameters(recurse=False):
            # Get the Nanotron parameter
            nanotron_key = "model." + hf_to_nanotron[f"{module_name_hf}.{param_name_hf}"]
            param = nanotron_model_state_dict[nanotron_key]
            if "qkv_proj" in nanotron_key:
                proj_name = module_name_hf.split(".")[4][0]
                param = _handle_attention_block(param, proj_name)
            elif "gate_up_proj" in nanotron_key:
                gate = "gate" in param_name_hf
                param = _handle_gate_up_proj(param, gate)
            with torch.no_grad():
                param_hf.copy_(param)
    return hf_model


def _handle_attention_block(qkv: torch.Tensor, part: Literal["q", "k", "v"]) -> torch.Tensor:
    assert part in ["q", "k", "v"], "part must be one of [q, k, v]"
    if not qkv.shape[0] % 3 == 0:
        raise ValueError("qkv shape must be a multiple of 3")
    # Divide by 3 beceause we have q, k, v, each of which represents
    # one third of the total size of the first dimension
    weight_size = qkv.shape[0] // 3
    if part == "q":
        return qkv[:weight_size]
    elif part == "k":
        return qkv[weight_size : 2 * weight_size]
    else:
        return qkv[2 * weight_size :]


def _handle_gate_up_proj(gate_up_proj: torch.Tensor, gate: bool) -> torch.Tensor:
    weight_size = gate_up_proj.shape[0] // 2
    if gate:
        return gate_up_proj[:weight_size]
    else:
        return gate_up_proj[weight_size:]


def load_nanotron_model(
    model_config: NanotronLlamaConfig, device: torch.device, dtype: torch.dtype, checkpoint_path: Path
) -> LlamaForTraining:
    parallel_config = ParallelismArgs(
        dp=1,
        pp=1,
        tp=1,
        pp_engine=AllForwardAllBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )
    parallel_context = ParallelContext(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
    )
    nanotron_model = build_model(
        model_builder=lambda: LlamaForTraining(
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=None,
        ),
        parallel_context=parallel_context,
        dtype=dtype,
        device=device,
    )
    mark_tied_parameters(model=nanotron_model, parallel_context=parallel_context)
    # Load checkpoint directly in memory and then only keep the state dictionary
    load_weights(model=nanotron_model, parallel_context=parallel_context, root_folder=checkpoint_path)
    return nanotron_model


def convert_checkpoint_and_save(checkpoint_path: Path, save_path: Path):
    device = torch.device("cuda")
    with open(checkpoint_path / "model_config.json", "r") as f:
        attrs = json.load(f)
        model_config = NanotronLlamaConfig(**attrs)
    dtype = getattr(torch, "bfloat16")
    nanotron_model = load_nanotron_model(
        model_config=model_config, device=device, dtype=dtype, checkpoint_path=checkpoint_path
    )
    # Init the HF mode
    model_config_hf = HFLlamaConfig(
        bos_token_id=model_config.bos_token_id,
        eos_token_id=model_config.eos_token_id,
        hidden_act=model_config.hidden_act,
        hidden_size=model_config.hidden_size,
        initializer_range=model_config.initializer_range,
        intermediate_size=model_config.intermediate_size,
        max_position_embeddings=model_config.max_position_embeddings,
        num_attention_heads=model_config.num_attention_heads,
        num_hidden_layers=model_config.num_hidden_layers,
        num_key_value_heads=model_config.num_key_value_heads,
        pad_token_id=model_config.pad_token_id,
        pretraining_tp=model_config.pretraining_tp,
        rms_norm_eps=model_config.rms_norm_eps,
        rope_scaling=model_config.rope_scaling,
        tie_word_embeddings=model_config.tie_word_embeddings,
        use_cache=model_config.use_cache,
        vocab_size=model_config.vocab_size,
    )
    # Initialised HF model
    with init_on_device_and_dtype(device, dtype):
        hf_model = LlamaForCausalLM._from_config(model_config_hf)
    hf_model = convert_nanotron_to_hf(nanotron_model, hf_model, model_config)
    # Save the model
    hf_model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


def check_converted_model_generation(save_path: Path, tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    input_ids = tokenizer(HARCODED_PROMPT, return_tensors="pt")["input_ids"]
    print("Inputs:", tokenizer.batch_decode(input_ids))
    model = LlamaForCausalLM.from_pretrained(save_path)
    out = model.generate(input_ids, max_new_tokens=100)
    print("Generation (converted): ", tokenizer.batch_decode(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Nanotron weights to HF format")
    parser.add_argument("--checkpoint_path", type=str, default="llama-7b", help="Path to the checkpoint")
    parser.add_argument("--save_path", type=str, default="llama-7b-hf", help="Path to save the HF model")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    args = parser.parse_args()
    save_path = Path(args.save_path)
    checkpoint_path = Path(args.checkpoint_path)
    # Convert Nanotron model to HF format
    convert_checkpoint_and_save(checkpoint_path=checkpoint_path, save_path=save_path)
    # check if the conversion was successful by generating some text
    check_converted_model_generation(save_path=save_path, tokenizer_name=args.tokenizer_name)
