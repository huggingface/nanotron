# ruff: noqa: E402
"""
Converts a nanotron model to HF format
Command:
    torchrun --nproc_per_node=1 convert_nanotron_to_hf.py --checkpoint_path=weights-tp1 --save_path=HF_130M
"""

import argparse
import json
from pathlib import Path

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
    model_nanotron_state_dict = nanotron_model.state_dict()
    del nanotron_model
    # Get mapping of Nanotron layer and HF layer
    hf_to_nanotron = {}

    # Static mappings
    hf_to_nanotron["backbone.embeddings.weight"] = "token_position_embeddings.pp_block.token_embedding.weight"
    hf_to_nanotron["backbone.norm_f.weight"] = "final_layer_norm.pp_block.weight"
    hf_to_nanotron["lm_head.weight"] = "lm_head.pp_block.weight"

    # Dynamic mappings within a loop
    for i in range(model_config.num_hidden_layers):
        hf_to_nanotron[f"backbone.layers.{i}.mixer.A_log"] = f"decoder.{i}.pp_block.mixer.A_log"
        hf_to_nanotron[f"backbone.layers.{i}.mixer.D"] = f"decoder.{i}.pp_block.mixer.D"
        hf_to_nanotron[f"backbone.layers.{i}.mixer.in_proj.weight"] = f"decoder.{i}.pp_block.mixer.in_proj.weight"
        hf_to_nanotron[f"backbone.layers.{i}.mixer.conv1d.weight"] = f"decoder.{i}.pp_block.mixer.conv1d.weight"
        hf_to_nanotron[f"backbone.layers.{i}.mixer.conv1d.bias"] = f"decoder.{i}.pp_block.mixer.conv1d.bias"
        hf_to_nanotron[f"backbone.layers.{i}.mixer.x_proj.weight"] = f"decoder.{i}.pp_block.mixer.x_proj.weight"
        hf_to_nanotron[f"backbone.layers.{i}.mixer.x_proj.bias"] = f"decoder.{i}.pp_block.mixer.x_proj.bias"
        hf_to_nanotron[f"backbone.layers.{i}.mixer.dt_proj.weight"] = f"decoder.{i}.pp_block.mixer.dt_proj.weight"
        hf_to_nanotron[f"backbone.layers.{i}.mixer.dt_proj.bias"] = f"decoder.{i}.pp_block.mixer.dt_proj.bias"
        hf_to_nanotron[f"backbone.layers.{i}.mixer.out_proj.weight"] = f"decoder.{i}.pp_block.mixer.out_proj.weight"
        hf_to_nanotron[f"backbone.layers.{i}.mixer.out_proj.bias"] = f"decoder.{i}.pp_block.mixer.out_proj.bias"
        hf_to_nanotron[f"backbone.layers.{i}.norm.weight"] = f"decoder.{i}.pp_block.norm.weight"

    def _reverse_interleave_pattern(N):
        """
        Compute the reverse of the interleave pattern given by _interleave_pattern.
        Example:
        reverse_interleave_pattern(4) -> [0, 2, 1, 3]
        reverse_interleave_pattern(8) -> [0, 2, 4, 6, 1, 3, 5, 7]
        """
        assert N % 2 == 0, "N must be even"

        def __interleave_pattern(N):
            """
            interleave_pattern(4) -> [0, 2, 1, 3]
            interleave_pattern(8) -> [0, 4, 1, 5, 2, 6, 3, 7]
            """
            assert N % 2 == 0, "N must be even"
            pattern = []
            for i in range(N // 2):
                pattern.append(i)
                pattern.append(i + N // 2)
            return pattern

        interleaved_pattern = __interleave_pattern(N)
        reverse_pattern = [0] * N
        for original_index, interleaved_index in enumerate(interleaved_pattern):
            reverse_pattern[interleaved_index] = original_index
        return reverse_pattern

    # Loop over the state dict and convert the keys to HF format
    for module_name_hf, module_hf in hf_model.named_modules():
        for param_name_hf, param_hf in module_hf.named_parameters(recurse=False):
            # Get the Nanotron parameter
            nanotron_key = "model." + hf_to_nanotron[f"{module_name_hf}.{param_name_hf}"]
            param = model_nanotron_state_dict[nanotron_key]

            if "in_proj" in nanotron_key:
                # Undo the interleaving weights in Nanotron to make it HF compatible
                param = param[_reverse_interleave_pattern(param.shape[0]), :]

            with torch.no_grad():
                param_hf.copy_(param)
    return hf_model


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
    dtype = getattr(torch, model_config.dtype)
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
    hf_model = convert_nanotron_to_hf(nanotron_model=nanotron_model, hf_model=hf_model)
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
