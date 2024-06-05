# ruff: noqa: E402
"""
Converts a nanotron model to HF format
Command:
    torchrun --nproc_per_node=1 convert_nanotron_to_hf.py --checkpoint_path=nanotron_weights --save_path=HF_weights
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from config import MambaModelConfig
from mamba import MambaForTraining
from nanotron import logging
from nanotron.config import (
    AllForwardAllBackwardPipelineEngine,
    ParallelismArgs,
    TensorParallelLinearMode,
)
from nanotron.models import build_model, init_on_device_and_dtype
from nanotron.parallel import ParallelContext
from nanotron.serialize import load_weights
from nanotron.trainer import mark_tied_parameters
from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM

logger = logging.get_logger(__name__)


def convert_checkpoint_and_save(checkpoint_path: Path, save_path: Path):
    device = torch.device("cuda")

    with open(checkpoint_path / "config.yaml", "r") as f:
        attrs = yaml.safe_load(f)
        tokenizer_name = attrs["tokenizer"]["tokenizer_name_or_path"]

    with open(checkpoint_path / "model_config.json", "r") as f:
        attrs = json.load(f)
        model_config = MambaModelConfig(**attrs)

    dtype = getattr(torch, model_config.dtype)

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

    model_nanotron = build_model(
        model_builder=lambda: MambaForTraining(
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=None,
        ),
        parallel_context=parallel_context,
        dtype=dtype,
        device=device,
    )

    mark_tied_parameters(model=model_nanotron, parallel_context=parallel_context)

    # Load checkpoint directly in memory and then only keep the state dictionary
    load_weights(model=model_nanotron, parallel_context=parallel_context, root_folder=checkpoint_path)
    model_nanotron_state_dict = model_nanotron.state_dict()
    del model_nanotron

    # Init the HF mode
    if model_config.ssm_cfg is None:
        model_config_hf = MambaConfig(
            vocab_size=model_config.vocab_size,
            num_hidden_layers=model_config.num_hidden_layers,
            residual_in_fp32=model_config.residual_in_fp32,
            layer_norm_epsilon=model_config.rms_norm_eps,
            hidden_size=model_config.d_model,
        )
    else:
        model_config_hf = MambaConfig(
            vocab_size=model_config.vocab_size,
            num_hidden_layers=model_config.num_hidden_layers,
            residual_in_fp32=model_config.residual_in_fp32,
            layer_norm_epsilon=model_config.rms_norm_eps,
            hidden_size=model_config.d_model,
            state_size=model_config.ssm_cfg["d_state"],
            expand=model_config.ssm_cfg["expand"],
            conv_kernel=model_config.ssm_cfg["d_conv"],
            use_bias=model_config.ssm_cfg["bias"],
            use_conv_bias=model_config.ssm_cfg["conv_bias"],
            time_step_rank=model_config.ssm_cfg["dt_rank"],
            time_step_scale=model_config.ssm_cfg["dt_scale"],
            time_step_min=model_config.ssm_cfg["dt_min"],
            time_step_max=model_config.ssm_cfg["dt_max"],
            time_step_init_scheme=model_config.ssm_cfg["dt_init"],
            time_step_floor=model_config.ssm_cfg["dt_init_floor"],
        )

    # Initialised HF model
    with init_on_device_and_dtype(device, dtype):
        model_hf = MambaForCausalLM._from_config(model_config_hf)

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
    for module_name_hf, module_hf in model_hf.named_modules():
        for param_name_hf, param_hf in module_hf.named_parameters(recurse=False):
            # Get the Nanotron parameter
            nanotron_key = "model." + hf_to_nanotron[f"{module_name_hf}.{param_name_hf}"]
            param = model_nanotron_state_dict[nanotron_key]

            if "in_proj" in nanotron_key:
                # Undo the interleaving weights in Nanotron to make it HF compatible
                param = param[_reverse_interleave_pattern(param.shape[0]), :]

            with torch.no_grad():
                param_hf.copy_(param)

    # Save the model
    model_hf.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

    # Save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(save_path)
    print(f"Tokenizer saved to {save_path}")


def check_converted_model_generation(save_path: Path):
    HARCODED_PROMPT = "What is your "

    tokenizer = AutoTokenizer.from_pretrained(save_path)
    input_ids = tokenizer(HARCODED_PROMPT, return_tensors="pt")["input_ids"]
    print("Inputs:", tokenizer.batch_decode(input_ids))

    model = MambaForCausalLM.from_pretrained(save_path)
    out = model.generate(input_ids, max_new_tokens=100)
    print("Generation (converted): ", tokenizer.batch_decode(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Nanotron weights to HF format")
    parser.add_argument("--checkpoint_path", type=str, default="mamba-130m")
    parser.add_argument("--save_path", type=str, default="mamba-hf")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    checkpoint_path = Path(args.checkpoint_path)

    # Convert Nanotron model to HF format
    convert_checkpoint_and_save(checkpoint_path=checkpoint_path, save_path=save_path)

    # check if the conversion was successful by generating some text
    check_converted_model_generation(save_path=save_path)
