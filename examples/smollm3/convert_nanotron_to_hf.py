"""
Converts a nanotron model to HF format
Command:
    torchrun --nproc_per_node=1 convert_nanotron_to_hf.py --checkpoint_path=nanotron-path --save_path=hf-path
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, Optional, Type

import torch
from nanotron.config import Qwen2Config as NanotronQwen2Config
from nanotron.config import NanotronConfigs
from nanotron.models import init_on_device_and_dtype
from nanotron.models.qwen import Qwen2ForTraining
from transformers import AutoTokenizer, SmolLM3ForCausalLM
from transformers import SmolLM3Config as HFSmollm3Config

from convert_weights import get_config_mapping, get_weight_mapping, load_nanotron_model

TEST_PROMPT = "What is the meaning of the word chutzpah?\nThe word chutzpah means"


def _handle_attention_block(
        qkv: torch.Tensor,
        part: Literal["q", "k", "v"],
        n_q_heads: int,
        n_kv_heads: int,
        d_qk: int,
        interleave,
) -> torch.Tensor:
    # Huggingface Llama separates the q, k, v weights (as opposed to nanotron).
    # Furthermore, in the rotary embeddings in nanotron expects interleaved pairs of even
    # and odd dimensions GPT-J style, while the huggingface implementation expects
    # the whole 1st half and then the whole 2nd half GPT-NeoX style (for more information
    # see flash_attn.layers.rotary.RotaryEmbedding).
    # This function selects the proper chunk of the bundled qkv tensor and permutation
    # to ensure correct transformation to huggingface.

    def interleave_weight(w: torch.Tensor):
        w_new = []
        for head_w in w.split(d_qk):
            head_w = head_w.view(d_qk // 2, 2, -1).transpose(0, 1).reshape(d_qk, -1)
            w_new.append(head_w)
        return torch.cat(w_new)

    assert part in ["q", "k", "v"], "part must be one of [q, k, v]"

    index_end_q = n_q_heads * d_qk
    index_end_k = index_end_q + n_kv_heads * d_qk
    if part == "q":
        return interleave_weight(qkv[:index_end_q]) if interleave else qkv[:index_end_q]
    elif part == "k":
        return interleave_weight(qkv[index_end_q:index_end_k]) if interleave else qkv[index_end_q:index_end_k]
    elif part == "v":
        return qkv[index_end_k:]


def _handle_gate_up_proj(gate_up_proj: torch.Tensor, gate: bool) -> torch.Tensor:
    # The gate and up projection are bundled in nanotron.
    # This function selects the proper chunk in the bundled weights to return
    # either the gate or the up projection only.
    weight_size = gate_up_proj.shape[0] // 2
    if gate:
        return gate_up_proj[:weight_size]
    else:
        return gate_up_proj[weight_size:]


def convert_nt_to_hf(
        nanotron_model: Qwen2ForTraining,
        hf_model: SmolLM3ForCausalLM,
        model_config: NanotronQwen2Config,
        interleave_qkv: bool = False,
):
    """Converts the weights from the nanotron_model to hf_model, making modifications
    in-place."""

    nanotron_model_state_dict = nanotron_model.state_dict()

    hf_to_nt = get_weight_mapping(model_config, nt_to_hf=False)
    for module_name_hf, module_hf in hf_model.named_modules():
        for param_name_hf, param_hf in module_hf.named_parameters(recurse=False):
            # Get the Nanotron parameter
            nanotron_key = hf_to_nt[f"{module_name_hf}.{param_name_hf}"]
            param = nanotron_model_state_dict[nanotron_key]

            if "qkv_proj" in nanotron_key:
                proj_name = module_name_hf.split(".")[4][0]
                param = _handle_attention_block(
                    param,
                    proj_name,
                    model_config.num_attention_heads,
                    model_config.num_key_value_heads,
                    model_config.hidden_size // model_config.num_attention_heads,
                    interleave_qkv,
                )

            elif "gate_up_proj" in nanotron_key:
                gate = "gate" in module_name_hf
                param = _handle_gate_up_proj(param, gate)

            with torch.no_grad():
                # print(f"Copying parameter {module_name_hf} ({param_hf.shape}) from {nanotron_key} ({param.shape})")
                param_hf.copy_(param)


def get_hf_config(config: NanotronConfigs) -> HFSmollm3Config:
    """Converts a nanotron configuration to huggingface configuration."""
    attrs = {key: getattr(config, value) for key, value in get_config_mapping(nt_to_hf=False).items()}

    if config.sliding_window_size is not None:
        attrs["use_sliding_window"] = True
        attrs["sliding_window"] = config.sliding_window_size
    else:
        attrs["use_sliding_window"] = False
        attrs["sliding_window"] = None
    attrs["no_rope_layer_interval"] = config.no_rope_layer
    return HFSmollm3Config(**attrs)


def convert_checkpoint_and_save(
        checkpoint_path: Path,
        save_path: Path,
        tokenizer_name: Optional[str] = None,
        config_cls: Type[NanotronConfigs] = NanotronQwen2Config,
):
    """Loads the nanotron checkpoint in `checkpoint_path`, creates
    a new huggingface instance, copies the weights from the nanotron checkpoint
    and saves the transformed huggingface to `save_path`."""

    # Init nanotron model.
    with open(checkpoint_path / "model_config.json", "r") as f:
        attrs = json.load(f)
        model_config = config_cls(**attrs)
    nanotron_model = load_nanotron_model(
        model_config=model_config,
        checkpoint_path=checkpoint_path,
    )
    # Init huggingface model.
    with init_on_device_and_dtype(torch.device("cuda"), torch.bfloat16):
        model_config_hf = get_hf_config(model_config)
        hf_model = SmolLM3ForCausalLM._from_config(model_config_hf)

    # Copy weights, initialize tokenizer and save model.
    if tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.save_pretrained(save_path)
    convert_nt_to_hf(nanotron_model, hf_model, model_config)
    hf_model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


def check_converted_model_generation(save_path: Path):
    """Loads a huggingface model and tokenizer from `save_path` and
    performs a dummy text generation."""

    tokenizer = AutoTokenizer.from_pretrained(save_path)
    input_ids = tokenizer(TEST_PROMPT, return_tensors="pt")["input_ids"].cuda()
    print("Inputs:", tokenizer.batch_decode(input_ids))

    model = SmolLM3ForCausalLM.from_pretrained(save_path).cuda().bfloat16()
    out = model.generate(input_ids, max_new_tokens=100)
    print("Generation (converted): ", tokenizer.batch_decode(out))


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert Nanotron weights to HF format")
    parser.add_argument("--checkpoint_path", type=Path, default="smollm3/", help="Path to the checkpoint")
    parser.add_argument("--save_path", type=Path, default="smollm3-hf/", help="Path to save the HF model")
    parser.add_argument("--tokenizer_name", type=str, default="HuggingFaceTB/SmolLM3-3B")
    args = parser.parse_args()

    # Convert Nanotron model to HF format.
    convert_checkpoint_and_save(
        checkpoint_path=args.checkpoint_path, save_path=args.save_path, tokenizer_name=args.tokenizer_name,
        config_cls=NanotronQwen2Config
    )

    # Check if the conversion was successful by generating some text.
    if args.tokenizer_name is not None:
        check_converted_model_generation(save_path=args.save_path)