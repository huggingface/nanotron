"""
Converts a HF model to nanotron format
Command:
    torchrun --nproc_per_node=1 examples/llama/convert_hf_to_nanotron.py --checkpoint_path=hf_weights --save_path=nanotron_weights
"""

import dataclasses
import json
from argparse import ArgumentParser
from pathlib import Path

import nanotron
import torch
from nanotron.config import LlamaConfig as NanotronLlamaConfig
from nanotron.config import Qwen2Config as NanotronQwen2Config
from nanotron.models.llama import LlamaForTraining
from transformers import LlamaConfig as HFLlamaConfig
from transformers import LlamaForCausalLM

from .convert_weights import get_config_mapping, get_weight_mapping, load_nanotron_model


def _handle_attention_block(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_q_heads: int,
    n_kv_heads: int,
    d_qk: int,
    interleave: bool,
) -> torch.Tensor:
    # Huggingface Llama separates the q, k, v weights (as opposed to nanotron).
    # Furthermore, in the rotary embeddings in nanotron expects interleaved pairs of even
    # and odd dimensions GPT-J style, while the huggingface implementation expects
    # the whole 1st half and then the whole 2nd half GPT-NeoX style (for more information
    # see flash_attn.layers.rotary.RotaryEmbedding).
    # This function handles the concatenation of the q, k, v weights and proper permutation
    # to ensure correct transformation.

    def interleave_weight(w: torch.Tensor):
        w_new = []
        for head_w in w.split(d_qk):
            head_w = head_w.view(2, d_qk // 2, -1).transpose(0, 1).reshape(d_qk, -1)
            w_new.append(head_w)
        return torch.cat(w_new)

    q = interleave_weight(q) if interleave else q
    k = interleave_weight(k) if interleave else k
    return torch.cat([q, k, v])


def convert_hf_to_nt(
    model_hf: LlamaForCausalLM, model_nt: LlamaForTraining, config: NanotronLlamaConfig, interleave_qkv: bool = False
):
    """Converts the weights from the model_hf to model_nt, making modifications
    in-place."""

    hf_sd = model_hf.state_dict()
    nt_to_hf = get_weight_mapping(config, nt_to_hf=True)

    for module_name_nt, module_nt in model_nt.named_modules():
        for param_name_nt, param_nt in module_nt.named_parameters(recurse=False):
            # In the case of qkv_proj, the nt_to_hf has exactly three keys, ccorresponding
            # to q, k, v.
            if "qkv_proj" in module_name_nt:
                key_k, key_q, key_v = sorted(nt_to_hf[f"{module_name_nt}.{param_name_nt}"])
                q = hf_sd[key_q]
                k = hf_sd[key_k]
                v = hf_sd[key_v]
                param = _handle_attention_block(
                    q,
                    k,
                    v,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.hidden_size // config.num_attention_heads,
                    interleave_qkv,
                )
            # The case of gate_up_proj, nt_to_hf_map has two keys.
            elif "gate_up_proj" in module_name_nt:
                key_gate, key_up = sorted(nt_to_hf[f"{module_name_nt}.{param_name_nt}"])
                gate = hf_sd[key_gate]
                up = hf_sd[key_up]
                param = torch.cat([gate, up])
            # All other cases are simple 1-to-1 correspondence.
            else:
                hf_key = nt_to_hf[f"{module_name_nt}.{param_name_nt}"]
                param = hf_sd[hf_key]

            with torch.no_grad():
                param_nt.copy_(param)


def get_nanotron_config(config: HFLlamaConfig) -> NanotronQwen2Config:
    """Converts a huggingface configuration to nanotron configuration."""
    attrs = {key: getattr(config, value) for key, value in get_config_mapping(nt_to_hf=True).items()}
    return NanotronQwen2Config(**attrs)


def convert_checkpoint_and_save(checkpoint_path: Path, save_path: Path):
    """Loads the huggingface checkpoint in `checkpoint_path`, creates
    a new nanotron instance, copies the weights from the huggingface checkpoint
    and saves the transformed nanotron to `save_path`."""

    # Load huggingface.
    hf_model = LlamaForCausalLM.from_pretrained(checkpoint_path)

    # Init nanotron model.
    model_config = get_nanotron_config(hf_model.config)
    nanotron_model = load_nanotron_model(model_config=model_config)

    # Copy weights and save model.
    parallel_context = nanotron.parallel.ParallelContext(
        data_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=1
    )
    convert_hf_to_nt(hf_model, nanotron_model, model_config)
    nanotron.serialize.save_weights(model=nanotron_model, parallel_context=parallel_context, root_folder=save_path)
    with open(save_path / "model_config.json", "w+") as f:
        json.dump(dataclasses.asdict(model_config), f)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert HF weights to nanotron format")
    parser.add_argument("--checkpoint_path", type=Path, default="llama-7b", help="Path to the checkpoint")
    parser.add_argument("--save_path", type=Path, default="llama-7b-hf", help="Path to save the nanotron model")
    args = parser.parse_args()

    # Convert HF model to nanotron format.
    convert_checkpoint_and_save(checkpoint_path=args.checkpoint_path, save_path=args.save_path)
