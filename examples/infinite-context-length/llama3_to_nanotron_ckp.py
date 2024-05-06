"""
Converts a HF model to nanotron format
Command:
    torchrun --nproc_per_node=1 examples/infinite-context-length/llama3_to_nanotron_ckp.py
"""

import json
from pathlib import Path
from typing import Optional

import nanotron
import torch
import yaml

# from convert_weights import get_config_mapping, get_weight_mapping, load_nanotron_model, make_parallel_config
from nanotron.config import LlamaConfig as NanotronLlamaConfig
from nanotron.config.config import Config, GeneralArgs, ModelArgs, TokenizerArgs
from nanotron.config.models_config import RandomInit
from nanotron.models.llama import LlamaForTraining
from nanotron.trainer import mark_tied_parameters
from transformers import LlamaConfig as HFLlamaConfig
from transformers import LlamaForCausalLM


def get_weight_mapping(config: NanotronLlamaConfig, nt_to_hf: bool = True) -> dict[str, str]:
    """Returns the nanotron to huggingface parameter mapping if `nt_to_hf`, otherwise the
    huggingface to nanotron mapping."""

    hf_to_nt_map = {}
    hf_to_nt_map["lm_head.weight"] = "model.lm_head.pp_block.weight"
    hf_to_nt_map["model.embed_tokens.weight"] = "model.token_position_embeddings.pp_block.token_embedding.weight"
    hf_to_nt_map["model.norm.weight"] = "model.final_layer_norm.pp_block.weight"
    hf_to_nt_map["model.embed_tokens.weight"] = "model.token_position_embeddings.pp_block.token_embedding.weight"

    for i in range(config.num_hidden_layers):
        hf_prefix = f"model.layers.{i}"
        nt_prefix = f"model.decoder.{i}.pp_block"
        hf_to_nt_map[f"{hf_prefix}.self_attn.q_proj.weight"] = f"{nt_prefix}.attn.qkv_proj.weight"
        hf_to_nt_map[f"{hf_prefix}.self_attn.k_proj.weight"] = f"{nt_prefix}.attn.qkv_proj.weight"
        hf_to_nt_map[f"{hf_prefix}.self_attn.v_proj.weight"] = f"{nt_prefix}.attn.qkv_proj.weight"
        hf_to_nt_map[f"{hf_prefix}.self_attn.o_proj.weight"] = f"{nt_prefix}.attn.o_proj.weight"
        hf_to_nt_map[f"{hf_prefix}.mlp.gate_proj.weight"] = f"{nt_prefix}.mlp.gate_up_proj.weight"
        hf_to_nt_map[f"{hf_prefix}.mlp.gate_proj.bias"] = f"{nt_prefix}.mlp.gate_up_proj.bias"
        hf_to_nt_map[f"{hf_prefix}.mlp.up_proj.weight"] = f"{nt_prefix}.mlp.gate_up_proj.weight"
        hf_to_nt_map[f"{hf_prefix}.mlp.up_proj.bias"] = f"{nt_prefix}.mlp.gate_up_proj.bias"
        hf_to_nt_map[f"{hf_prefix}.mlp.down_proj.weight"] = f"{nt_prefix}.mlp.down_proj.weight"
        hf_to_nt_map[f"{hf_prefix}.mlp.down_proj.bias"] = f"{nt_prefix}.mlp.down_proj.bias"
        hf_to_nt_map[f"{hf_prefix}.input_layernorm.weight"] = f"{nt_prefix}.input_layernorm.weight"
        hf_to_nt_map[f"{hf_prefix}.post_attention_layernorm.weight"] = f"{nt_prefix}.post_attention_layernorm.weight"

    if nt_to_hf:
        nt_to_hf_map = {}
        for hf, nt in hf_to_nt_map.items():
            # Because the qkv and gate_up projections are separated in the
            # huggingface format, when we return nanotron to huggingface
            # we will need to return a list of parameters instead (e.g.
            # the `qkv_proj` will point to a list `[q_proj, k_proj, v_proj]`).
            if nt in nt_to_hf_map and isinstance(nt_to_hf_map[nt], list):
                nt_to_hf_map[nt].append(hf)
            elif nt in nt_to_hf_map:
                nt_to_hf_map[nt] = [nt_to_hf_map[nt], hf]
            else:
                nt_to_hf_map[nt] = hf
        return nt_to_hf_map
    return hf_to_nt_map


def get_config_mapping(nt_to_hf: bool = True) -> dict[str, str]:
    """Returns either the nanotron to huggingface (if `nt_to_hf`)
    configuration mapping, or the huggingface to nanotron."""

    hf_to_nt_map = {
        "bos_token_id": "bos_token_id",
        "eos_token_id": "eos_token_id",
        "hidden_act": "hidden_act",
        "hidden_size": "hidden_size",
        "initializer_range": "initializer_range",
        "intermediate_size": "intermediate_size",
        "max_position_embeddings": "max_position_embeddings",
        "num_attention_heads": "num_attention_heads",
        "num_hidden_layers": "num_hidden_layers",
        "num_key_value_heads": "num_key_value_heads",
        "pad_token_id": "pad_token_id",
        "pretraining_tp": "pretraining_tp",
        "rms_norm_eps": "rms_norm_eps",
        "rope_scaling": "rope_scaling",
        "tie_word_embeddings": "tie_word_embeddings",
        "use_cache": "use_cache",
        "vocab_size": "vocab_size",
    }
    if nt_to_hf:
        return {nt: hf for hf, nt in hf_to_nt_map.items()}
    return hf_to_nt_map


def make_parallel_config(
    dp: int = 1,
    pp: int = 1,
    tp: int = 1,
):
    parallel_config = nanotron.config.ParallelismArgs(
        dp=dp,
        pp=pp,
        tp=tp,
        pp_engine=nanotron.config.AllForwardAllBackwardPipelineEngine(),
        tp_mode=nanotron.config.TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )
    return parallel_config


def load_nanotron_model(
    parallel_config: nanotron.config.ParallelismArgs = None,
    model_config: Optional[NanotronLlamaConfig] = None,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_path: Optional[Path] = None,
) -> LlamaForTraining:
    """
    Creates and returns a nanotron model.
    If `model_config` is None, then `checkpoint_path` must be set, in which case
    the configuration will be loaded from such path.
    If `checkpoint_path` is None, then `model_config` must be set, in which case
    the model created will have random weights.
    """

    if model_config is None:
        assert checkpoint_path is not None
        with open(checkpoint_path / "model_config.json") as f:
            model_config = NanotronLlamaConfig(**json.load(f))
    if parallel_config is None:
        parallel_config = make_parallel_config()
    parallel_context = nanotron.parallel.ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )
    nanotron_model = nanotron.models.build_model(
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
    if checkpoint_path is not None:
        nanotron.serialize.load_weights(
            model=nanotron_model, parallel_context=parallel_context, root_folder=checkpoint_path
        )
    return nanotron_model


def _handle_attention_block(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_q_heads: int, n_kv_heads: int, d_qk: int
) -> torch.Tensor:
    # Huggingface Llama separates the q, k, v weights (as opposed to nanotron).
    # Furthermore, in the rotary embeddings in nanotron expects interleaved pairs of even
    # and odd dimensions GPT-J style, while the huggingface implementation expects
    # the whole 1st half and then the whole 2nd half GPT-NeoX style (for more information
    # see flash_attn.layers.rotary.RotaryEmbedding).
    # This function handles the concatenation of the q, k, v weights and proper permutation
    # to ensure correct transformation.

    def interleave(w: torch.Tensor):
        w_new = []
        for head_w in w.split(d_qk):
            head_w = head_w.view(2, d_qk // 2, -1).transpose(0, 1).reshape(d_qk, -1)
            w_new.append(head_w)
        return torch.cat(w_new)

    q = interleave(q)
    k = interleave(k)
    return torch.cat([q, k, v])


def convert_hf_to_nt(model_hf: LlamaForCausalLM, model_nt: LlamaForTraining, config: NanotronLlamaConfig):
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


def get_nanotron_config(config: HFLlamaConfig) -> NanotronLlamaConfig:
    """Converts a huggingface configuration to nanotron configuration."""
    attrs = {key: getattr(config, value) for key, value in get_config_mapping(nt_to_hf=True).items()}
    return NanotronLlamaConfig(**attrs)


def convert_checkpoint_and_save(checkpoint_path: Path, save_path: Path, dp: int, pp: int, tp: int):
    """Loads the huggingface checkpoint in `checkpoint_path`, creates
    a new nanotron instance, copies the weights from the huggingface checkpoint
    and saves the transformed nanotron to `save_path`."""

    # Copy weights and save model.
    parallel_context = nanotron.parallel.ParallelContext(
        data_parallel_size=dp, pipeline_parallel_size=pp, tensor_parallel_size=tp
    )
    import torch.distributed as dist

    rank = dist.get_rank()

    print(f"[rank={rank}] inited parallel context \n\n")

    # Load huggingface.
    hf_model = LlamaForCausalLM.from_pretrained(checkpoint_path)

    # Init nanotron model.
    model_config = get_nanotron_config(hf_model.config)

    print(f"[rank={rank}] after model_config \n\n")

    nanotron_model = load_nanotron_model(model_config=model_config)

    print(f"[rank={rank}] after nanotron_model \n\n")

    convert_hf_to_nt(hf_model, nanotron_model, model_config)
    nanotron.serialize.save_weights(model=nanotron_model, parallel_context=parallel_context, root_folder=save_path)
    with open(save_path / "model_config.json", "w+") as f:
        json.dump(vars(model_config), f)
    parallel_config = make_parallel_config(dp=dp, pp=pp, tp=tp)
    with open(save_path / "config.yaml", "w") as f:
        config = Config(
            general=GeneralArgs(project="test", run="llama"),
            parallelism=parallel_config,
            model=ModelArgs(
                init_method=RandomInit(std=0.2),
                model_config=model_config,
            ),
            tokenizer=TokenizerArgs(checkpoint_path),
        )
        yaml.dump(config.as_dict(), f)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    import os

    os.environ["TRANSFORMERS_CACHE"] = "/admin/home/phuc_nguyen/.cache/huggingface/hub"

    checkpoint_path = "meta-llama/Meta-Llama-3-8B"
    save_path = Path("/fsx/nouamane/projects/nanotron/pretrained/llama3-8B-interleave")

    convert_checkpoint_and_save(checkpoint_path=checkpoint_path, save_path=save_path, dp=1, pp=1, tp=1)
    # torchrun --nproc_per_node=1 examples/infinite-context-length/llama3_to_nanotron_ckp.py
