# ruff: noqa: E402
"""
This module converts

Command:
torchrun --nproc_per_node=1 examples/mixtral/convert_trfrs_to_nanotron.py --model_name  hf-internal-testing/Mixtral-tiny --save_path ./pretrained/mixtral
"""
import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch
from config_mixtral import MixtralConfig

sys.path.append(Path(__file__).parent.parent.as_posix())

import json

import nanotron.distributed as dist
from config_mixtral_tiny import CONFIG as CONFIG_NANOTRON
from config_mixtral_tiny import PARALLELISM as PARALLELISM_NANOTRON
from mixtral import MixtralForTraining
from nanotron.models import build_model
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.parameters import NanotronParameter, sanity_check
from nanotron.serialize import save
from nanotron.trainer import mark_tied_parameters
from transformers import MixtralForCausalLM


def get_args():
    parser = argparse.ArgumentParser(description="Convert transformers weights to nanotron weights")
    parser.add_argument("--model_name", type=str, default="hf-internal-testing/Mixtral-tiny")
    parser.add_argument("--save_path", type=str, default="pretrained/mixtral")
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    return parser.parse_args()


def permute_for_rotary(tensor, num_heads, per_head_hidden_size, hidden_size):
    return (
        tensor.view(num_heads, 2, per_head_hidden_size // 2, hidden_size)
        .transpose(1, 2)
        .contiguous()
        .view(num_heads * per_head_hidden_size, hidden_size)
    )


def get_transformers_weight(
    name: str, ref_module_state_dict: Dict[str, torch.Tensor], ref_module: MixtralForCausalLM, get_grad: bool = False
) -> torch.Tensor:
    """From our nanotron implementation, we get the equivalent tensor in transformers implementation"""
    config = ref_module.config
    nanotron_prefix = "model."
    assert name.startswith(nanotron_prefix)
    name = name[len(nanotron_prefix) :]

    path = name.split(".")
    path.remove("pp_block")
    name = ".".join(path)

    if get_grad is False:

        def get_tensor(path: str):
            return ref_module_state_dict[path]

        def get_tensors(path: List[str]):
            return [get_tensor(p) for p in path]

    else:

        def get_tensor(path: str):
            weight = ref_module.get_parameter(path)
            return weight.grad

        def get_tensors(path: List[str]):
            return [get_tensor(p) for p in path]

    if name == "token_position_embeddings.token_embedding.weight":
        return get_tensor("model.embed_tokens.weight")

    elif name == "lm_head.weight":
        # This only used when weights are not shared
        return get_tensor("lm_head.weight")

    elif name == "final_layer_norm.weight":
        return get_tensor("model.norm.weight")

    if path[0] == "decoder":
        transformer_path = ["model"] + ["layers"] + [path[1]]

        if path[2] == "attn":
            path[2] = "self_attn"

        if path[2] == "block_sparse_moe":
            if path[3] == "gate":
                return get_tensor(".".join(transformer_path + path[2:4] + path[5:]))

            if path[3] == "experts":
                path.remove("mlp"), path.remove("module")
                tensor_list = []
                for exp in range(config.num_local_experts):
                    weight = get_tensor(
                        ".".join(transformer_path + ["block_sparse_moe.experts"] + [str(exp)] + path[4:5] + ["weight"])
                    )
                    tensor_list.append(weight)
                return torch.cat(tensor_list, dim=0).T if "w2" not in name else torch.cat(tensor_list, dim=1).T

        if path[3] == "qkv_proj":
            proj_names = ["q_proj", "k_proj", "v_proj"]
            tensor_list = get_tensors(
                [".".join(transformer_path + path[2:3] + [proj_name] + path[4:]) for proj_name in proj_names]
            )
            # Permute q/k
            per_head_hidden_size = config.hidden_size // config.num_attention_heads
            # Permute q
            print(f"Permuting q {tensor_list[0].shape}")
            tensor_list[0] = permute_for_rotary(
                tensor=tensor_list[0],
                num_heads=config.num_attention_heads,
                per_head_hidden_size=per_head_hidden_size,
                hidden_size=config.hidden_size,
            )
            # Permute k
            print(f"Permuting k {tensor_list[1].shape}")
            tensor_list[1] = permute_for_rotary(
                tensor=tensor_list[1],
                num_heads=config.num_key_value_heads,
                per_head_hidden_size=per_head_hidden_size,
                hidden_size=config.hidden_size,
            )
            return torch.cat(tensor_list, dim=0)

        return get_tensor(".".join(transformer_path + path[2:]))

    else:
        raise ValueError(f"Couldn't find transformer equivalent of {name}")


def initialize_nanotron_model(dtype, parallel_context, parallel_config, model_config):
    model = build_model(
        model_builder=lambda: MixtralForTraining(
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=None,
        ),
        dtype=dtype,
        parallel_context=parallel_context,
        device=torch.device("cpu"),
    )
    return model


def fix_device_map_for_pp(model_config_nanotron, model, parallel_context):
    device_map = {}
    current_pp_rank = dist.get_rank(group=parallel_context.pp_pg)
    device_map["model.embed_tokens"] = (
        model.model.token_position_embeddings.rank
        if current_pp_rank == model.model.token_position_embeddings.rank
        else "meta"
    )
    for i in range(model_config_nanotron.num_hidden_layers):
        device_map[f"model.layers.{i}"] = (
            model.model.decoder[i].rank if current_pp_rank == model.model.decoder[i].rank else "meta"
        )
    device_map["model.norm"] = (
        model.model.final_layer_norm.rank if current_pp_rank == model.model.final_layer_norm.rank else "meta"
    )
    device_map["lm_head"] = model.model.lm_head.rank if current_pp_rank == model.model.lm_head.rank else "meta"


def convert_trfrs_to_nanotron(dp, pp, tp, model_name="huggyllama/llama-7b", save_path="pretrained/llama-7b"):
    # check save_path doesnt exist or is empty
    save_path = Path(save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    assert len(list(save_path.iterdir())) == 0, f"save_path {save_path} is not empty"

    parallel_config = PARALLELISM_NANOTRON

    parallel_config.dp = dp
    parallel_config.pp = pp
    parallel_config.tp = tp
    parallel_config.expert_parallel_size = 1

    # Initialise all process groups
    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
        expert_parallel_size=parallel_config.expert_parallel_size,
    )
    # params
    dtype = torch.bfloat16  # Flash attention doesn't support fp32

    # Initialise nanotron model
    nanotron_model_config = MixtralConfig.from_hf_config(model_name)
    model = initialize_nanotron_model(dtype, parallel_context, parallel_config, nanotron_model_config)

    # Initialise transformers model
    device_map = fix_device_map_for_pp(nanotron_model_config, model, parallel_context)
    model_ref = MixtralForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device_map)
    print(model)
    print(model_ref)
    # Copy weights from trfrs to nanotron
    ref_state_dict = model_ref.state_dict()
    for name, param in model.named_parameters():
        print(f"Syncing {name}")
        ref_param = get_transformers_weight(name=name, ref_module_state_dict=ref_state_dict, ref_module=model_ref)

        param_is_tp_sharded = (
            isinstance(param, NanotronParameter)
            and param.is_sharded
            and parallel_context.world_ranks_to_pg[param.get_sharded_info().global_ranks] == parallel_context.tp_pg
        )

        if param_is_tp_sharded:
            sharded_info = param.get_sharded_info()
            # copy param data (not just the reference)
            with torch.no_grad():
                for local_global_slices_pair in sharded_info.local_global_slices_pairs:
                    local_slices = local_global_slices_pair.local_slices
                    global_slices = local_global_slices_pair.global_slices
                    param[local_slices].copy_(ref_param[global_slices])
        else:
            assert (
                ref_param.shape == param.shape
            ), f"Parameter shape don't match for {name}\n{ref_param.shape} != {param.shape}"
            # copy param data (not just the reference)
            with torch.no_grad():
                param.copy_(ref_param)
                ref_param = None
                # torch.cuda.empty_cache()

    # TODO @nouamanetazi: assert weights are the same
    # Marks parameters as NanotronParameters
    mark_tied_parameters(model=model, parallel_context=parallel_context, parallel_config=parallel_config)

    sanity_check(root_module=model)

    checkpoint_metadata = {
        "last_train_step": 0,
        "consumed_train_samples": 0,
    }
    save(
        config=CONFIG_NANOTRON,
        model=model,
        optimizer=None,
        lr_scheduler=None,
        parallel_context=parallel_context,
        root_folder=save_path,
        should_save_config=False,
        should_save_optimizer=False,
        should_save_lr_scheduler=False,
        checkpoint_metadata=checkpoint_metadata,
        sanity_checks=False,
    )

    if dist.get_rank(parallel_context.world_pg) == 0:
        with open(save_path / "model_config.json", mode="w") as fo:
            fo.write(json.dumps(asdict(nanotron_model_config), indent=4))

    print(f"Model saved to {save_path}")
    print("You can test the model by running the following command:")
    print(f"torchrun --nproc_per_node=1 generate_mixtral.py --ckpt-path {save_path}")

    return model, model_ref


def ref_generate(model_ref):
    # Check model_ref outputs the same as model
    model_ref.eval()
    model_ref.to("cuda")
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_ref.config._name_or_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    text = "def fib(n)"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    print("Reference hf generation with prompt :", text)
    output = model_ref.generate(**inputs, max_new_tokens=128, do_sample=False)
    from pprint import pprint

    out = output[0][len(inputs.input_ids[0]) :]
    pprint(
        {
            "input": text,
            "generation": tokenizer.decode(out, clean_up_tokenization_spaces=False),
            "generation_ids": out,
        }
    )


def main():
    get_args()
    # model, model_ref = convert_trfrs_to_nanotron(**vars(args))
    # Ref generation
    model_ref = MixtralForCausalLM.from_pretrained(
        "hf-internal-testing/Mixtral-tiny", torch_dtype=torch.bfloat16, device_map={"": "cuda"}
    )
    ref_generate(model_ref)


if __name__ == "__main__":
    main()
