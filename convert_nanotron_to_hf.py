"""
torchrun --nproc-per-node 1 convert_nanotron_to_hf.py --tp 1 --nanotron-checkpoint-path n_c/second --hugging-face-checkpoint-path hf_c/second
"""
import argparse
import os
from dataclasses import asdict
from pathlib import Path

import torch
from nanotron.config import Config, ParallelismArgs, get_config_from_file
from nanotron.models import build_model
from nanotron.models.llama import LlamaForTraining
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.pipeline_parallel.engine import AllForwardAllBackwardPipelineEngine
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode
from nanotron.serialize import load_weights
from nanotron.trainer import mark_tied_parameters
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import LlamaConfig as LlamaConfigHF

# TODO Currentyly just sopporting Llama8B that doesn't needs any kind of parallelism
DP = 1
PP = 1


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="Nanotron Model")
    group.add_argument(
        "--nanotron-checkpoint-path",
        type=str,
        required=True,
        help="A path to a directory with a Nanotron Checkpoint",
    )

    group = parser.add_argument_group(title="Nanotron Parallelism")
    group.add_argument("--tp", type=int, required=True, help="Tensor Parallelism Degree of the Nanotron Checkpoint")

    group = parser.add_argument_group(title="HuggingFace Model")
    group.add_argument(
        "--hugging-face-checkpoint-path",
        type=str,
        required=True,
        help="A path to a directory to store the converted checkpoint",
    )
    # TODO Add push to hub

    args = parser.parse_args()

    return args


def main(args):
    # Load Nanotron checkpoint config
    nanotron_config = get_config_from_file(
        os.path.join(args.nanotron_checkpoint_path, "config.yaml"), config_class=Config, model_config_class=None
    )
    nanotron_llama_config = nanotron_config.model.model_config

    # Init Llama3-8B Nanotron model
    parallel_config = ParallelismArgs(
        dp=DP,
        pp=PP,
        tp=args.tp,
        pp_engine=AllForwardAllBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )
    assert (
        parallel_config.tp_mode == TensorParallelLinearMode.ALL_REDUCE
        and parallel_config.tp_linear_async_communication is False
    )

    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    nanotron_model = build_model(
        model_builder=lambda: LlamaForTraining(
            config=nanotron_config.model.model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=None,
        ),
        parallel_context=parallel_context,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )

    mark_tied_parameters(model=nanotron_model, parallel_context=parallel_context)
    sanity_check(root_module=nanotron_model)

    # Load Nanotron Checkpoint
    load_weights(
        model=nanotron_model, parallel_context=parallel_context, root_folder=Path(args.nanotron_checkpoint_path)
    )

    # Build empty HF Model
    ## TODO This takes pretty long time
    hf_model = AutoModelForCausalLM.from_config(
        config=LlamaConfigHF(**asdict(nanotron_llama_config)),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")

    # Copy params from Nanotron to HF
    # Token embeddings
    assert (
        nanotron_model.model.token_position_embeddings.pp_block.token_embedding.weight.shape
        == hf_model.model.embed_tokens.weight.shape
    )
    with torch.no_grad():
        hf_model.model.embed_tokens.weight.copy_(
            nanotron_model.model.token_position_embeddings.pp_block.token_embedding.weight
        )

    # Decoder layers
    for i in range(nanotron_config.model.model_config.num_hidden_layers):
        # Input layer norm
        assert (
            hf_model.model.layers[i].input_layernorm.weight.shape
            == nanotron_model.model.decoder[i].pp_block.input_layernorm.weight.shape
        )
        with torch.no_grad():
            hf_model.model.layers[i].input_layernorm.weight.copy_(
                nanotron_model.model.decoder[i].pp_block.input_layernorm.weight
            )

        # Self attn
        # Split Nanotrn qkv projection into q, k, v
        q, k, v = torch.split(
            nanotron_model.model.decoder[i].pp_block.attn.qkv_proj.weight,
            [
                nanotron_llama_config.num_attention_heads * nanotron_model.model.decoder[i].pp_block.attn.d_qk,
                nanotron_llama_config.num_key_value_heads * nanotron_model.model.decoder[i].pp_block.attn.d_qk,
                nanotron_llama_config.num_key_value_heads * nanotron_model.model.decoder[i].pp_block.attn.d_qk,
            ],
        )
        assert q.shape == hf_model.model.layers[i].self_attn.q_proj.weight.shape
        assert k.shape == hf_model.model.layers[i].self_attn.k_proj.weight.shape
        assert v.shape == hf_model.model.layers[i].self_attn.v_proj.weight.shape

        with torch.no_grad():
            hf_model.model.layers[i].self_attn.q_proj.weight.copy_(q)
            hf_model.model.layers[i].self_attn.k_proj.weight.copy_(k)
            hf_model.model.layers[i].self_attn.v_proj.weight.copy_(v)

        ## O
        assert (
            hf_model.model.layers[i].self_attn.o_proj.weight.shape
            == nanotron_model.model.decoder[i].pp_block.attn.o_proj.weight.shape
        )
        with torch.no_grad():
            hf_model.model.layers[i].self_attn.o_proj.weight.copy_(
                nanotron_model.model.decoder[i].pp_block.attn.o_proj.weight
            )

        # MLP
        ## Gate Up Proj
        gate_proj, up_proj = torch.split(
            nanotron_model.model.decoder[i].pp_block.mlp.gate_up_proj.weight,
            split_size_or_sections=[nanotron_llama_config.intermediate_size, nanotron_llama_config.intermediate_size],
        )
        assert gate_proj.shape == hf_model.model.layers[i].mlp.gate_proj.weight.shape
        assert up_proj.shape == hf_model.model.layers[i].mlp.up_proj.weight.shape

        with torch.no_grad():
            hf_model.model.layers[i].mlp.gate_proj.weight.copy_(gate_proj)
            hf_model.model.layers[i].mlp.up_proj.weight.copy_(up_proj)

        ## Down Proj
        assert (
            hf_model.model.layers[i].mlp.down_proj.weight.shape
            == nanotron_model.model.decoder[i].pp_block.mlp.down_proj.weight.shape
        )
        with torch.no_grad():
            hf_model.model.layers[i].mlp.down_proj.weight.copy_(
                nanotron_model.model.decoder[i].pp_block.mlp.down_proj.weight
            )

        # Post attn layer norm
        assert (
            hf_model.model.layers[i].post_attention_layernorm.weight.shape
            == nanotron_model.model.decoder[i].pp_block.post_attention_layernorm.weight.shape
        )
        with torch.no_grad():
            hf_model.model.layers[i].post_attention_layernorm.weight.copy_(
                nanotron_model.model.decoder[i].pp_block.post_attention_layernorm.weight
            )

    # Last layer norm
    assert nanotron_model.model.final_layer_norm.pp_block.weight.shape == hf_model.model.norm.weight.shape
    with torch.no_grad():
        hf_model.model.norm.weight.copy_(nanotron_model.model.final_layer_norm.pp_block.weight)

    # LM_Head
    assert nanotron_model.model.lm_head.pp_block.weight.shape == hf_model.lm_head.weight.shape
    with torch.no_grad():
        hf_model.lm_head.weight.copy_(nanotron_model.model.lm_head.pp_block.weight)

    # Store weights
    hf_model.save_pretrained(args.hugging_face_checkpoint_path, from_pt=True)
    # Store tokenizer
    tokenizer = AutoTokenizer.from_pretrained(nanotron_config.tokenizer.tokenizer_name_or_path)
    tokenizer.save_pretrained(args.hugging_face_checkpoint_path)


if __name__ == "__main__":
    _args = get_args()
    main(_args)
