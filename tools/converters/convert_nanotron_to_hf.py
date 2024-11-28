"""
torchrun --nproc-per-node 1 tools/converters/convert_nanotron_to_hf.py --nanotron-checkpoint-path checkpoints/nanotron_pretrained_checkpoints/Nanotron-Llama-3.2-3B --hugging-face-checkpoint-path checkpoints/huggingface_converted/Converted-Nanotron-Llama-3.2-3B
"""
import argparse
import os
from dataclasses import asdict
from pathlib import Path

import torch
from nanotron import logging
from nanotron.config import Config, LoggingArgs, ParallelismArgs, get_config_from_file
from nanotron.logging import log_rank, set_ranks_logging_level
from nanotron.models import build_model
from nanotron.models.llama import LlamaForTraining
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.serialize import load_weights
from nanotron.trainer import mark_tied_parameters
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import LlamaConfig as LlamaConfigHF

logger = logging.get_logger(__name__)

DEVICE = torch.device("cpu")
TORCH_DTYPE = torch.bfloat16


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="Nanotron Model")
    group.add_argument(
        "--nanotron-checkpoint-path",
        type=str,
        required=True,
        help="A path to a directory with a Nanotron Checkpoint",
    )

    group = parser.add_argument_group(title="HuggingFace Model")
    group.add_argument(
        "--hugging-face-checkpoint-path",
        type=str,
        required=True,
        help="A path to a directory to store the converted checkpoint",
    )

    args = parser.parse_args()

    return args


def main(args):
    # Init Nanotron Parallel Utilities
    parallel_config = ParallelismArgs(dp=1, pp=1, tp=1)

    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    set_ranks_logging_level(parallel_context=parallel_context, logging_config=LoggingArgs())

    # Load Nanotron checkpoint config
    log_rank(
        f"Loading Nanotron checkpoint config file: {os.path.join(args.nanotron_checkpoint_path, 'config.yaml')}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    nanotron_config = get_config_from_file(
        os.path.join(args.nanotron_checkpoint_path, "config.yaml"), config_class=Config, model_config_class=None
    )
    nanotron_llama_config = nanotron_config.model.model_config

    # Init Llama3-8B Nanotron model
    log_rank("Init empty Nanotron Llama3 Model", logger=logger, level=logging.INFO, rank=0)

    nanotron_model = build_model(
        model_builder=lambda: LlamaForTraining(
            config=nanotron_config.model.model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=None,
        ),
        parallel_context=parallel_context,
        dtype=TORCH_DTYPE,
        device=DEVICE,
    )

    mark_tied_parameters(model=nanotron_model, parallel_context=parallel_context)
    sanity_check(root_module=nanotron_model)

    # Load Nanotron Checkpoint
    log_rank("Loading Nanotron Llama3 Model...", logger=logger, level=logging.INFO, rank=0)
    load_weights(
        model=nanotron_model, parallel_context=parallel_context, root_folder=Path(args.nanotron_checkpoint_path)
    )

    # Build empty HF Model
    log_rank("Init empty HF Llama3 Model", logger=logger, level=logging.INFO, rank=0)
    hf_model = AutoModelForCausalLM.from_config(  # WARN This takes a long time
        config=LlamaConfigHF(**asdict(nanotron_llama_config)),
        torch_dtype=TORCH_DTYPE,
        attn_implementation="flash_attention_2",
    ).to(DEVICE)

    # Copy params from Nanotron to HF
    log_rank("Copying weights from Nanotron model to HF model...", logger=logger, level=logging.INFO, rank=0)
    with torch.no_grad():
        # Token embeddings
        log_rank("Copying Token Embeddings...", logger=logger, level=logging.INFO, rank=0)
        assert (
            nanotron_model.model.token_position_embeddings.pp_block.token_embedding.weight.shape
            == hf_model.model.embed_tokens.weight.shape
        )
        hf_model.model.embed_tokens.weight.copy_(
                nanotron_model.model.token_position_embeddings.pp_block.token_embedding.weight
            )

        # Decoder layers
        for i in tqdm(
            range(nanotron_llama_config.num_hidden_layers),
            desc="Copying Hidden Layers",
            total=nanotron_llama_config.num_hidden_layers,
        ):
            # Input layer norm
            assert (
                hf_model.model.layers[i].input_layernorm.weight.shape
                == nanotron_model.model.decoder[i].pp_block.input_layernorm.weight.shape
            )
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

            hf_model.model.layers[i].self_attn.q_proj.weight.copy_(q)
            hf_model.model.layers[i].self_attn.k_proj.weight.copy_(k)
            hf_model.model.layers[i].self_attn.v_proj.weight.copy_(v)

            ## O
            assert (
                hf_model.model.layers[i].self_attn.o_proj.weight.shape
                == nanotron_model.model.decoder[i].pp_block.attn.o_proj.weight.shape
            )
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

            hf_model.model.layers[i].mlp.gate_proj.weight.copy_(gate_proj)
            hf_model.model.layers[i].mlp.up_proj.weight.copy_(up_proj)

            ## Down Proj
            assert (
                hf_model.model.layers[i].mlp.down_proj.weight.shape
                == nanotron_model.model.decoder[i].pp_block.mlp.down_proj.weight.shape
            )
            hf_model.model.layers[i].mlp.down_proj.weight.copy_(
                    nanotron_model.model.decoder[i].pp_block.mlp.down_proj.weight
                )

            # Post attn layer norm
            assert (
                hf_model.model.layers[i].post_attention_layernorm.weight.shape
                == nanotron_model.model.decoder[i].pp_block.post_attention_layernorm.weight.shape
            )
            hf_model.model.layers[i].post_attention_layernorm.weight.copy_(
                    nanotron_model.model.decoder[i].pp_block.post_attention_layernorm.weight
                )

        # Last layer norm
        log_rank("Copying Final Layer Norm...", logger=logger, level=logging.INFO, rank=0)
        assert nanotron_model.model.final_layer_norm.pp_block.weight.shape == hf_model.model.norm.weight.shape
        hf_model.model.norm.weight.copy_(nanotron_model.model.final_layer_norm.pp_block.weight)

        # LM_Head
        log_rank("Copying LM Head...", logger=logger, level=logging.INFO, rank=0)
        assert nanotron_model.model.lm_head.pp_block.weight.shape == hf_model.lm_head.weight.shape
        hf_model.lm_head.weight.copy_(nanotron_model.model.lm_head.pp_block.weight)

    log_rank("Copied weights from Nanotron model to HF model!", logger=logger, level=logging.INFO, rank=0)
    # Store weights
    log_rank("Storing HF model Checkpoint and Tokenizer!", logger=logger, level=logging.INFO, rank=0)
    hf_model.save_pretrained(args.hugging_face_checkpoint_path, from_pt=True)
    # Store tokenizer
    tokenizer = AutoTokenizer.from_pretrained(nanotron_config.tokenizer.tokenizer_name_or_path)
    tokenizer.save_pretrained(args.hugging_face_checkpoint_path)

    log_rank(
        f"Checkpoint conversion finished, check {args.hugging_face_checkpoint_path}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )


if __name__ == "__main__":
    _args = get_args()
    main(_args)