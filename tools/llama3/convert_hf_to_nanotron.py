"""
torchrun --nproc-per-node 1 tools/llama3/convert_hf_to_nanotron.py --nanotron-checkpoint-path nanotron_checkpoints/NanotronLlama38B --pretrained-model-name-or-path meta-llama/Meta-Llama-3-8B-Instruct
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from nanotron.config import Config, GeneralArgs, ModelArgs, ParallelismArgs, TokenizerArgs
from nanotron.config.models_config import ExistingCheckpointInit
from nanotron.config.models_config import LlamaConfig as LlamaConfigNanotron
from nanotron.models import build_model
from nanotron.models.llama import LlamaForTraining
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.pipeline_parallel.engine import AllForwardAllBackwardPipelineEngine
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode
from nanotron.serialize import TrainingMetadata, save_meta, save_weights
from nanotron.serialize.metadata import DataStageMetadata
from nanotron.trainer import mark_tied_parameters
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cpu")
TORCH_DTYPE = torch.bfloat16


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="Nanotron Model")
    group.add_argument(
        "--nanotron-checkpoint-path",
        type=str,
        required=True,
        help="A path to a directory to store the converted Nanotron Checkpoint",
    )

    group = parser.add_argument_group(title="HuggingFace Model")
    group.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing model weights saved using save_pretrained() or the model id of a pretrained model hosted inside a model repo on the Hugging Face Hub",
    )

    args = parser.parse_args()

    return args


def main(args):
    # Init Nanotron Parallel Utilities
    parallel_config = ParallelismArgs(
        dp=1,
        pp=1,
        tp=1,
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

    # Load Llama3-8B HF model
    print(f"Loading pretrained Llama3 Model: {args.pretrained_model_name_or_path}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=TORCH_DTYPE, attn_implementation="flash_attention_2"
    ).to(DEVICE)
    hf_config = hf_model.config

    # Set Nanotron LlamaConfig
    nanotron_llama_config = LlamaConfigNanotron(
        bos_token_id=hf_config.bos_token_id,
        eos_token_id=hf_config.eos_token_id,
        hidden_act=hf_config.hidden_act,
        hidden_size=hf_config.hidden_size,
        initializer_range=hf_config.initializer_range,
        intermediate_size=hf_config.intermediate_size,
        is_llama_config=True,
        max_position_embeddings=hf_config.max_position_embeddings,
        num_attention_heads=hf_config.num_attention_heads,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_key_value_heads=hf_config.num_key_value_heads,
        pad_token_id=None,
        pretraining_tp=hf_config.pretraining_tp,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_scaling=hf_config.rope_scaling,
        rope_theta=hf_config.rope_theta,
        rope_interleaved=False,
        tie_word_embeddings=hf_config.tie_word_embeddings,
        use_cache=hf_config.use_cache,
        vocab_size=hf_config.vocab_size,
    )

    # Init Llama3-8B Nanotron model
    print("Init empty Nanotron Llama3 Model")
    nanotron_model = build_model(
        model_builder=lambda: LlamaForTraining(
            config=nanotron_llama_config,
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

    # Copy params from HF to Nanotron
    print("Copyng weights from HF model to Nanotron model...")
    # Token embeddings
    print("Copyng Token Embeddings...")
    assert (
        nanotron_model.model.token_position_embeddings.pp_block.token_embedding.weight.shape
        == hf_model.model.embed_tokens.weight.shape
    )
    with torch.no_grad():
        nanotron_model.model.token_position_embeddings.pp_block.token_embedding.weight.copy_(
            hf_model.model.embed_tokens.weight
        )

    # Decoder layers
    for i in tqdm(
        range(nanotron_llama_config.num_hidden_layers),
        desc="Copyng Hidden Layers",
        total=nanotron_llama_config.num_hidden_layers,
    ):
        # Input layer norm
        assert (
            hf_model.model.layers[i].input_layernorm.weight.shape
            == nanotron_model.model.decoder[i].pp_block.input_layernorm.weight.shape
        )
        with torch.no_grad():
            nanotron_model.model.decoder[i].pp_block.input_layernorm.weight.copy_(
                hf_model.model.layers[i].input_layernorm.weight
            )

        # Self attn
        ## QKV
        tmp_qkv_proj = torch.cat(
            [
                hf_model.model.layers[i].self_attn.q_proj.weight,
                hf_model.model.layers[i].self_attn.k_proj.weight,
                hf_model.model.layers[i].self_attn.v_proj.weight,
            ],
            dim=0,
        )
        assert tmp_qkv_proj.shape == nanotron_model.model.decoder[i].pp_block.attn.qkv_proj.weight.shape
        with torch.no_grad():
            nanotron_model.model.decoder[i].pp_block.attn.qkv_proj.weight.copy_(tmp_qkv_proj)

        ## O
        assert (
            hf_model.model.layers[i].self_attn.o_proj.weight.shape
            == nanotron_model.model.decoder[i].pp_block.attn.o_proj.weight.shape
        )
        with torch.no_grad():
            nanotron_model.model.decoder[i].pp_block.attn.o_proj.weight.copy_(
                hf_model.model.layers[i].self_attn.o_proj.weight
            )

        # MLP
        ## Gate Up Proj
        tmp_gate_up_proj = torch.cat(
            [
                hf_model.model.layers[i].mlp.gate_proj.weight,
                hf_model.model.layers[i].mlp.up_proj.weight,
            ],
            dim=0,
        )

        assert tmp_gate_up_proj.shape == nanotron_model.model.decoder[i].pp_block.mlp.gate_up_proj.weight.shape
        with torch.no_grad():
            nanotron_model.model.decoder[i].pp_block.mlp.gate_up_proj.weight.copy_(tmp_gate_up_proj)

        ## Down Proj
        assert (
            hf_model.model.layers[i].mlp.down_proj.weight.shape
            == nanotron_model.model.decoder[i].pp_block.mlp.down_proj.weight.shape
        )
        with torch.no_grad():
            nanotron_model.model.decoder[i].pp_block.mlp.down_proj.weight.copy_(
                hf_model.model.layers[i].mlp.down_proj.weight
            )

        # Post attn layer norm
        assert (
            hf_model.model.layers[i].post_attention_layernorm.weight.shape
            == nanotron_model.model.decoder[i].pp_block.post_attention_layernorm.weight.shape
        )
        with torch.no_grad():
            nanotron_model.model.decoder[i].pp_block.post_attention_layernorm.weight.copy_(
                hf_model.model.layers[i].post_attention_layernorm.weight
            )

    # Last layer norm
    print("Copyng Final Layer Norm...")
    assert nanotron_model.model.final_layer_norm.pp_block.weight.shape == hf_model.model.norm.weight.shape
    with torch.no_grad():
        nanotron_model.model.final_layer_norm.pp_block.weight.copy_(hf_model.model.norm.weight)

    # LM_Head
    print("Copyng LM Head...")
    assert nanotron_model.model.lm_head.pp_block.weight.shape == hf_model.lm_head.weight.shape
    with torch.no_grad():
        nanotron_model.model.lm_head.pp_block.weight.copy_(hf_model.lm_head.weight)

    print("Copied weights from HF model to Nanotron model!")
    # Store weights
    nanotron_checkpoint_path = Path(args.nanotron_checkpoint_path)
    save_weights(model=nanotron_model, parallel_context=parallel_context, root_folder=nanotron_checkpoint_path)

    # Store metadata
    print("Storing Nanotron model Configs and Metadata!")
    training_metadata = TrainingMetadata(
        last_train_step=0,
        consumed_train_samples=0,
        data_stages=[DataStageMetadata(name="Empty", consumed_train_samples=0, start_training_step=0)],
    )
    save_meta(
        root_folder=nanotron_checkpoint_path, parallel_context=parallel_context, training_metadata=training_metadata
    )
    # Store Tokenizer into Nanotron Checkpoint folder
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer.save_pretrained(nanotron_checkpoint_path)

    # Store Config and Model Config files
    with open(nanotron_checkpoint_path / "config.yaml", "w") as f:
        config = Config(
            general=GeneralArgs(project="conversion", run="Llama3-8B"),
            parallelism=parallel_config,
            model=ModelArgs(
                init_method=ExistingCheckpointInit(nanotron_checkpoint_path),
                model_config=nanotron_llama_config,
            ),
            tokenizer=TokenizerArgs(nanotron_checkpoint_path),
        )
        print("Saving config ...")
        yaml.dump(config.as_dict(), f)

    with open(nanotron_checkpoint_path / "model_config.json", "w") as f:
        print("Saving model config ...")
        json.dump(asdict(nanotron_llama_config), f)

    print(f"Checkpoint conversion finished, check {args.nanotron_checkpoint_path}")


if __name__ == "__main__":
    _args = get_args()
    main(_args)
