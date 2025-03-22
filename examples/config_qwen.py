""" Example python script to generate a YAML config file which can be used to run a training with nanotron. Refer to "examples" section in the `/README.md` for more information."""
import argparse
import os
from typing import Dict, Tuple

from nanotron.config import (
    AdamWOptimizerArgs,
    CheckpointsArgs,
    Config,
    DataArgs,
    DatasetStageArgs,
    GeneralArgs,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    OptimizerArgs,
    ParallelismArgs,
    Qwen2Config,
    RandomInit,
    SFTDatasetsArgs,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.logging import human_format

MODEL_SIZES: Dict[str, Tuple[int, int, int, int, int]] = {
    # (layers, hidden, heads, kv_heads, ffn_size)
    "160m": (12, 768, 12, 12, 3072),  # ~160M params
    "410m": (24, 1024, 16, 16, 4096),  # ~410M params
    # Small to medium models
    "1b": (16, 2048, 16, 16, 5632),  # ~1B params
    "3b": (28, 3072, 32, 32, 8192),  # ~3B params
    # Standard sizes
    "7b": (32, 4096, 32, 32, 11008),  # ~7B params
    "13b": (40, 5120, 40, 40, 13824),  # ~13B params
    # Large models
    "30b": (60, 6656, 52, 52, 17920),  # ~30B params
    "70b": (80, 8192, 64, 8, 28672),  # ~70B params (MQA)
    "custom": (12, 192, 4, 4, 768),
}


def get_model_config(model_size: str) -> Qwen2Config:
    """Get a Qwen2Config for the specified model size."""
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Model size '{model_size}' not found. Available sizes: {list(MODEL_SIZES.keys())}")

    layers, hidden, heads, kv_heads, intermediate = MODEL_SIZES[model_size]

    return Qwen2Config(
        # Config for a tiny model model with MoE layers
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=hidden,
        initializer_range=0.02,
        intermediate_size=intermediate,
        max_position_embeddings=256,
        num_attention_heads=heads,
        num_hidden_layers=layers,
        num_key_value_heads=kv_heads,
        pretraining_tp=1,
        rms_norm_eps=1e-05,
        rope_scaling=None,
        tie_word_embeddings=True,
        use_cache=True,
        vocab_size=32000,  # Standard Llama tokenizer vocab size
        is_qwen2_config=True,
        pad_token_id=None,
    )


def calculate_parameters(model_config: Qwen2Config) -> str:
    """Calculate the number of parameters in a model."""
    params = model_config.vocab_size * model_config.hidden_size * 2 + model_config.num_hidden_layers * (  # Embeddings
        3 * model_config.hidden_size * model_config.intermediate_size  # MLP
        + 4 * model_config.hidden_size * model_config.hidden_size  # Attention
    )
    return human_format(params)


seed = 42

learning_rate = LRSchedulerArgs(
    learning_rate=3e-4, lr_warmup_steps=2, lr_warmup_style="linear", lr_decay_style="cosine", min_decay_lr=1e-5
)

optimizer = OptimizerArgs(
    zero_stage=0,
    weight_decay=0.01,
    clip_grad=1.0,
    accumulate_grad_in_fp32=True,
    learning_rate_scheduler=learning_rate,
    optimizer_factory=AdamWOptimizerArgs(
        adam_eps=1e-08,
        adam_beta1=0.9,
        adam_beta2=0.95,
        torch_adam_is_fused=True,
    ),
)

parallelism = ParallelismArgs(
    dp=2,
    pp=1,
    tp=1,
    context_parallel_size=1,
    expert_parallel_size=1,
    pp_engine="1f1b",
    tp_mode="REDUCE_SCATTER",
    tp_linear_async_communication=True,
    moe_layer_recompute=True,  # Add recompute option for MoE layers
)

tokens = TokensArgs(sequence_length=256, train_steps=15, micro_batch_size=2, batch_accumulation_per_replica=1)

data_stages = [
    DatasetStageArgs(
        name="Stable Training Stage",
        start_training_step=1,
        data=DataArgs(
            # For pretraining:
            # dataset=PretrainDatasetsArgs(
            #     hf_dataset_or_datasets="trl-lib/tldr",
            #     text_column_name="text",
            # ),
            # For SFT (uncomment to use):
            dataset=SFTDatasetsArgs(
                hf_dataset_or_datasets="trl-lib/tldr",
                hf_dataset_splits="train",
                debug_max_samples=1000,
            ),
            seed=seed,
        ),
    ),
]

checkpoints_path = "./checkpoints"
os.makedirs(checkpoints_path, exist_ok=True)


def create_config(model_config: Qwen2Config, output_dir: str) -> Config:
    # Determine run name based on whether MoE is enabled
    run_name = "tiny_qwen_moe_%date_%jobid" if model_config.is_moe_model else "tiny_qwen_%date_%jobid"

    return Config(
        general=GeneralArgs(project="debug", run=run_name, seed=seed, ignore_sanity_checks=False),
        checkpoints=CheckpointsArgs(checkpoints_path=checkpoints_path, checkpoint_interval=10),
        parallelism=parallelism,
        model=ModelArgs(init_method=RandomInit(std=0.025), model_config=model_config),
        tokenizer=TokenizerArgs("robot-test/dummy-tokenizer-wordlevel"),
        optimizer=optimizer,
        logging=LoggingArgs(),
        tokens=tokens,
        data_stages=data_stages,
        # profiler=ProfilerArgs(profiler_export_path="./tb_logs"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Qwen model configuration YAML file")
    parser.add_argument(
        "--model_size",
        choices=MODEL_SIZES.keys(),
        default="custom",
        help="Model size to generate config for (e.g., 7b, 13b)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.path.dirname(__file__), help="Output directory for the config file"
    )
    parser.add_argument("--run_name", type=str, default="qwen_%date_%jobid", help="Run name for the config file")
    parser.add_argument("--dp", type=int, default=2, help="Data parallel size")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--cp", type=int, default=1, help="Context parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size")
    args = parser.parse_args()

    model_config = get_model_config(args.model_size)
    num_params = calculate_parameters(model_config)
    print(f"Model has {num_params} parameters using --model_size {args.model_size}")
    config = create_config(model_config, args.output_dir)

    # Save config as YAML file - name depends on whether MoE is enabled
    config_path = f"{args.output_dir}/config_qwen.yaml"
    config.save_as_yaml(config_path)

    # You can now train a model with this config using
    print("You can now train a model with this config using:")
    world_size = args.dp * args.tp * args.pp * args.cp
    if world_size <= 8:
        print(
            f"CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node={world_size} run_train.py --config-file {config_path}"
        )
    else:
        print("Checkout slurm_launcher.py to launch a multi-node job")
