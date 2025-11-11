"""Script to prepare a training configuration YAML file from a Nanotron checkpoint.

This script reads a Nanotron checkpoint, parses the model architecture, and generates
a training configuration YAML file ready for finetuning or continued pretraining using
Megatron/NeMo IndexedDataset format (.bin/.idx files).

Usage:
    python climllama/prepare_training_config.py \
        --checkpoint_path /path/to/nanotron/checkpoint \
        --data_prefix /path/to/indexed/dataset \
        --output_config config_train.yaml \
        --tokenizer_path /path/to/tokenizer \
        --mode pretrain

Arguments:
    --checkpoint_path: Path to the Nanotron checkpoint directory (required)
    --data_prefix: Path prefix to Megatron indexed dataset (.bin/.idx) (required)
    --output_config: Path to save the output YAML config (default: config_finetune.yaml)
    --tokenizer_path: HF tokenizer path or name (default: uses checkpoint path)
    --mode: Training mode - 'finetune' or 'pretrain' (default: finetune)
    --train_steps: Number of training steps (default: 5000)
    --learning_rate: Learning rate (default: 1e-5 for finetune, 3e-4 for pretrain)
    --micro_batch_size: Micro batch size (default: 2)
    --sequence_length: Sequence length (default: 4096)
    --dp: Data parallelism degree (default: 4)
    --tp: Tensor parallelism degree (default: 2)
    --pp: Pipeline parallelism degree (default: 1)
    --splits_string: Train/val/test split ratios (default: 969,30,1)
    --index_mapping_dir: Directory to cache index mappings (default: None)
    --skip_warmup: Skip warmup when building indexed dataset (default: False)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Union, List, Dict, Optional

from nanotron.config import (
    AdamWOptimizerArgs,
    CheckpointsArgs,
    Config,
    DataArgs,
    DatasetStageArgs,
    GeneralArgs,
    IndexedDatasetsArgs,
    LlamaConfig,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    OptimizerArgs,
    ParallelismArgs,
    RandomInit,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.constants import MODEL_CONFIG_FILE_NAME
from nanotron.logging import human_format


def load_model_config(checkpoint_path: str) -> LlamaConfig:
    """Load model configuration from checkpoint."""
    config_path = Path(checkpoint_path) / MODEL_CONFIG_FILE_NAME

    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config file not found at {config_path}. "
            f"Make sure {checkpoint_path} is a valid Nanotron checkpoint."
        )

    with open(config_path, "r") as f:
        model_config_dict = json.load(f)

    # Remove the is_llama_config flag before creating LlamaConfig
    model_config_dict.pop("is_llama_config", None)

    # Handle optional fields that might be None in the JSON
    if "_attn_implementation" in model_config_dict and model_config_dict["_attn_implementation"] is None:
        model_config_dict.pop("_attn_implementation")

    model_config = LlamaConfig(**model_config_dict)

    return model_config


def calculate_parameters(model_config: LlamaConfig) -> int:
    """Calculate approximate number of parameters."""
    # Base parameters (embeddings)
    base_params = model_config.vocab_size * model_config.hidden_size * 2

    # FFN parameters
    ffn_params = model_config.num_hidden_layers * (3 * model_config.hidden_size * model_config.intermediate_size)

    # Attention parameters
    attn_params = model_config.num_hidden_layers * (4 * model_config.hidden_size * model_config.hidden_size)

    # Total parameters
    total_params = base_params + ffn_params + attn_params

    return total_params


def parse_data_prefix(data_prefix_str: str) -> List:
    """Parse data_prefix string into proper format for IndexedDatasetsArgs.

    Supports:
    - Single path: "/path/to/dataset" -> ["/path/to/dataset"]
    - Multiple paths with weights (comma-separated): "0.6,/path/to/dataset1,0.4,/path/to/dataset2"
      -> [0.6, "/path/to/dataset1", 0.4, "/path/to/dataset2"]
    - Simple list (comma-separated): "/path/to/dataset1,/path/to/dataset2"
      -> ["/path/to/dataset1", "/path/to/dataset2"]
    """
    if "," in data_prefix_str:
        parts = [p.strip() for p in data_prefix_str.split(",")]
        # Check if it's a weighted format (alternating weights and paths)
        try:
            # Try to parse as weighted format: weight1,path1,weight2,path2,...
            result: List = []
            for part in parts:
                try:
                    # Try to convert to float (weight)
                    result.append(float(part))
                except ValueError:
                    # It's a path string
                    result.append(part)
            return result
        except Exception:
            # If parsing fails, return as list of paths
            return parts
    else:
        # Single path, but IndexedDatasetsArgs expects List, so wrap it
        return [data_prefix_str]


def create_training_config(
    checkpoint_path: str,
    tokenizer_path: str,
    data_prefix: str,
    mode: str = "pretrain",
    train_steps: int = 5000,
    learning_rate: Optional[float] = None,
    micro_batch_size: int = 2,
    sequence_length: int = 4096,
    dp: int = 4,
    tp: int = 2,
    pp: int = 1,
    batch_accumulation: int = 1,
    checkpoint_interval: int = 500,
    seed: int = 42,
    splits_string: str = "969,30,1",
    index_mapping_dir: Optional[str] = None,
    skip_warmup: bool = False,
) -> Config:
    """Create a training configuration from checkpoint."""

    # Load model config from checkpoint
    model_config = load_model_config(checkpoint_path)

    # Calculate parameters for logging
    total_params = calculate_parameters(model_config)
    num_params = human_format(total_params).replace(".", "p")
    print(f"Model configuration loaded: {num_params} parameters")
    print(f"  - Hidden size: {model_config.hidden_size}")
    print(f"  - Num layers: {model_config.num_hidden_layers}")
    print(f"  - Num attention heads: {model_config.num_attention_heads}")
    print(f"  - Intermediate size: {model_config.intermediate_size}")
    print(f"  - Vocab size: {model_config.vocab_size}")
    print(f"  - Max position embeddings: {model_config.max_position_embeddings}")

    # Set default learning rate based on mode
    if learning_rate is None:
        learning_rate = 1e-5 if mode == "finetune" else 3e-4

    # Learning rate scheduler
    lr_scheduler = LRSchedulerArgs(
        learning_rate=learning_rate,
        lr_warmup_steps=100,
        lr_warmup_style="linear",
        lr_decay_style="cosine",
        min_decay_lr=learning_rate * 0.1,  # 10% of initial LR
    )

    # Optimizer configuration
    optimizer = OptimizerArgs(
        zero_stage=0,
        weight_decay=0.01,
        clip_grad=1.0,
        accumulate_grad_in_fp32=True,
        learning_rate_scheduler=lr_scheduler,
        optimizer_factory=AdamWOptimizerArgs(
            adam_eps=1e-08,
            adam_beta1=0.9,
            adam_beta2=0.95,
            torch_adam_is_fused=True,
        ),
    )

    # Parallelism configuration
    parallelism = ParallelismArgs(
        dp=dp,
        pp=pp,
        tp=tp,
        context_parallel_size=1,
        pp_engine="1f1b",
        tp_mode="REDUCE_SCATTER",
        tp_linear_async_communication=True,
    )

    # Token/batch configuration
    tokens = TokensArgs(
        sequence_length=sequence_length,
        train_steps=train_steps,
        micro_batch_size=micro_batch_size,
        batch_accumulation_per_replica=batch_accumulation,
    )

    # Data configuration using Megatron IndexedDataset
    # Parse data_prefix to support single path or weighted blending
    parsed_data_prefix = parse_data_prefix(data_prefix)

    data_stages = [
        DatasetStageArgs(
            name="Training Stage",
            start_training_step=1,
            data=DataArgs(
                dataset=IndexedDatasetsArgs(
                    data_prefix=parsed_data_prefix,
                    splits_string=splits_string,
                    validation_drop_last=True,
                    eod_mask_loss=False,
                    no_seqlen_plus_one_input_tokens=False,
                    index_mapping_dir=index_mapping_dir,
                    skip_warmup=skip_warmup,
                ),
                seed=seed,
            ),
        ),
    ]

    # Checkpoint configuration
    checkpoints_path = os.path.join(os.path.dirname(checkpoint_path), "finetuned_checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)

    run_name = f"{mode}_%date_%jobid"

    # Create full config
    config = Config(
        general=GeneralArgs(
            project=f"llama_{mode}",
            run=run_name,
            seed=seed,
            ignore_sanity_checks=False,
        ),
        checkpoints=CheckpointsArgs(
            checkpoints_path=checkpoints_path,
            checkpoint_interval=checkpoint_interval,
            resume_checkpoint_path=checkpoint_path,
            load_lr_scheduler=False,  # Don't load old scheduler for finetuning
            load_optimizer=False,  # Don't load old optimizer for finetuning
        ),
        parallelism=parallelism,
        model=ModelArgs(
            init_method=RandomInit(std=0.025),
            model_config=model_config,
        ),
        tokenizer=TokenizerArgs(tokenizer_name_or_path=tokenizer_path),
        optimizer=optimizer,
        logging=LoggingArgs(),
        tokens=tokens,
        data_stages=data_stages,
    )

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate training config from Nanotron checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the Nanotron checkpoint directory",
    )

    parser.add_argument(
        "--output_config",
        type=str,
        default="config_finetune.yaml",
        help="Path to save the output YAML config (default: config_finetune.yaml)",
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="HF tokenizer path or name (default: uses checkpoint path)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["finetune", "pretrain"],
        default="finetune",
        help="Training mode (default: finetune)",
    )

    parser.add_argument(
        "--data_prefix",
        type=str,
        required=True,
        help="Path prefix to Megatron indexed dataset (.bin/.idx files) or list of paths with weights",
    )

    parser.add_argument(
        "--train_steps",
        type=int,
        default=5000,
        help="Number of training steps (default: 5000)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (default: 1e-5 for finetune, 3e-4 for pretrain)",
    )

    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=2,
        help="Micro batch size (default: 2)",
    )

    parser.add_argument(
        "--sequence_length",
        type=int,
        default=4096,
        help="Sequence length (default: 4096)",
    )

    parser.add_argument(
        "--dp",
        type=int,
        default=4,
        help="Data parallelism degree (default: 4)",
    )

    parser.add_argument(
        "--tp",
        type=int,
        default=2,
        help="Tensor parallelism degree (default: 2)",
    )

    parser.add_argument(
        "--pp",
        type=int,
        default=1,
        help="Pipeline parallelism degree (default: 1)",
    )

    parser.add_argument(
        "--batch_accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps per replica (default: 1)",
    )

    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--splits_string",
        type=str,
        default="969,30,1",
        help="Train/val/test split ratios (default: 969,30,1)",
    )

    parser.add_argument(
        "--index_mapping_dir",
        type=str,
        default=None,
        help="Directory to cache index mappings (default: None)",
    )

    parser.add_argument(
        "--skip_warmup",
        action="store_true",
        help="Skip warmup when building indexed dataset",
    )

    args = parser.parse_args()

    # Use checkpoint path as tokenizer path if not specified
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.checkpoint_path

    print(f"Creating {args.mode} configuration from checkpoint: {args.checkpoint_path}")
    print(f"Tokenizer path: {tokenizer_path}")
    print(f"Data prefix: {args.data_prefix}")
    print(f"Training steps: {args.train_steps}")
    print(f"Learning rate: {args.learning_rate if args.learning_rate else 'auto'}")
    print(f"Parallelism: DP={args.dp}, TP={args.tp}, PP={args.pp}")
    print(f"Splits: {args.splits_string}")
    print()

    # Create config
    config = create_training_config(
        checkpoint_path=args.checkpoint_path,
        tokenizer_path=tokenizer_path,
        data_prefix=args.data_prefix,
        mode=args.mode,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        micro_batch_size=args.micro_batch_size,
        sequence_length=args.sequence_length,
        dp=args.dp,
        tp=args.tp,
        pp=args.pp,
        batch_accumulation=args.batch_accumulation,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        splits_string=args.splits_string,
        index_mapping_dir=args.index_mapping_dir,
        skip_warmup=args.skip_warmup,
    )

    # Save config
    config.save_as_yaml(args.output_config)
    print(f"\n✓ Config saved to {args.output_config}")
    print(f"\nYou can now start training with:")
    print(f"  export CUDA_DEVICE_MAX_CONNECTIONS=1")
    print(f"  torchrun --nproc_per_node={args.dp * args.tp * args.pp} run_train.py --config-file {args.output_config}")


if __name__ == "__main__":
    main()
