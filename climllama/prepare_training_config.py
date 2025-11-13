"""Script to prepare a training configuration YAML file from a Nanotron checkpoint.

This script reads a Nanotron checkpoint, parses the model architecture, and generates
a training configuration YAML file ready for finetuning or continued pretraining using
Megatron/NeMo IndexedDataset format (.bin/.idx files).

Usage:
    # Single dataset
    python climllama/prepare_training_config.py \
        --checkpoint_path /path/to/nanotron/checkpoint \
        --data_prefix /path/to/indexed/dataset \
        --output_config config_train.yaml \
        --mode pretrain

    # Multiple datasets with wildcards
    python climllama/prepare_training_config.py \
        --checkpoint_path /path/to/checkpoint \
        --data_prefix "/path/to/data/*" \
        --output_config config_train.yaml

    # Weighted blending with wildcards
    python climllama/prepare_training_config.py \
        --checkpoint_path /path/to/checkpoint \
        --data_prefix "0.7,/path/data_*,0.3,/other/files_*" \
        --output_config config_train.yaml

    # Enable WandB logging
    python climllama/prepare_training_config.py \
        --checkpoint_path /path/to/checkpoint \
        --data_prefix /path/to/data \
        --output_config config_train.yaml \
        --enable_wandb \
        --wandb_project my_project \
        --wandb_entity my_team

Arguments:
    --checkpoint_path: Path to the Nanotron checkpoint directory (required)
    --data_prefix: Path prefix to Megatron indexed dataset (.bin/.idx) (required)
                   Supports wildcards: "/path/data_*" or "/path/*"
                   Supports weighted blending: "0.6,/path1,0.4,/path2"
    --output_config: Path to save the output YAML config (default: config_finetune.yaml)
    --tokenizer_path: HF tokenizer path or name (default: uses checkpoint path)
    --mode: Training mode - 'finetune' or 'pretrain' (default: finetune)
    --train_steps: Number of training steps (default: 5000)
    --learning_rate: Learning rate (default: 1e-5 for finetune, 3e-4 for pretrain)
    --micro_batch_size: Micro batch size (default: 4)
    --sequence_length: Sequence length (default: 4096)
    --dp: Data parallelism degree (default: 4)
    --tp: Tensor parallelism degree (default: 2)
    --pp: Pipeline parallelism degree (default: 1)
    --splits_string: Train/val/test split ratios (default: 969,30,1)
    --index_mapping_dir: Directory to cache index mappings (default: None)
    --skip_warmup: Skip warmup when building indexed dataset (default: False)
    --sampler_type: Megatron sampler type - 'sequential', 'random', or 'cyclic' (default: sequential)
    --pad_samples_to_global_batch_size: Pad last batch to global batch size (default: True)
    --enable_wandb: Enable Weights & Biases logging (default: False)
    --wandb_project: WandB project name (default: uses mode name)
    --wandb_entity: WandB entity/team name (default: None)
    --zero_stage: ZeRO optimizer stage - 0 (disabled), 1 (optimizer states), 2 (+ gradients), 3 (+ parameters) (default: 0)
"""

import argparse
import dataclasses
import glob
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

from nanotron.config import (
    AdamWOptimizerArgs,
    CheckpointsArgs,
    Config,
    DataArgs,
    DatasetStageArgs,
    GeneralArgs,
    IndexedDatasetsArgs,
    LoggingArgs,
    LRSchedulerArgs,
    MetricsLoggingArgs,
    ModelArgs,
    OptimizerArgs,
    ParallelismArgs,
    Qwen2Config,
    RandomInit,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.constants import MODEL_CONFIG_FILE_NAME
from nanotron.logging import human_format


def load_model_config(checkpoint_path: str) -> Qwen2Config:
    """Load model configuration from checkpoint.

    This function loads Qwen2Config and filters out any incompatible fields,
    making it robust to checkpoints from different model types.

    Args:
        checkpoint_path: Path to the checkpoint directory
    """
    config_path = Path(checkpoint_path) / MODEL_CONFIG_FILE_NAME

    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config file not found at {config_path}. "
            f"Make sure {checkpoint_path} is a valid Nanotron checkpoint."
        )

    with open(config_path, "r") as f:
        model_config_dict = json.load(f)

    # Always use Qwen2Config (hardcoded in nanotron)
    config_cls = Qwen2Config
    config_name = "Qwen2Config"

    print(f"Using config type: {config_name}")

    # Get all valid field names for the detected config using dataclass introspection
    valid_fields = {field.name for field in dataclasses.fields(config_cls)}

    # Filter out any fields that are not valid for the config
    filtered_config = {k: v for k, v in model_config_dict.items() if k in valid_fields}

    # Log any fields that were filtered out for debugging
    removed_fields = set(model_config_dict.keys()) - set(filtered_config.keys())
    if removed_fields:
        print(f"Note: Filtered out fields not in {config_name}: {sorted(removed_fields)}")

    # Handle optional fields that might be None in the JSON
    if "_attn_implementation" in filtered_config and filtered_config["_attn_implementation"] is None:
        filtered_config.pop("_attn_implementation")

    model_config = config_cls(**filtered_config)

    return model_config


def calculate_parameters(model_config: Qwen2Config) -> int:
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


def expand_wildcard_paths(path_pattern: str) -> List[str]:
    """Expand wildcard pattern to list of matching paths without extensions.

    For IndexedDataset files, we need to match .bin/.idx pairs and return the prefix.
    Example: "/path/data_*.bin" -> ["/path/data_1", "/path/data_2"]
    """
    # If the pattern ends with .bin or .idx, match those files and strip extension
    if path_pattern.endswith('.bin') or path_pattern.endswith('.idx'):
        matched_files = glob.glob(path_pattern)
        # Remove .bin/.idx extension to get the prefix
        return sorted(list(set([os.path.splitext(f)[0] for f in matched_files])))

    # If pattern contains wildcard but no extension, try matching with .bin
    if '*' in path_pattern and not path_pattern.endswith(('.bin', '.idx')):
        bin_pattern = path_pattern + '.bin' if not path_pattern.endswith('/') else path_pattern + '*.bin'
        matched_files = glob.glob(bin_pattern)
        if matched_files:
            return sorted(list(set([os.path.splitext(f)[0] for f in matched_files])))

    # If it's a directory pattern like "path/*", look for all .bin files
    if path_pattern.endswith('/*') or path_pattern.endswith('*'):
        bin_pattern = path_pattern + '.bin' if not path_pattern.endswith('*') else path_pattern.rstrip('*') + '*.bin'
        matched_files = glob.glob(bin_pattern)
        if matched_files:
            return sorted(list(set([os.path.splitext(f)[0] for f in matched_files])))

    # No wildcard or no matches, return as-is
    return [path_pattern]


def parse_data_prefix(data_prefix_str: str) -> List:
    """Parse data_prefix string into proper format for IndexedDatasetsArgs.

    Supports:
    - Single path: "/path/to/dataset" -> ["/path/to/dataset"]
    - Wildcard patterns: "/path/data_*" -> ["/path/data_1", "/path/data_2", ...]
    - Directory wildcard: "/path/*" -> ["/path/file1", "/path/file2", ...]
    - Multiple paths with weights (comma-separated): "0.6,/path/to/dataset1,0.4,/path/to/dataset2"
      -> [0.6, "/path/to/dataset1", 0.4, "/path/to/dataset2"]
    - Weighted with wildcards: "0.6,/path/data_*,0.4,/other/file_*"
    - Simple list (comma-separated): "/path/to/dataset1,/path/to/dataset2"
      -> ["/path/to/dataset1", "/path/to/dataset2"]
    """
    if "," in data_prefix_str:
        parts = [p.strip() for p in data_prefix_str.split(",")]
        # Check if it's a weighted format (alternating weights and paths)
        result: List = []
        for part in parts:
            try:
                # Try to convert to float (weight)
                result.append(float(part))
            except ValueError:
                # It's a path string, expand wildcards
                expanded = expand_wildcard_paths(part)
                if len(expanded) == 1:
                    result.append(expanded[0])
                else:
                    # If wildcard expands to multiple files, add them with equal weights
                    # Calculate per-file weight (divide the previous weight by number of files)
                    if result and isinstance(result[-1], float):
                        # We have a weight before this, split it among expanded files
                        weight = result.pop()
                        per_file_weight = weight / len(expanded)
                        for exp_path in expanded:
                            result.append(per_file_weight)
                            result.append(exp_path)
                    else:
                        # No weight specified, just add all paths
                        result.extend(expanded)
        return result
    else:
        # Single path or pattern, expand wildcards
        expanded = expand_wildcard_paths(data_prefix_str)
        return expanded


def create_training_config(
    checkpoint_path: str,
    tokenizer_path: str,
    data_prefix: str,
    mode: str = "pretrain",
    train_steps: int = 5000,
    learning_rate: Optional[float] = None,
    micro_batch_size: int = 4,
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
    sampler_type: str = "sequential",
    pad_samples_to_global_batch_size: bool = True,
    enable_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    zero_stage: int = 0,
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
        zero_stage=zero_stage,
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

    # Log the expanded dataset paths
    print(f"\nDataset configuration:")
    if isinstance(parsed_data_prefix, list):
        # Check if it's weighted format (alternating floats and strings)
        has_weights = any(isinstance(item, float) for item in parsed_data_prefix)

        # Configuration for auto-folding
        MAX_DISPLAY_ROWS = 10
        MAX_PATH_LENGTH = 100

        def truncate_path(path: str, max_length: int = MAX_PATH_LENGTH) -> str:
            """Truncate path if too long, showing start and end."""
            if len(path) <= max_length:
                return path
            keep_chars = (max_length - 5) // 2
            return f"{path[:keep_chars]} ... {path[-keep_chars:]}"

        def print_items(items, format_fn):
            """Print items with auto-folding if too many."""
            total = len(items)
            if total > MAX_DISPLAY_ROWS:
                show_first = MAX_DISPLAY_ROWS // 2
                show_last = MAX_DISPLAY_ROWS - show_first
                for item in items[:show_first]:
                    print(format_fn(item))
                print(f"    ... ({total - MAX_DISPLAY_ROWS} more datasets) ...")
                for item in items[-show_last:]:
                    print(format_fn(item))
            else:
                for item in items:
                    print(format_fn(item))

        if has_weights:
            # Collect all weighted entries
            weighted_entries = []
            i = 0
            while i < len(parsed_data_prefix):
                if isinstance(parsed_data_prefix[i], float):
                    weight = parsed_data_prefix[i]
                    path = parsed_data_prefix[i + 1] if i + 1 < len(parsed_data_prefix) else "unknown"
                    weighted_entries.append((weight, path))
                    i += 2
                else:
                    i += 1

            print(f"  Using weighted blending with {len(weighted_entries)} datasets:")
            print_items(weighted_entries, lambda x: f"    - {x[0]:.3f}: {truncate_path(x[1])}")
        else:
            print(f"  Using {len(parsed_data_prefix)} dataset(s):")
            print_items(parsed_data_prefix, lambda path: f"    - {truncate_path(path)}")

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
                    # Megatron sampler configuration
                    sampler_type=sampler_type,
                    pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
                ),
                seed=seed,
            ),
        ),
    ]

    # Checkpoint configuration
    checkpoints_path = os.path.join(os.path.dirname(checkpoint_path), "finetuned_checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)

    run_name = f"{mode}_%date_%jobid_%githash"

    # Set WandB project name
    if wandb_project is None:
        wandb_project = f"llama_{mode}"

    # Metrics logging configuration (for detailed wandb logging)
    metrics_logging = None
    if enable_wandb:
        metrics_logging = MetricsLoggingArgs(
            log_level=0,  # 0 = basic metrics, 1 = detailed per-layer metrics
            log_detail_interval=100,  # Log detailed metrics every N steps
        )
        print(f"\nWandB logging enabled:")
        print(f"  - Project: {wandb_project}")
        if wandb_entity:
            print(f"  - Entity: {wandb_entity}")
        print(f"  - Run name: {run_name}")

    # Create full config
    config = Config(
        general=GeneralArgs(
            project=wandb_project,
            run=run_name,
            seed=seed,
            ignore_sanity_checks=False,
            _expand_run_template=False,  # Don't expand when saving - delay until runtime
        ),
        checkpoints=CheckpointsArgs(
            checkpoints_path=checkpoints_path,
            checkpoint_interval=checkpoint_interval,
            resume_checkpoint_path=checkpoint_path,
            load_lr_scheduler=False,  # Don't load old scheduler for finetuning
            load_optimizer=False,  # Don't load old optimizer for finetuning
            checkpoints_path_is_shared_file_system=True,
        ),
        parallelism=parallelism,
        model=ModelArgs(
            init_method=RandomInit(std=0.025),
            model_config=model_config,
        ),
        tokenizer=TokenizerArgs(tokenizer_name_or_path=tokenizer_path),
        optimizer=optimizer,
        logging=LoggingArgs(
            iteration_step_info_interval=10,
        ),
        tokens=tokens,
        data_stages=data_stages,
        metrics_logging=metrics_logging,
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
        help="Path prefix to Megatron indexed dataset (.bin/.idx files). "
        "Supports wildcards (e.g., '/path/data_*' or '/path/*') and "
        "weighted blending (e.g., '0.6,/path1,0.4,/path2')",
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
        default=4,
        help="Micro batch size (default: 4)",
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

    parser.add_argument(
        "--sampler_type",
        type=str,
        choices=["sequential", "random", "cyclic"],
        default="sequential",
        help="Megatron sampler type: 'sequential' (deterministic), 'random' (shuffled), or 'cyclic' (default: sequential)",
    )

    parser.add_argument(
        "--pad_samples_to_global_batch_size",
        action="store_true",
        default=True,
        help="Pad last batch to global batch size (default: True)",
    )

    parser.add_argument(
        "--no_pad_samples_to_global_batch_size",
        dest="pad_samples_to_global_batch_size",
        action="store_false",
        help="Do not pad last batch to global batch size",
    )

    parser.add_argument(
        "--enable_wandb",
        action="store_true",
        help="Enable Weights & Biases logging (default: False)",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project name (default: llama_{mode})",
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB entity/team name (default: None)",
    )

    parser.add_argument(
        "--zero_stage",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="ZeRO optimizer stage: 0 (disabled), 1 (optimizer states), 2 (+ gradients), 3 (+ parameters) (default: 0)",
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
    print(f"ZeRO stage: {args.zero_stage}")
    print(f"Splits: {args.splits_string}")
    print(f"Megatron Sampler: {args.sampler_type} (pad_last_batch={args.pad_samples_to_global_batch_size})")
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
        sampler_type=args.sampler_type,
        pad_samples_to_global_batch_size=args.pad_samples_to_global_batch_size,
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        zero_stage=args.zero_stage,
    )

    # Check if output file exists and ask for permission to overwrite
    if os.path.exists(args.output_config):
        print(f"\n⚠️  Warning: File '{args.output_config}' already exists!")
        response = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Operation cancelled. No file was written.")
            return

    # Save config
    config.save_as_yaml(args.output_config)
    print(f"\n✓ Config saved to {args.output_config}")


if __name__ == "__main__":
    main()
