"""
Convert a Qwen2 checkpoint to ClimLlama format.

This script loads a Qwen2 checkpoint, creates a ClimLlama model with additional
position embeddings (variable, resolution, leadtime), and saves the new checkpoint.
The original Qwen2 weights are preserved, and the new embedding layers are initialized.

Usage:
    torchrun --nproc_per_node=1 climllama/upgrade_ckpt_to_climllama.py \
        --qwen2_ckpt_path=/path/to/qwen2_checkpoint \
        --config_path=climllama/config_finetune_with_pe.yaml \
        --save_path=/path/to/climllama_checkpoint
"""

import argparse
import dataclasses
import json
from pathlib import Path

import torch
import yaml
from yaml import SafeLoader

import nanotron
from nanotron import logging
from nanotron.config import Config, get_config_from_file
from nanotron.config.models_config import ClimLlamaConfig, Qwen2Config
from nanotron.logging import log_rank
from nanotron.models import build_model
from nanotron.models.climllama import ClimLlamaForTraining
from nanotron.models.qwen import Qwen2ForTraining
from nanotron.parallel import ParallelContext
from nanotron.serialize.weights import load_weights, save_weights
from nanotron.trainer import mark_tied_parameters

logger = logging.get_logger(__name__)


def get_qwen2_config_from_checkpoint(ckpt_path: Path) -> Qwen2Config:
    """Load Qwen2 config from checkpoint's model_config.json."""
    config_file = ckpt_path / "model_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Could not find model_config.json in {ckpt_path}")

    with open(config_file) as f:
        config_dict = json.load(f)

    # Remove climllama-specific keys if present (for backwards compatibility)
    climllama_keys = [
        "is_climllama_config", "use_absolute_position_embeddings",
        "var_vocab_size", "variables", "res_vocab_size", "resolutions",
        "leadtime_vocab_size", "leadtimes", "use_spatial_temporal_encoding",
        "max_tp"
    ]
    for key in climllama_keys:
        config_dict.pop(key, None)

    return Qwen2Config(**config_dict)


def get_climllama_config_from_yaml(config_path: Path) -> ClimLlamaConfig:
    """Extract ClimLlamaConfig from training config YAML."""
    with open(config_path) as f:
        config_dict = yaml.load(f, Loader=SafeLoader)

    model_config_dict = config_dict.get("model", {}).get("model_config", {})
    return ClimLlamaConfig(**model_config_dict)


def build_parallel_context(tp_size: int = 1, pp_size: int = 1, dp_size: int = 1) -> ParallelContext:
    """Build a parallel context for model operations."""
    return nanotron.parallel.ParallelContext(
        data_parallel_size=dp_size,
        pipeline_parallel_size=pp_size,
        tensor_parallel_size=tp_size,
    )


def copy_qwen2_weights_to_climllama(
    qwen2_model: Qwen2ForTraining,
    climllama_model: ClimLlamaForTraining,
) -> None:
    """Copy weights from Qwen2 model to ClimLlama model.

    This copies all shared weights (token embeddings, decoder layers, etc.)
    from the Qwen2 model to the ClimLlama model. The additional ClimLlama
    embeddings (var, res, leadtime) are left as initialized.
    """
    qwen2_state = qwen2_model.state_dict()
    climllama_state = climllama_model.state_dict()

    # Map Qwen2 param names to ClimLlama param names
    # The main difference is in the embedding layer structure
    copied_count = 0
    skipped_new = []

    for name, param in climllama_state.items():
        # Handle embedding layer name mapping
        # Qwen2: model.token_position_embeddings.pp_block.token_embedding.weight
        # ClimLlama: model.token_position_embeddings.pp_block.token_embedding.weight (same)
        # ClimLlama also has: var_embedding, res_embedding, leadtime_embedding

        if name in qwen2_state:
            qwen2_param = qwen2_state[name]
            if param.shape == qwen2_param.shape:
                with torch.no_grad():
                    param.copy_(qwen2_param)
                copied_count += 1
            else:
                log_rank(
                    f"Shape mismatch for {name}: ClimLlama {param.shape} vs Qwen2 {qwen2_param.shape}",
                    logger=logger,
                    level=logging.WARNING,
                    rank=0,
                )
        else:
            skipped_new.append(name)

    log_rank(
        f"Copied {copied_count} parameters from Qwen2 to ClimLlama",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    if skipped_new:
        log_rank(
            f"Initialized {len(skipped_new)} new ClimLlama parameters: {skipped_new[:5]}...",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )


def upgrade_checkpoint(
    qwen2_ckpt_path: Path,
    config_path: Path,
    save_path: Path,
    tp_size: int = 1,
) -> None:
    """Upgrade a Qwen2 checkpoint to ClimLlama format.

    Args:
        qwen2_ckpt_path: Path to the source Qwen2 checkpoint
        config_path: Path to the ClimLlama training config YAML
        save_path: Path to save the new ClimLlama checkpoint
        tp_size: Tensor parallel size for loading/saving
    """
    # Build parallel context first (required for log_rank and distributed operations)
    print(f"Building parallel context with TP={tp_size}")
    parallel_context = build_parallel_context(tp_size=tp_size)

    log_rank(
        f"Loading ClimLlama config from {config_path}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    climllama_config = get_climllama_config_from_yaml(config_path)

    # Load the full training config to get parallel config
    training_config = get_config_from_file(str(config_path))
    parallel_config = training_config.parallelism

    # Build Qwen2 model for loading weights
    log_rank(
        f"Building Qwen2 model to load source checkpoint",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    # Create a Qwen2Config from the ClimLlama config (they share most fields)
    qwen2_config_dict = {
        k: v for k, v in dataclasses.asdict(climllama_config).items()
        if k in [f.name for f in dataclasses.fields(Qwen2Config)]
    }
    qwen2_config_dict["is_qwen2_config"] = True
    qwen2_config_dict.pop("is_climllama_config", None)
    qwen2_config = Qwen2Config(**qwen2_config_dict)

    # Get dtype - handle both string and torch.dtype
    model_dtype = training_config.model.dtype
    if isinstance(model_dtype, str):
        model_dtype = getattr(torch, model_dtype)

    qwen2_model = build_model(
        model_builder=lambda: Qwen2ForTraining(
            config=qwen2_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
        ),
        parallel_context=parallel_context,
        dtype=model_dtype,
        device=torch.device("cuda"),
    )
    mark_tied_parameters(model=qwen2_model, parallel_context=parallel_context, parallel_config=parallel_config)

    # Load Qwen2 checkpoint
    log_rank(
        f"Loading Qwen2 checkpoint from {qwen2_ckpt_path}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    load_weights(
        model=qwen2_model,
        parallel_context=parallel_context,
        root_folder=qwen2_ckpt_path,
    )

    # Build ClimLlama model
    log_rank(
        "Building ClimLlama model",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    climllama_model = build_model(
        model_builder=lambda: ClimLlamaForTraining(
            config=climllama_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
        ),
        parallel_context=parallel_context,
        dtype=model_dtype,
        device=torch.device("cuda"),
    )
    mark_tied_parameters(model=climllama_model, parallel_context=parallel_context, parallel_config=parallel_config)

    # Copy weights from Qwen2 to ClimLlama
    # Note: We don't call init_model_randomly() because:
    # 1. Shared weights (token embeddings, decoder layers, lm_head) will be copied from Qwen2
    # 2. New ClimLlama embeddings are already initialized during model construction
    log_rank(
        "Copying Qwen2 weights to ClimLlama model",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    copy_qwen2_weights_to_climllama(qwen2_model, climllama_model)

    # Save ClimLlama checkpoint
    log_rank(
        f"Saving ClimLlama checkpoint to {save_path}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    save_path.mkdir(parents=True, exist_ok=True)
    save_weights(
        model=climllama_model,
        parallel_context=parallel_context,
        root_folder=save_path,
    )

    # Save model config
    config_save_path = save_path / "model_config.json"
    with open(config_save_path, "w") as f:
        json.dump(dataclasses.asdict(climllama_config), f, indent=2)

    # Save training config YAML with updated checkpoint path
    config_base_name = config_path.stem  # e.g., "config_finetune_with_pe"
    yaml_save_path = config_path.parent / f"{config_base_name}_climllama.yaml"

    # Load original config and update checkpoint path
    with open(config_path) as f:
        training_config_dict = yaml.load(f, Loader=SafeLoader)

    # Update the resume checkpoint path to point to the new checkpoint
    if "checkpoints" in training_config_dict:
        training_config_dict["checkpoints"]["resume_checkpoint_path"] = str(save_path)

    # Save the updated config
    with open(yaml_save_path, "w") as f:
        yaml.dump(training_config_dict, f, default_flow_style=False, sort_keys=False)

    log_rank(
        f"Saved updated training config to {yaml_save_path}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    log_rank(
        f"Successfully saved ClimLlama checkpoint to {save_path}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Qwen2 checkpoint to ClimLlama format"
    )
    parser.add_argument(
        "--qwen2_ckpt_path",
        type=Path,
        required=True,
        help="Path to the source Qwen2 checkpoint",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        required=True,
        help="Path to the ClimLlama training config YAML",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        required=True,
        help="Path to save the new ClimLlama checkpoint",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)",
    )
    args = parser.parse_args()

    upgrade_checkpoint(
        qwen2_ckpt_path=args.qwen2_ckpt_path,
        config_path=args.config_path,
        save_path=args.save_path,
        tp_size=args.tp_size,
    )


if __name__ == "__main__":
    main()
