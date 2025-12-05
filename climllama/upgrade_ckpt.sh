#!/bin/bash

# Script to upgrade a Qwen2 checkpoint to ClimLlama format
# This adds the climate-specific position embeddings (variable, resolution, leadtime)
set -euo pipefail

# Source Qwen2 checkpoint path
QWEN2_CKPT_PATH=/iopsstor/scratch/cscs/lhuang/FoundationModel/outputs/megatron_checkpoints/exp_fsq_245_split_vocab32768/llama_3B_vocab_32768/iter_0204000_nanotron

# ClimLlama training config (contains model configuration with position embeddings)
CONFIG_PATH=climllama/config_finetune_with_pe.yaml

# Output path for the upgraded ClimLlama checkpoint
SAVE_PATH=/iopsstor/scratch/cscs/lhuang/FoundationModel/outputs/megatron_checkpoints/exp_fsq_245_split_vocab32768/llama_3B_vocab_32768/iter_0204000_nanotron_climllama

# Tensor parallel size (should match the checkpoint's TP size)
TP_SIZE=1

# Activate environment
source .venv/bin/activate

echo "Upgrading Qwen2 checkpoint to ClimLlama format..."
echo "  Source: $QWEN2_CKPT_PATH"
echo "  Config: $CONFIG_PATH"
echo "  Output: $SAVE_PATH"
echo ""

# Run the upgrade script
torchrun --nproc_per_node=$TP_SIZE climllama/upgrade_ckpt_to_climllama.py \
    --qwen2_ckpt_path "$QWEN2_CKPT_PATH" \
    --config_path "$CONFIG_PATH" \
    --save_path "$SAVE_PATH" \
    --tp_size $TP_SIZE

echo ""
echo "Checkpoint upgrade complete!"
echo "ClimLlama checkpoint saved to: $SAVE_PATH"
echo ""