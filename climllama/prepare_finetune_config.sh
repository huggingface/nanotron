#!/bin/bash

# Configuration script to prepare finetuning config from converted checkpoint
# This script demonstrates how to use prepare_training_config.py with Megatron IndexedDataset
set -euo pipefail
# Paths from your conversion script
CKPT_PATH_NT=/iopsstor/scratch/cscs/lhuang/FoundationModel/outputs/megatron_checkpoints/exp_fsq_245_split_vocab32768/llama_3B_vocab_32768/iter_0204000_nanotron
TOKENIZER_PATH=/iopsstor/scratch/cscs/lhuang/FoundationModel/outputs/megatron_datasets/245-fsq_2025-02-07-14-06-33-00230000-split/pretrained_tokenizer

# Output config file
OUTPUT_CONFIG=climllama/config_finetune.yaml

# Training settings - now using Megatron IndexedDataset
# DATA_PREFIX should point to your .bin/.idx dataset files (without extension)
# Supports wildcards: "/path/data_*" or "/path/*" to match multiple files
# Example: "/path/data_*" will match data_001.bin, data_002.bin, etc.
DATA_PREFIX="/iopsstor/scratch/cscs/lhuang/FoundationModel/outputs/megatron_datasets/245-fsq_2025-02-07-14-06-33-00230000-split/*"
TRAIN_STEPS=5000
LEARNING_RATE=1e-5
SEQUENCE_LENGTH=4096

# IndexedDataset specific settings
SPLITS_STRING="969,30,1"  # Train/val/test split ratios
INDEX_MAPPING_DIR=""  # Optional: directory to cache index mappings
SKIP_WARMUP=""  # Optional: add "--skip_warmup" to skip warmup

# Parallelism settings (adjust based on your GPU count)
MICRO_BATCH_SIZE=8
DP=2  # Data parallel
TP=4  # Tensor parallel
PP=1  # Pipeline parallel
GRAD_ACCUMULATION_STEPS=8  # Gradient accumulation steps
ZERO_STAGE=0  # ZeRO optimizer stage

# Activate environment
source .venv/bin/activate

# Generate config
python climllama/prepare_training_config.py \
    --checkpoint_path $CKPT_PATH_NT \
    --tokenizer_path $TOKENIZER_PATH \
    --output_config $OUTPUT_CONFIG \
    --mode finetune \
    --data_prefix "$DATA_PREFIX" \
    --train_steps $TRAIN_STEPS \
    --learning_rate $LEARNING_RATE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --sequence_length $SEQUENCE_LENGTH \
    --dp $DP \
    --tp $TP \
    --pp $PP \
    --batch_accumulation $GRAD_ACCUMULATION_STEPS \
    --zero_stage $ZERO_STAGE \
    --enable_wandb \
    --splits_string $SPLITS_STRING \
    ${INDEX_MAPPING_DIR:+--index_mapping_dir $INDEX_MAPPING_DIR} \
    $SKIP_WARMUP

echo ""
echo "Config generated successfully!"
echo "To start training, run:"
echo "  sbatch climllama/run_finetune.sh"
