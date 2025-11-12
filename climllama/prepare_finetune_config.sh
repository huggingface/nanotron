#!/bin/bash

# Configuration script to prepare finetuning config from converted checkpoint
# This script demonstrates how to use prepare_training_config.py

# Paths from your conversion script
CKPT_PATH_NT=/iopsstor/scratch/cscs/lhuang/FoundationModel/outputs/megatron_checkpoints/exp_fsq_245_split_vocab32768/llama_3B_vocab_32768/iter_0204000_nanotron
TOKENIZER_PATH=/iopsstor/scratch/cscs/lhuang/FoundationModel/outputs/megatron_datasets/245-fsq_2025-02-07-14-06-33-00230000-split/pretrained_tokenizer

# Output config file
OUTPUT_CONFIG=climllama/config_finetune.yaml

# Training settings
DATASET="/iopsstor/scratch/cscs/lhuang/FoundationModel/outputs/megatron_datasets/245-fsq_2025-02-07-14-06-33-00230000-split"  # Change to your SFT dataset
TRAIN_STEPS=5000
LEARNING_RATE=1e-5
MICRO_BATCH_SIZE=2
SEQUENCE_LENGTH=4096

# Parallelism settings (adjust based on your GPU count)
DP=4  # Data parallel
TP=2  # Tensor parallel
PP=1  # Pipeline parallel

# Activate environment
source nanotron/bin/activate

# Generate config
python climllama/prepare_training_config.py \
    --checkpoint_path $CKPT_PATH_NT \
    --tokenizer_path $TOKENIZER_PATH \
    --output_config $OUTPUT_CONFIG \
    --mode finetune \
    --dataset $DATASET \
    --train_steps $TRAIN_STEPS \
    --learning_rate $LEARNING_RATE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --sequence_length $SEQUENCE_LENGTH \
    --dp $DP \
    --tp $TP \
    --pp $PP

echo ""
echo "Config generated successfully!"
echo "To start training, run:"
echo "  bash climllama/run_finetune.sh"
