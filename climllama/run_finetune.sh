#!/bin/bash

# Script to run finetuning with the generated config
# Make sure you've run prepare_finetune_config.sh first to generate the config file

CONFIG_FILE=climllama/config_finetune.yaml

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    echo "Please run prepare_finetune_config.sh first to generate the config."
    exit 1
fi

# Activate environment
source nanotron/bin/activate

# Set environment variables for distributed training
export CUDA_DEVICE_MAX_CONNECTIONS=1
export FI_PROVIDER="efa"  # For AWS EFA networking, remove if not using EFA

# Calculate number of GPUs needed (DP * TP * PP)
# You should adjust this based on your config
NPROC=8  # Default: DP=4, TP=2, PP=1 -> 8 GPUs

echo "Starting finetuning with config: $CONFIG_FILE"
echo "Using $NPROC GPUs"
echo ""

# Run training
torchrun --nproc_per_node=$NPROC run_train.py --config-file $CONFIG_FILE
