#!/bin/bash
#SBATCH --job-name=climllama_finetune
#SBATCH -A a122
#SBATCH --mem=260000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --time=12:00:00
#SBATCH --output=logs/finetune_%j.out

# Script to run finetuning with the generated config
# Make sure you've run prepare_finetune_config.sh first to generate the config file

set -euo pipefail

CONFIG_FILE=/capstor/scratch/cscs/lhuang/nanotron_climllama/climllama/config_finetune.yaml
BASE_PATH=/capstor/scratch/cscs/lhuang/nanotron_climllama/climllama

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    echo "Please run prepare_finetune_config.sh first to generate the config."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate environment
source $BASE_PATH/.venv/bin/activate

export http_proxy=http://proxy.cscs.ch:8080
export https_proxy=http://proxy.cscs.ch:8080

echo "Starting finetuning with config: $CONFIG_FILE"
echo "Using $NPROC GPUs"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

# Run training
srun -A a122 --environment=cuda129_ub2404 \
 numactl --membind=0-3 torchrun \
 --nproc_per_node=4 \
 --nnodes=$SLURM_NNODES \
 --start-method forkserver \
 $BASE_PATH/run_train.py --config-file $CONFIG_FILE
