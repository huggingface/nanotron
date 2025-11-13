#!/bin/bash
#SBATCH --job-name=climllama_finetune
#SBATCH -A a122
#SBATCH --mem=260000
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --time=12:00:00
#SBATCH --output=logs/finetune_%j.out

# Script to run finetuning with the generated config
# Make sure you've run prepare_finetune_config.sh first to generate the config file

set -euo pipefail

CONFIG_FILE=/capstor/scratch/cscs/lhuang/nanotron_climllama/climllama/config_finetune.yaml
WORKDIR=/capstor/scratch/cscs/lhuang/nanotron_climllama

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    echo "Please run prepare_finetune_config.sh first to generate the config."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs


export http_proxy=http://proxy.cscs.ch:8080
export https_proxy=http://proxy.cscs.ch:8080
export OMP_NUM_THREADS=4
export LOCAL_WORLD_SIZE=4 # Number of GPUs per node
export CUDA_DEVICE_MAX_CONNECTIONS=1 # required for TP > 1

# ******** Master port, address and world size MUST be passed as variables for DDP to work
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_PORT"=$MASTER_PORT

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
# ******************************************************************************************

# Git metadata
if git -C "$WORKDIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_BRANCH=$(git -C "$WORKDIR" branch --show-current)
  GIT_COMMIT=$(git -C "$WORKDIR" rev-parse HEAD)
  GIT_LOG=$(git -C "$WORKDIR" log -1 --pretty=format:'%h | %an | %ad | %s' --date=iso)
  export GIT_COMMIT_HASH=$GIT_COMMIT # Export for saving ckpt in training scripts
  echo "Git branch: ${GIT_BRANCH}"
  echo "Git commit: ${GIT_COMMIT}"
  echo "Last commit: ${GIT_LOG}"
else
  echo "Git metadata unavailable."
fi

echo "Starting finetuning with config: $CONFIG_FILE"
echo "Using 4 GPUs"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

CMD="source $WORKDIR/.venv/bin/activate && \
 torchrun --nproc_per_node=4 \
 --node_rank $SLURM_NODEID \
 --nnodes=$SLURM_NNODES \
 --start-method forkserver \
 --rdzv_backend=c10d \
 --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
 $WORKDIR/run_train.py --config-file $CONFIG_FILE"

# Run training
# numactl --membind=0-3 # not available on container
srun --label -A a122 --environment=cuda129_ub2404 \
    bash -c "$CMD"
 
