#!/bin/bash

#SBATCH --job-name=smolm2-bench    # Job name
#SBATCH --time=00:15:00
#SBATCH --partition=hopper-prod
#SBATCH --qos=high

#SBATCH -o /fsx/nouamane/projects/nanotron/logs/%j-%x.out

#SBATCH --nodes=2                 # Number of nodes (modify as needed)
#SBATCH --ntasks-per-node=1       # Number of tasks per node
#SBATCH --cpus-per-task=60         # CPU cores per task
#SBATCH --gres=gpu:8              # Number of GPUs per node
#SBATCH --exclusive               # Exclusive use of nodes

set -x -e

# Load any necessary modules for your system
source /etc/profile.d/modules.sh # for some reason module isn't loaded
module load cuda/12.1

# Activate your conda environment if needed
source /fsx/nouamane/miniconda/bin/activate
conda activate 2-1-cu121
export PATH=/fsx/nouamane/miniconda/envs/2-1-cu121/bin:$PATH

# Get the node names from SLURM
export NODELIST=`scontrol show hostnames $SLURM_JOB_NODELIST`
export MASTER_NODE=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n1`
export MASTER_PORT=12356

# Calculate total number of processes
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE=8
export WORLD_SIZE=$(($NNODES * $GPUS_PER_NODE))

# Set some environment variables for better distributed training
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_DEBUG=INFO

# Nanotron specific
export NANOTRON_BENCHMARK=1

# Print some debugging information
echo "Master node: $MASTER_NODE"
echo "All nodes: $NODELIST"
echo "World size: $WORLD_SIZE"

# Launch the training script using srun
srun torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_NODE:$MASTER_PORT \
    run_train.py \
    --config-file examples/config_tiny_llama.yaml
