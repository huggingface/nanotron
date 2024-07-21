#!/bin/bash
#SBATCH --job-name=long-context
#SBATCH --qos=high
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hopper-prod
#SBATCH --gres=gpu:8
#SBATCH --mem=0

# HF stuff
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# script for a benchmark
set -x -e

# replace with your own env
echo "START TIME: $(date)"
source /admin/home/haojun_zhao/miniconda3/etc/profile.d/conda.sh
conda activate /admin/home/haojun_zhao/miniconda3/envs/nt_mamba
echo python3 version = $(python3 --version)

# SLURM stuff
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9001
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export FI_PROVIDER="efa"
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "NCCL version: $(python -c "import torch;print(torch.cuda.nccl.version())")"
echo "CUDA version: $(python -c "import torch;print(torch.version.cuda)")"

module load cuda/12.1

current_dir=$(pwd)
parent_of_parent_dir=$(dirname $(dirname "$current_dir"))

TRAIN_SCRIPT= "$parent_of_parent_dir/run_train.py"
CONFIG_FILE= "$current_dir/1M_4stages/config_0_theta=22.4M_steps=10_seq_len=65536.yaml"

srun torchrun \
   --nnodes=4 \
   --nproc_per_node=8 \
   --rdzv_backend=c10d \
   --rdzv-id ${SLURM_JOB_ID} \
   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
   --max_restarts=0 \
   --tee=3 \
   $TRAIN_SCRIPT --config-file $CONFIG_FILE
