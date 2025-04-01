#!/bin/bash
# =============================================================================
# Nanotron Multi-Node Training Script
# =============================================================================
# This script runs distributed training for Nanotron models using SLURM.
# It supports both sbatch and salloc/srun modes of execution.
#
# Usage:
#   1. Submit as a SLURM job:
#      sbatch --nodes=N run_multinode.sh
#
#   2. Run within an interactive allocation:
#      SALLOC_JOBID=<jobid> NNODES=<N> bash run_multinode.sh
#
# All paths and settings can be customized by setting environment variables
# before running the script. See the configuration section below.
# =============================================================================

#SBATCH --job-name=smolm2-bench   # Job name
#SBATCH --time=00:02:00
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --nodes=2                 # Number of nodes (modify as needed)
#SBATCH --ntasks-per-node=1       # Number of tasks per node
#SBATCH --cpus-per-task=60        # CPU cores per task
#SBATCH --gres=gpu:8              # Number of GPUs per node
#SBATCH --exclusive               # Exclusive use of nodes
#SBATCH --wait-all-nodes=1        # fail if any node is not ready
#SBATCH -o /fsx/nouamane/projects/nanotron/logs/%j-%x.out  # Default log path (can't use variables in SBATCH directives)

# =============================================================================
# Configuration variables (can be overridden by environment variables)
# =============================================================================
: ${NANOTRON_BENCHMARK:=1} # Nanotron BENCHMARK mode
: ${PROJECT_NAME:="nanotron"}
: ${LOG_DIR:="/fsx/nouamane/projects/nanotron/logs"}  # LOG_DIR in case of srun
: ${CONDA_PATH:="/fsx/nouamane/miniconda"}
: ${CONDA_ENV:="2-1-cu121"}
: ${PROJECT_DIR:="/fsx/nouamane/projects/nanotron"}
: ${CONFIG_FILE:="examples/config_tiny_llama.yaml"}
: ${MASTER_PORT:=12356}
: ${GPUS_PER_NODE:=8}             # Number of GPUs per node
: ${CUDA_MODULE:="cuda/12.1"}     # CUDA module to load

set -x -e

echo "Running script: $0"
echo "Using configuration:"
echo "  PROJECT_NAME: ${PROJECT_NAME}"
echo "  LOG_DIR: ${LOG_DIR}"
echo "  CONDA_PATH: ${CONDA_PATH}"
echo "  CONDA_ENV: ${CONDA_ENV}"
echo "  PROJECT_DIR: ${PROJECT_DIR}"
echo "  CONFIG_FILE: ${CONFIG_FILE}"
echo "  GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "  CUDA_MODULE: ${CUDA_MODULE}"
echo "  NANOTRON_BENCHMARK: ${NANOTRON_BENCHMARK}"

# =============================================================================
# SLURM environment setup
# =============================================================================
# If not running under SLURM, set default SLURM environment variables
if [ -z "${SLURM_JOB_ID}" ]; then
    if [ -z "${SALLOC_JOBID}" ]; then
        echo "Error: SALLOC_JOBID environment variable is required but not set. Please run this script within an salloc session."
        exit 1
    fi
    if [ -z "${NNODES}" ]; then
        echo "Error: NNODES environment variable is required but not set. Please run this script within an salloc session."
        exit 1
    fi
    export SALLOC_MODE=1
    export SLURM_JOB_ID=$SALLOC_JOBID
    export SLURM_NNODES=$NNODES
    export SLURM_JOB_NODELIST=$(squeue -j $SALLOC_JOBID -h -o "%N")
fi

# =============================================================================
# Environment setup
# =============================================================================
# Load any necessary modules for your system
source /etc/profile.d/modules.sh # for some reason module isn't loaded
module load ${CUDA_MODULE}

# Activate your conda environment if needed
source ${CONDA_PATH}/bin/activate
conda activate ${CONDA_ENV}
export PATH=${CONDA_PATH}/envs/${CONDA_ENV}/bin:$PATH

# =============================================================================
# Node configuration
# =============================================================================
# Get the node names from SLURM
if [ -z "${SALLOC_MODE}" ]; then # sbatch mode
    export NODELIST=`scontrol show hostnames $SLURM_JOB_NODELIST`
else # srun mode
    export NODELIST=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n$SLURM_NNODES`
fi
export MASTER_NODE=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n1`

# Calculate total number of processes
export NNODES=$SLURM_NNODES
export WORLD_SIZE=$(($NNODES * $GPUS_PER_NODE))

# =============================================================================
# Distributed training configuration
# =============================================================================
# Nanotron specific
export NANOTRON_BENCHMARK
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE=disabled

# debug
export NCCL_DEBUG=WARN # INFO, WARN
# export NCCL_DEBUG_SUBSYS=ALL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_P2P_LEVEL=NVL
# export CUDA_LAUNCH_BLOCKING=1

# =============================================================================
# Job execution
# =============================================================================
# Print GPU topology information
if [ -z "${SALLOC_MODE}" ]; then
    echo "=== GPU Topology ==="
    nvidia-smi topo -m
    echo "=================="
    export SRUN_ALLOC_ARGS=""
else
    export JOBNAME="${PROJECT_NAME}-job"
    export OUTPUT_FILE="${LOG_DIR}/$SLURM_JOB_ID-$(date +%Y-%m-%d-%H-%M-%S)-$JOBNAME.out"
    export SRUN_ALLOC_ARGS="--jobid=$SLURM_JOB_ID --nodes=$NNODES --gres=gpu:$GPUS_PER_NODE --time=01:02:00 --job-name=$JOBNAME"
fi

# Print some debugging information
echo "Master node: $MASTER_NODE"
echo "All nodes: $NODELIST"
echo "World size: $WORLD_SIZE"

# Launch the training script using srun in background
if [ -n "${SALLOC_MODE}" ]; then # srun mode
    srun $SRUN_ALLOC_ARGS --wait=0 --kill-on-bad-exit=1 torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_NODE:$MASTER_PORT \
        --max_restarts 0 \
        --rdzv_conf timeout=60 \
        ${PROJECT_DIR}/run_train.py \
        --config-file ${CONFIG_FILE} > $OUTPUT_FILE 2>&1 &
    # Store the process ID
    SRUN_PID=$!
    echo "Job started in background with PID: $SRUN_PID" | tee -a $OUTPUT_FILE

    # Optionally, you can add:
    echo "To check job status: ps -p $SRUN_PID" | tee -a $OUTPUT_FILE
    echo "To kill the job: kill $SRUN_PID" | tee -a $OUTPUT_FILE

else # sbatch mode
    srun $SRUN_ALLOC_ARGS --wait=0 --kill-on-bad-exit=1 torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_NODE:$MASTER_PORT \
        --max_restarts 0 \
        --rdzv_conf timeout=60 \
        ${PROJECT_DIR}/run_train.py \
        --config-file ${CONFIG_FILE}
fi
