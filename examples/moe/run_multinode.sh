#!/bin/bash
#SBATCH --job-name=smolm2-bench   # Job name
#SBATCH --time=00:02:00
#SBATCH --partition=hopper-prod
#SBATCH --qos=low

#SBATCH -o /fsx/phuc/new_workspace/experiments/qwen_moe/benchmark/exp0a0_benhmark_num_experts_topk_and_ep_in_a_node/logs/%j-%x.out

#SBATCH --nodes=2                 # Number of nodes (modify as needed)
#SBATCH --ntasks-per-node=1       # Number of tasks per node
#SBATCH --cpus-per-task=60         # CPU cores per task
#SBATCH --gres=gpu:8              # Number of GPUs per node
#SBATCH --exclusive               # Exclusive use of nodes
#SBATCH --wait-all-nodes=1        # fail if any node is not ready

# run using
# sbatch --nodes=1 run_multinode.sh
# or
# SALLOC_JOBID=13482276 NNODES=1 bash run_multinode.sh

set -x -e
echo "Running script: $0"


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

# Load any necessary modules for your system
source /etc/profile.d/modules.sh # for some reason module isn't loaded
module load cuda/12.1
# Unset FI_PROVIDER to avoid potential libfabric provider issues
# unset FI_PROVIDER


# Activate your conda environment if needed
source /admin/home/phuc_nguyen/.bashrc
source /admin/home/phuc_nguyen/miniconda3/etc/profile.d/conda.sh
conda activate /fsx/phuc/temp/env_for_qwen_moe/env/
# conda activate 2-1-cu121
# export PATH=/fsx/nouamane/miniconda/envs/2-1-cu121/bin:$PATH
# export PATH=/fsx/phuc/temp/env_for_qwen_moe/env//bin:$PATH

# Get the node names from SLURM
if [ -z "${SALLOC_MODE}" ]; then # sbatch mode
    export NODELIST=`scontrol show hostnames $SLURM_JOB_NODELIST`

else # srun mode
    export NODELIST=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n$SLURM_NNODES`
fi
export MASTER_NODE=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n1`
export MASTER_PORT=12356

# Calculate total number of processes
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE=8
export WORLD_SIZE=$(($NNODES * $GPUS_PER_NODE))

# Set some environment variables for better distributed training
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN # INFO, WARN
# export NCCL_DEBUG_SUBSYS=ALL
# export CUDA_LAUNCH_BLOCKING=1

# Nanotron specific
export NANOTRON_BENCHMARK=1
export WANDB_MODE=disabled

# export TORCH_NCCL_USE_COMM_NONBLOCKING=1

# Trying to avoid hangs
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# debug
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# export NCCL_P2P_LEVEL=NVL
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_IB_CUDA_SUPPORT=0 # Disable RDMA
# export NCCL_NET_GDR_LEVEL=LOC
# Test Script - save as test_comm.sh

# Test 1 - Force TCP
# echo "Running with TCP only..."
# export NCCL_P2P_LEVEL=LOC

# # Match bandwidth patterns
# export NCCL_MAX_NCHANNELS=2
# export NCCL_MIN_NCHANNELS=2


# export NCCL_NET_GDR_LEVEL=LOC # Disable RDMA
# export NCCL_SHM_DISABLE=0 # disables the Shared Memory (SHM) transport
# export NCCL_IB_DISABLE=0 # disables the InfiniBand (IB) transport
# export NCCL_IB_TIMEOUT=60  # 20 = ~4 seconds , 21 = ~8 seconds , 22 = ~16 seconds
# export NCCL_IB_RETRY_CNT=7  # Increase retry count as well

# Force SHM
# export NCCL_NET_PLUGIN=none # fixes hang but doesnt work multinode
# export NCCL_SOCKET_NTHREADS=1
# export FI_PROVIDER="tcp"

# Print GPU topology information
if [ -z "${SALLOC_MODE}" ]; then
    echo "=== GPU Topology ==="
    nvidia-smi topo -m
    echo "=================="
    export SRUN_ALLOC_ARGS=""
else
    export JOBNAME="smolm2-bench"
    export OUTPUT_FILE="/fsx/phuc/new_workspace/experiments/qwen_moe/benchmark/exp0a0_benhmark_num_experts_topk_and_ep_in_a_node/logs/$SLURM_JOB_ID-$(date +%Y-%m-%d-%H-%M-%S)-$JOBNAME.out"
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
        /fsx/phuc/temp/env_for_qwen_moe/nanotron/run_train.py \
        --config-file examples/config_tiny_llama.yaml > $OUTPUT_FILE 2>&1 &
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
        /fsx/phuc/temp/env_for_qwen_moe/nanotron/run_train.py \
        --config-file examples/config_tiny_llama.yaml
