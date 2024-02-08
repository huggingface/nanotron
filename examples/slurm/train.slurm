#!/bin/bash
#SBATCH --job-name=aaa
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem-per-cpu=11G # This is essentially 1.1T / 96
#SBATCH --exclusive
#SBATCH --partition=hopper-prod
#SBATCH -o /fsx/nouamane/logs/moe/%x-%j-train.out
#SBATCH --qos=high
#SBATCz --exclude ip-26-0-150-70,ip-26-0-154-155
#SBATCz --array 1-10%1
#SBATCz --dependency=afterany:521974
#SBATCz --begin=now+0minutes

set -x -e

source ~/.bashrc
source activate /fsx/nouamane/miniconda/envs/2-1-cu121

module load cuda/12.1

echo "START TIME: $(date)"

# show git commit
echo "Git commit: $(git rev-parse HEAD)"

echo "printenv:"
printenv

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# get from env or set default
# if CONFIG_FILE is not set
if [ -z ${CONFIG_FILE+x} ]; then
    echo "CONFIG_FILE is unset, using default"
else
    echo "CONFIG_FILE is set to '$CONFIG_FILE'"
    TMP_CONFIG_FOLDER=/fsx/nouamane/logs/configs

    # remove yaml extension
    CONFIG_FILE_NAME=${CONFIG_FILE##*/}
    CONFIG_FILE_NAME=${CONFIG_FILE_NAME%.yaml}

    # copy config file to tmp folder
    TMP_CONFIG_FILE=$TMP_CONFIG_FOLDER/${CONFIG_FILE_NAME}_${SLURM_JOB_ID}.yaml
    echo "Renaming config file to $TMP_CONFIG_FILE"
    # create TMP_CONFIG_FILE folder if it doesn't exist
    mkdir -p $(dirname $TMP_CONFIG_FILE)
    cp $CONFIG_FILE $TMP_CONFIG_FILE
    CONFIG_FILE=$TMP_CONFIG_FILE
fi

# Use the fast modeling code
export USE_FAST=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NANOTRON_BENCHMARK=1

CMD=" \
    run_train.py \
    --config-file $CONFIG_FILE
    "

export LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

echo $CMD

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN  # VERSION, WARN, INFO

srun $SRUN_ARGS -u bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD"

echo "END TIME: $(date)"
