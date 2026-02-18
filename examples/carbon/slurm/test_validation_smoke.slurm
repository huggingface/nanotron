#!/bin/bash
#SBATCH --job-name=test-smoke
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/loubna/projects_v2/carbon/loubna-workspace/nanotron_repos/main/nanotron/examples/carbon/logs/test-smoke-%j.out
#SBATCH --qos=normal
#SBATCH --begin=now+0minutes
#SBATCH --time=5-00:00:00


set -x
TRAINER_PYTHON_FILE=/fsx/loubna/projects_v2/carbon/loubna-workspace/nanotron_repos/main/nanotron/run_train.py
nvidia-smi
echo "/!\ please verify that you use the right env"
which python3
echo "START TIME: $(date)"
secs_to_human() {
    echo "$(( $1 / 3600 )):$(( ($1 / 60) % 60 )):$(( $1 % 60 ))"
}
export PRINT_FILES=1
# export NCCL_DEBUG=INFO
start=$(date +%s)
echo "$(date -d @${start} "+%Y-%m-%d %H:%M:%S"): ${SLURM_JOB_NAME} start id=${SLURM_JOB_ID}\n"

# SLURM stuff
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
# Export Weights & Biases entity for HuggingFace
export WANDB_ENTITY="huggingface"

export FORCE_UPDATE_CACHE_S3=1

export TMPDIR=/scratch
export CUDA_DEVICE_MAX_CONNECTIONS="1"

module load cuda/12.4
source /fsx/loubna/.venv-2-6-cu124/bin/activate
echo "PATH: $PATH"
# Log environment variables
echo "Environment variables:"
printenv | sort

# Log python path
echo "Python path: $(which python)"

# Log torchrun path
echo "Torchrun path: $(which torchrun)"

# Log installed Python packages
echo "Installed Python packages:"
python -m pip freeze

nvcc --version

echo go $COUNT_NODE
echo $HOSTNAMES

##### MOVE TO YAML ######

CMD=" \
    $TRAINER_PYTHON_FILE \
    --config-file /fsx/loubna/projects_v2/carbon/loubna-workspace/nanotron_repos/main/nanotron/examples/carbon/configs/test_validation_smoke_with_evo2.yaml
    "

export LAUNCHER="torchrun \
    --nproc_per_node 8 \
    --nnodes $COUNT_NODE \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_id $SLURM_JOB_ID \
    --node_rank $SLURM_PROCID \
    --role $SLURMD_NODENAME: \
    --max_restarts 0 \
    --tee 3 \
    "

# Wait a random number between 0 and 1000 (milliseconds) to avoid too many concurrent requests to the hub
random_milliseconds=$(( RANDOM % 1001 ))
sleep_time=$(bc <<< "scale=3; $random_milliseconds / 1000")
echo "Sleeping for $sleep_time seconds..."
sleep $sleep_time

launch_args="srun $SRUN_ARGS -u bash -c $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD"

srun $SRUN_ARGS -u bash -c "$LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD"


echo "END TIME: $(date)"
