#!/bin/bash
# bash script that copies config file to tmp folder and runs another slurm script

set -x -e

# bash examples/slurm/launcher.slurm examples/slurm/train.slurm examples/config_llama.yaml 1
SLURM_SCRIPT=$1
CONFIG_FILE=$2
TMP_CONFIG_FOLDER=/fsx/nouamane/logs/configs
NNODES=$3

if [ -z "$CONFIG_FILE" ]; then
    echo "Please provide a config file as second argument"
    exit 1
fi

# copy config file to tmp folder
DATETIME=$(date '+%Y%m%d_%H%M%S')
CONFIG_FILE_NAME=${CONFIG_FILE##*/}
CONFIG_FILE_NAME=${CONFIG_FILE_NAME%.yaml}
TMP_CONFIG_FILE=$TMP_CONFIG_FOLDER/${CONFIG_FILE_NAME}_${DATETIME}.yaml
echo "Copying config file to $TMP_CONFIG_FILE"
cp $CONFIG_FILE $TMP_CONFIG_FILE
export CONFIG_FILE=$TMP_CONFIG_FILE

# run another slurm script in arg 1
echo "Running $SLURM_SCRIPT on $NNODES nodes"

# sbatch $SLURM_SCRIPT nnodes
sbatch --nodes $NNODES $SLURM_SCRIPT
