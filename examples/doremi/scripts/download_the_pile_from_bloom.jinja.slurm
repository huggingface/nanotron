#!/bin/bash
#SBATCH --job-name=download_the_pile
#SBATCH --partition=hopper-cpu
#SBATCH --requeue
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=500
#SBATCH --qos=high
#SBATCH --array=2-29
#SBATCH -o /fsx/phuc/project_data/doremi/logs/doremi-%j-%a-%x.out


FILE_NUMBER=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

# Check if FILE_NUMBER is set
if [ -z "$FILE_NUMBER" ]; then
    echo "Error: FILE_NUMBER is not set."
    exit 1
fi


gcloud storage cp gs://bigscience/pile/raw/train/${FILE_NUMBER}.jsonl /fsx/phuc/project_data/doremi/datasets/the_pile_raw/
