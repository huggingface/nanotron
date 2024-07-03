#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=hopper-cpu
#SBATCH --job-name=nanotron_llama3-16k-passkey-retrieval-finetuning-meta-llama_Meta-Llama-3-8B-tok1
#SBATCH --time=20:00:00
#SBATCH --output=/fsx/phuc/projects/nanotron/examples/infinite-context-length/data/exp34/tokenization_logs/tokenization/slurm_logs/%A_%a.out
#SBATCH --error=/fsx/phuc/projects/nanotron/examples/infinite-context-length/data/exp34/tokenization_logs/tokenization/slurm_logs/%A_%a.out
#SBATCH --array=0-999
#SBATCH --requeue
#SBATCH --qos=normal
echo "Starting data processing job nanotron_llama3-16k-passkey-retrieval-finetuning-meta-llama_Meta-Llama-3-8B-tok1"

set -xe
export PYTHONUNBUFFERED=TRUE
srun -l launch_pickled_pipeline /fsx/phuc/projects/nanotron/examples/infinite-context-length/data/exp34/tokenization_logs/tokenization/executor.pik
