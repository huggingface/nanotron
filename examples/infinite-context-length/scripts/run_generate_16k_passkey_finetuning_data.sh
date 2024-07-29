#!/bin/bash
#SBATCH --job-name=generate_16k_passkey_finetuning_data
#SBATCH --partition=hopper-cpu
#SBATCH --requeue
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=500
#SBATCH --qos=high
#SBATCH --array=0-1000
#SBATCH -o /fsx/phuc/projects/nanotron/examples/infinite-context-length/data/exp34/finetuning_data_gen_logs/%j-%a-%x.out

# 15 context length * 17 depths * 4 jobs = 1020

# Define the list of context lengths, 15 context lengths
# context_lengths=(512 1024 1536 2048 2560 3072 3584 4096 4608 5120 5632 6144 6656 7168 7680)
context_lengths=(8192 8704 9216 9728 10240 10752 11264 11776 12288 12800 13312 13824 14336 14848 15360 15872)

# Define the depth_percent values, 17 values
depth_percents=(0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)

# # Calculate the context_length index, depth_percent index, and id based on the job ID
# context_length_index=$((SLURM_ARRAY_TASK_ID / (${#depth_percents[@]} * 3)))
# depth_percent_index=$(((SLURM_ARRAY_TASK_ID % (${#depth_percents[@]} * 3)) / 3))
# id=$((SLURM_ARRAY_TASK_ID % 5 + 1))

total_jobs=$((SLURM_ARRAY_TASK_COUNT))
jobs_per_context_length=$((total_jobs / ${#context_lengths[@]}))
jobs_per_depth_percent=$((jobs_per_context_length / ${#depth_percents[@]}))

# Calculate the context_length index, depth_percent index, and id based on the job ID
context_length_index=$((SLURM_ARRAY_TASK_ID / jobs_per_context_length))
depth_percent_index=$(((SLURM_ARRAY_TASK_ID % jobs_per_context_length) / jobs_per_depth_percent))
id=$((SLURM_ARRAY_TASK_ID % jobs_per_depth_percent))

# Get the context_length and depth_percent values for the current job
context_length=${context_lengths[$context_length_index]}
depth_percent=${depth_percents[$depth_percent_index]}

echo "Running job with context_length=$context_length, depth_percent=$depth_percent, and id=$id, SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

# Run the command with the current context_length, depth_percent, and id
python3 -u /fsx/phuc/projects/nanotron/examples/infinite-context-length/generate_data.py --save_path /fsx/phuc/projects/nanotron/examples/infinite-context-length/data/exp34/finetune_data --context_length $context_length --depth_percent $depth_percent --id $id --num_prompts 50 --tokenizer_path /fsx/haojun/lighteval_evaluation_model/NanotronLlama3-8B --is_exact_context_length 0 --is_padding 0 --is_eval 0 --check_key_in_dataset nanotron/llama3-16k-passkey-retrieval-eval
