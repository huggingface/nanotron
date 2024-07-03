#!/bin/bash
#SBATCH --job-name=generate_16k_eval_data
#SBATCH --partition=hopper-cpu
#SBATCH --requeue
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=500
#SBATCH --qos=high
#SBATCH --array=0-356
#SBATCH -o /fsx/phuc/projects/nanotron/examples/infinite-context-length/data/exp34/eval_data_gen_logs/%j-%a-%x.out


# Set the context length
context_length=16384

# Define the depth_percent values
depth_percents=(0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)

# Calculate the depth_percent index and id based on the job ID
depth_percent_index=$((SLURM_ARRAY_TASK_ID / 17))
id=$((SLURM_ARRAY_TASK_ID % 17 + 1))

# Get the depth_percent value for the current job
depth_percent=${depth_percents[$depth_percent_index]}

echo "Running job with depth_percent=$depth_percent and id=$id, SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

# Run the command with the current depth_percent and id
python3 -u /fsx/phuc/projects/nanotron/examples/infinite-context-length/generate_data.py --context_length $context_length --depth_percent $depth_percent --id $id --num_prompts 2 --tokenizer_path /fsx/haojun/lighteval_evaluation_model/NanotronLlama3-8B --is_exact_context_length 0 --is_padding 0
