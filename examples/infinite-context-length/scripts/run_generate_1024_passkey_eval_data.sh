#!/bin/bash
#SBATCH --job-name=generate_1k_eval_data_with_0_1_2_3_shots
#SBATCH --partition=hopper-cpu
#SBATCH --requeue
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=500
#SBATCH --qos=normal
#SBATCH --array=0-335
#SBATCH -o /fsx/phuc/projects/nanotron/examples/infinite-context-length/data/eval/logs/1024/%j-%a-%x.out

# Set the context length
context_length=1024

# Define the depth_percent values, 21 values
depth_percents=(0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)

# Define num_shots and num_digits values
num_shots_values=(0 1 2 3)
num_digits_values=(1 2 3 4)

# Calculate indices based on the job ID
depth_percent_index=$((SLURM_ARRAY_TASK_ID / 16))
num_shots_index=$(((SLURM_ARRAY_TASK_ID % 16) / 4))
num_digits_index=$((SLURM_ARRAY_TASK_ID % 4))

# Get the values for the current job
depth_percent=${depth_percents[$depth_percent_index]}
num_shots=${num_shots_values[$num_shots_index]}
num_digits=${num_digits_values[$num_digits_index]}

echo "Running job with depth_percent=$depth_percent, num_shots=$num_shots, num_digits=$num_digits, SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

# Run the command with the current parameters
# python3 -u /fsx/phuc/projects/nanotron/examples/infinite-context-length/generate_data.py \
#     --context_length $context_length \
#     --depth_percent $depth_percent \
#     --id $SLURM_ARRAY_TASK_ID \
#     --num_prompts 5 \
#     --tokenizer_path /fsx/haojun/lighteval_evaluation_model/NanotronLlama3-8B \
#     --is_exact_context_length 0 \
#     --is_padding 0 \
#     --num_shots $num_shots \
#     --num_digits $num_digits

# Run the command with the current parameters
python3 -u /fsx/phuc/projects/nanotron/examples/infinite-context-length/generate_data.py \
    --save_path /fsx/phuc/projects/nanotron/examples/infinite-context-length/data/eval/1024 \
    --context_length $context_length \
    --depth_percent $depth_percent \
    --id $SLURM_ARRAY_TASK_ID \
    --num_prompts 50 \
    --tokenizer_path /fsx/haojun/lighteval_evaluation_model/NanotronLlama3-8B \
    --is_exact_context_length "no" \
    --is_padding "no" \
    --is_eval "no" \
    --num_shots $num_shots \
    --num_digits $num_digits
