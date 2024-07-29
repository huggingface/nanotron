#!/bin/bash
#SBATCH --job-name=eval_infini_attention_32k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --mem-per-cpu=11G # This is essentially 1.1T / 96
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:1
#SBATCH --partition=hopper-prod
#SBATCH -o /fsx/phuc/new_workspace/experiments/infini_attention_8b_llama/exp19_1b_llama2_32k_ctx_length_and_2m_bs/checkpoints/90000/eval_logs/%x-%j-480-ckp-eval.out
#SBATCH --qos=high
#SBATCH --array=1-10

# Define the list of depth percentages
depth_values=(0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)

# Split the depth values into chunks for each SLURM job array
# This setup creates ten jobs, with two depths each, except the last job, which has three
chunks=(
  "${depth_values[@]:0:2}"   # Job 1: 0, 5
  "${depth_values[@]:2:2}"   # Job 2: 10, 15
  "${depth_values[@]:4:2}"   # Job 3: 20, 25
  "${depth_values[@]:6:2}"   # Job 4: 30, 35
  "${depth_values[@]:8:2}"   # Job 5: 40, 45
  # "${depth_values[@]:10:2}"  # Job 6: 50, 55
  # "${depth_values[@]:12:2}"  # Job 7: 60, 65
  # "${depth_values[@]:14:2}"  # Job 8: 70, 75
  # "${depth_values[@]:16:2}"  # Job 9: 80, 85
  # "${depth_values[@]:18:3}"  # Job 10: 90, 95, 100
)

# Determine the assigned depths based on SLURM_ARRAY_TASK_ID
depths=(${chunks[SLURM_ARRAY_TASK_ID - 1]})

# Run the job for each depth percent in the assigned chunk
for depth_percent in "${depths[@]}"; do
  echo "Running job with depth percent: $depth_percent"
  CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=1 /fsx/phuc/projects/nanotron/examples/infinite-context-length/run_evals.py --ckpt-path /fsx/phuc/new_workspace/experiments/infini_attention_8b_llama/exp19_1b_llama2_32k_ctx_length_and_2m_bs/checkpoints/90000 --context_length 32768 --depth_percent $depth_percent
done
