#!/bin/bash
#SBATCH --job-name=eval_8b_llama_infini_0_shot_and_50_samples
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=11G
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:4
#SBATCH --partition=hopper-prod
#SBATCH -o /fsx/phuc/new_workspace/experiments/infini_attention_8b_llama/exp57_8b_llama_1024_ctx_length_and_64_segment_length_and_100k_bs_and_global_lr_1.0e-5_and_balance_factor_lr_0.01_and_balance_factor_0_weight_decay/logs/evals/logs/%x-%A_%a.out
#SBATCH --qos=normal

# Calculate num_shots based on array index
num_shots=${SLURM_ARRAY_TASK_ID}

CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
    --nproc_per_node=4 \
    --master_port=25678 \
    run_passkey_eval.py \
    --ckpt-path /fsx/phuc/new_workspace/experiments/exp57_8b_llama_1024_ctx_length_and_64_segment_length_and_100k_bs_and_global_lr_1.0e-5_and_balance_factor_lr_0.01_and_balance_factor_0_weight_decay/checkpoints/20000 \
    --save_path /fsx/phuc/new_workspace/experiments/infini_attention_8b_llama/exp57_8b_llama_1024_ctx_length_and_64_segment_length_and_100k_bs_and_global_lr_1.0e-5_and_balance_factor_lr_0.01_and_balance_factor_0_weight_decay/logs/evals/results \
    --eval_dataset_path nanotron/llama3-1024-passkey-retrieval-eval \
    --num_shots 0 \
    --num_digits 0 \
    --seed 69 \
    --num_samples 50
