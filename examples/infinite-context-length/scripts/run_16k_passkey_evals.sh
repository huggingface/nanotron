#!/bin/bash
#SBATCH --job-name=eval_8b_llama_infini_at_20k_ckp_16k_passkey_no_finetuning
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --mem-per-cpu=11G # This is essentially 1.1T / 96
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:4
#SBATCH --partition=hopper-prod
#SBATCH -o /fsx/phuc/new_workspace/experiments/infini_attention_8b_llama/exp55_8b_llama_16384_ctx_length_and_8192_segment_length_and_1.3m_bs_and_global_lr_1.0e-5_and_balance_factor_lr_0.001/logs/evals/eval_8b_llama_infini_at_20k_ckp_16k_passkey_no_finetuning/job_logs/%x-%j.out
#SBATCH --qos=high

# CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 --master_port=25678 run_passkey_eval.py --ckpt-path /fsx/phuc/new_workspace/experiments/infini_attention_8b_llama/exp35_8b_llama_16384_ctx_length_and_8192_segment_length_and_2m_bs_and_needle_finetuning/checkpoints/1326 --save_path /fsx/phuc/projects/nanotron/examples/infinite-context-length/data/exp34/finetuning_passkey_eval_results
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 --master_port=25678 run_passkey_eval.py --ckpt-path /fsx/phuc/new_workspace/experiments/infini_attention_8b_llama/exp55_8b_llama_16384_ctx_length_and_8192_segment_length_and_1.3m_bs_and_global_lr_1.0e-5_and_balance_factor_lr_0.001/ckp_for_evals/20000 --save_path /fsx/phuc/new_workspace/experiments/infini_attention_8b_llama/exp55_8b_llama_16384_ctx_length_and_8192_segment_length_and_1.3m_bs_and_global_lr_1.0e-5_and_balance_factor_lr_0.001/logs/evals/eval_8b_llama_infini_at_20k_ckp_16k_passkey_no_finetuning
