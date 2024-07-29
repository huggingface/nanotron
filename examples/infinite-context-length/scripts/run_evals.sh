# Calculate depth_percent based on SLURM_ARRAY_TASK_ID
depth_percent=$(( SLURM_ARRAY_TASK_ID * 5 ))  # 0, 5, 10, ..., 100

# Print out the calculated depth percent for debugging/logging
echo "Running job with depth percent: $depth_percent"

# Run the torchrun command with the calculated depth_percent
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=1 examples/infinite-context-length/run_evals.py --ckpt-path /fsx/phuc/new_workspace/experiments/infini_attention_8b_llama/exp18_1b_llama2_100k_ctx_length_and_2m_bs/checkpoints/finetune_needle_checkpoints_with_needle_in_prediction/480 --context_length 32768 --depth_percent $depth_percent
