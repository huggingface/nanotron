export CUDA_DEVICE_MAX_CONNECTIONS=1

# Run the main training example
torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml

# Generate from the trained model
torchrun --nproc_per_node=8 run_generate.py ---ckpt-path checkpoints/10

# Run DoReMi example
# torchrun --nproc_per_node=8 ../examples/doremi/train_doremi.py ---config-file checkpoints

# Run MoE example

# Run Spectral µTransfer example

# Run Mamba example
