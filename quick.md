export NANOTRON_BENCHMARK=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=8 run_train.py --config-file examples/config_llama.yaml

# Multinode
bash examples/slurm/launcher.slurm examples/slurm/train.slurm examples/config_llama.yaml 1
