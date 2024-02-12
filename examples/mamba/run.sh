#!/bin/sh

if [ "$1" = "debug" ]; then
    python configs/make_config_mamba_fast.py && \
    FI_PROVIDER="efa" CUDA_DEVICE_MAX_CONNECTIONS=1 debugpy-run -p 1234 -m torch.distributed.launch \
                -- \
                --nproc_per_node=1 \
                --master_port=29600 \
                --rdzv_endpoint=localhost:6000 \
                --use_env \
                --tee=3 \
                ../../run_train.py \
                --config-file=configs/config_mamba.yaml
elif [ "$1" = "eval" ]; then
    FI_PROVIDER="efa" CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
                --nproc_per_node=1 \
                --master_port 29600 \
                ../generate.py \
                --pp 1 \
                --tp 1 \
                --ckpt-path /fsx/ferdinandmom/github/mamba/checkpoints/mamba-1p62M-stas-openwebtext-10k/7
else
    python configs/make_config_mamba_fast.py && \
    FI_PROVIDER="efa" CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
                --nproc_per_node=1 \
                --master_port=29600 \
                ../../run_train.py \
                --config-file=configs/config_mamba.yaml
fi