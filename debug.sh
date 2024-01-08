#!/bin/sh

if [ "$1" = "ref" ]; then
    USE_REF_ROTARY=1 USE_BENCH=0 USE_FAST=0 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
    --nproc_per_node=1 \
    run_generate.py \
    --pp 1 --tp 1 --dp 1 \
    --model_name huggyllama/llama-7b \
    --ckpt-path /admin/home/ferdinand_mom/.cache/huggingface/hub/models--HuggingFaceBR4--llama-7b-orig/snapshots/2160b3d0134a99d365851a7e95864b21e873e1c3 \
    --compare-with-no-cache --max-new-tokens 32
elif [ "$1" = "ref-debug" ]; then
    USE_REF_ROTARY=1 USE_BENCH=0 USE_FAST=0 CUDA_DEVICE_MAX_CONNECTIONS=1 debugpy-run -p 1234 -m torch.distributed.run \
    --\
    --nproc_per_node=1 \
    run_generate.py \
    --pp 1 --tp 1 --dp 1 \
    --model_name huggyllama/llama-7b \
    --ckpt-path /admin/home/ferdinand_mom/.cache/huggingface/hub/models--HuggingFaceBR4--llama-7b-orig/snapshots/2160b3d0134a99d365851a7e95864b21e873e1c3 \
    --compare-with-no-cache --max-new-tokens 64
elif [ "$1" = "bug" ]; then
    USE_REF_ROTARY=0 USE_BENCH=1 USE_FAST=0 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
    --compare-with-no-cache --max-new-tokens 32 
    --master_port 25510 \
    run_generate.py \
    --pp 1 --tp 1 --dp 1 \
    --ckpt-path /admin/home/ferdinand_mom/.cache/huggingface/hub/models--HuggingFaceBR4--llama-7b-orig/snapshots/2160b3d0134a99d365851a7e95864b21e873e1c3 \
    --compare-with-no-cache --max-new-tokens 32
elif [ "$1" = "bug-debug" ]; then
    USE_REF_ROTARY=0 USE_BENCH=0 USE_FAST=0 CUDA_DEVICE_MAX_CONNECTIONS=1 debugpy-run -p 5678 -m torch.distributed.run \
    -- \
    --nproc_per_node=1 \
    --master_addr localhost \
    --master_port 25510 \
    run_generate.py \
    --pp 1 --tp 1 --dp 1 \
    --model_name huggyllama/llama-7b \
    --ckpt-path /admin/home/ferdinand_mom/.cache/huggingface/hub/models--HuggingFaceBR4--llama-7b-orig/snapshots/2160b3d0134a99d365851a7e95864b21e873e1c3 \
    --compare-with-no-cache --max-new-tokens 32 
else
    echo "Unknown option"
fi
