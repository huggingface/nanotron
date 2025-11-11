#!/bin/bash

CKPT_PATH_HF=/iopsstor/scratch/cscs/lhuang/FoundationModel/outputs/megatron_checkpoints/exp_fsq_245_split_vocab32768/llama_3B_vocab_32768/iter_0204000_hf
CKPT_PATH_NT=/iopsstor/scratch/cscs/lhuang/FoundationModel/outputs/megatron_checkpoints/exp_fsq_245_split_vocab32768/llama_3B_vocab_32768/iter_0204000_nanotron

source .venv/bin/activate

python examples/llama/convert_hf_to_nanotron.py \
    --checkpoint_path $CKPT_PATH_HF \
    --save_path $CKPT_PATH_NT