#!/bin/bash

# srun --environment=pytorch_2510 -p debug --mem=260000 -t 1:30:00 -A a122 --pty bash

uv venv nanotron --python 3.11
source nanotron/bin/activate
uv pip install --upgrade pip

uv pip install torch --index-url https://download.pytorch.org/whl/cu130
CURRENT_PATH=$(dirname "$BASH_SOURCE[0]")
uv pip install -e $CURRENT_PATH/..

uv pip install datasets transformers datatrove[io] numba wandb setuptools psutil
# Fused kernels
MAX_JOBS=16 uv pip install ninja triton "flash-attn>=2.5.0" --no-build-isolation