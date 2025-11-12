#!/bin/bash

# srun --environment=cuda129_ub2404 -p debug --mem=260000 -t 1:30:00 -A a122 --pty bash

uv venv --python 3.11
source .venv/bin/activate
uv pip install --upgrade pip

uv pip install torch --index-url https://download.pytorch.org/whl/cu129
CURRENT_PATH=$(dirname "$BASH_SOURCE[0]")
uv pip install -e $CURRENT_PATH/..

uv pip install datasets transformers datatrove[io] numba wandb setuptools psutil ninja triton pybind11

# Fused kernels
MAX_JOBS=8 uv pip install "flash-attn>=2.5.0" --no-build-isolation
MAX_JOBS=16 uv pip install --no-build-isolation grouped-gemm
MAX_JOBS=16 uv pip install --no-build-isolation transformer_engine[pytorch]

CURRENT_DIR=$(pwd)
cd src/nanotron/data/nemo_dataset && make
cd $CURRENT_DIR