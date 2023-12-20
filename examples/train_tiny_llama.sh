#!/bin/bash

# Simple script to create a tiny llama model and train it

set -e -x

EXAMPLE_PATH=$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)
REPO_PATH=$(dirname $EXAMPLE_PATH)

python $EXAMPLE_PATH/config_tiny_llama.py

python -u -m torch.distributed.run \
    --nproc_per_node 1 \
    --nnodes 1 \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    $REPO_PATH/run_train.py --config-file $EXAMPLE_PATH/config_tiny_llama.yaml
