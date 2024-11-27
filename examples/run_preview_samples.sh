#!/bin/bash

# Print detokenized samples from a dataset / tokenizer specified in "--config-file" 

set -e -x

EXAMPLE_PATH=$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)
REPO_PATH=$(dirname $EXAMPLE_PATH)

export CUDA_DEVICE_MAX_CONNECTIONS=1
export FI_PROVIDER="efa"

python -u -m torch.distributed.run \
    --nproc_per_node 8 \
    --nnodes 1 \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    $REPO_PATH/run_preview_samples.py --config-file $EXAMPLE_PATH/config_tiny_llama.yaml

