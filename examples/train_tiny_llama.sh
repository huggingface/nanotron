#!/bin/bash

# Simple script to create a tiny llama model and train it

set -e -x

EXAMPLE_PATH=$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)
REPO_PATH=$(dirname $EXAMPLE_PATH)

python $EXAMPLE_PATH/config_tiny_llama.py
python $REPO_PATH/run_train.py --config-file $EXAMPLE_PATH/config_tiny_llama.yaml
