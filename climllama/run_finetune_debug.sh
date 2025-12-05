#!/bin/bash

CONFIG_FILE=/capstor/scratch/cscs/lhuang/nanotron_climllama/climllama/config_finetune_with_pe_climllama_test.yaml
WORKDIR=/capstor/scratch/cscs/lhuang/nanotron_climllama
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=18899
export LOCAL_RANK=0
ipython -i --pdb $WORKDIR/run_train.py -- --config-file $CONFIG_FILE
