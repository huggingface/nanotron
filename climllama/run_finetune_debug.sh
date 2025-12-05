#!/bin/bash

CONFIG_FILE=/capstor/scratch/cscs/lhuang/nanotron_climllama/climllama/config_finetune_with_pe_climllama_test.yaml
WORKDIR=/capstor/scratch/cscs/lhuang/nanotron_climllama

ipython -i -m torch.distributed.run -- \
 --nproc_per_node=1 \
 --node_rank 0 \
 --nnodes=1 \
 --start-method forkserver \
 $WORKDIR/run_train.py --config-file $CONFIG_FILE