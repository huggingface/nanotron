#!/bin/sh

if [ "$1" = "debug" ]; then
    debugpy-run -p 1234 -m torch.distributed.launch \
                -- \
                --nproc_per_node=4 \
                --rdzv_endpoint=localhost:6000 \
                --use_env \
                --tee=3 \
                use_trainer.py \
                --config-file=config_tiny.yaml
elif [ "$1" = "bench" ]; then
    python -m torch.distributed.launch \
                --nproc_per_node=1\
                --rdzv_endpoint=localhost:6000 \
                --use_env \
                --tee=3 \
                benchmark_generation.py
else
    python -m torch.distributed.launch \
                --nproc_per_node=4 \
                --rdzv_endpoint=localhost:6000 \
                --use_env \
                --tee=3 \
                use_trainer.py \
                --config-file=config_tiny.yaml
fi
