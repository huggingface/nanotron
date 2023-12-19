#!/bin/sh

if [ "$1" = "debug" ]; then
    debugpy-run -p 1234 -m torch.distributed.launch \
                -- \
                --nproc_per_node=4 \
                --rdzv_endpoint=localhost:6000 \
                --use_env \
                --tee=3 \
                main.py \
                --config-file=config_tiny.yaml
                # --tp=2 \
                # --dp=2 \
                # --pp=1 \
                # --hf-gpt2-model-name=gpt2 \
                # --zero-stage=0 \
                # --learning-rate=1e-3 \
                # --num-batches=2 \
                # --recompute-mode=selective \
                # --tp-mode=ALL_REDUCE \
                # --pp-engine=1f1b \
                # --dtype=float32 \
                # --micro-batch-size=2 \
                # --batch-accumulation-per-replica=2 \
                # --sequence-length=512 \
                # --hf-dataset-name=stas/openwebtext-10k \
                # --hf-dataset-split=train \
                # --dataset-processing-num-proc-per-process=12 \
                # --loading-num-proc-per-process=2 \
                # --checkpoint-path examples/gpt2/checkpoint \
                # --log-level=info \
                # --log-level-replica=info\
                # --iteration-step-info-interval=1
else
    python -m torch.distributed.launch \
                --nproc_per_node=4 \
                --rdzv_endpoint=localhost:6000 \
                --use_env \
                --tee=3 \
                main.py \
                --config-file=config_tiny.yaml
fi
