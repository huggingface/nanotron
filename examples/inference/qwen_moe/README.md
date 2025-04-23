# Qwen-MoE Inference

This guide explains how to convert Hugging face Qwen-MoE models to Nanotron format and run inference with them.

## Convert Qwen-MoE to Nanotron Format

Navigate to the `inference/qwen_moe` directory and run:

```bash
torchrun --nproc-per-node 1 examples/inference/qwen_moe/convert.py \
    --nanotron-checkpoint-path nanotron_checkpoints/Qwen1.5-MoE-A2.7B \
    --pretrained-model-name-or-path Qwen/Qwen1.5-MoE-A2.7B
```

This command will save the converted model weights to the specified path in `nanotron_checkpoints`

## Run Inference

From the root directory of Nanotron, run:

```bash
torchrun --rdzv_endpoint=localhost:29700 --rdzv-backend=c10d --nproc_per_node=1 \
    run_generate.py \
    --ckpt-path nanotron_checkpoints/Qwen1.5-MoE-A2.7B
```

This command will load the converted model weights and run inference.
