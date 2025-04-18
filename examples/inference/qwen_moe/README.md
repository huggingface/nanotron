# Qwen-MoE Inference

## Convert Qwen-MoE to Nanotron
Under inference/qwen_moe, run
```bash
torchrun --nproc-per-node 1 convert.py --nanotron-checkpoint-path nanotron_checkpoints/Qwen1.5-MoE-A2.7B --pretrained-model-name-or-path Qwen/Qwen1.5-MoE-A2.7B
```

## Inference
Under the root of nanotron, run
```bash
torchrun --rdzv_endpoint=localhost:29700 --rdzv-backend=c10d --nproc_per_node=1 run_generate.py --ckpt-path examples/inference/qwen_moe/nanotron_checkpoints/Qwen1.5-MoE-A2.7B
```
