# ClimLLaMA Scaling Guidelines

This note explains how to scale ClimLLaMA training when you change GPU counts, sequence lengths, or model sizes. It is aimed at the `convert_hf_to_nt.sh -> prepare_finetune_config.sh -> run_finetune.sh` workflow.

Also check:
- https://blog.eleuther.ai/mutransfer/
- https://howtoscalenn.github.io
- https://github.com/google-research/tuning_playbook
- https://jax-ml.github.io/scaling-book/
- https://huggingface.co/spaces/nanotron/ultrascale-playbook

## Quick formulas
- World size: `world_size = dp * tp * pp`
- Global batch (samples): `global_batch = micro_batch_size * batch_accumulation_per_replica * dp`
- Global batch (tokens): `global_batch_tokens = sequence_length * global_batch`
- Total tokens processed: `total_tokens = global_batch_tokens * train_steps`
- Consistency rules: `tp` must divide hidden_size and num_attention_heads; `pp` should divide num_hidden_layers.

## Parallelism choices
- **Data parallel (DP)**: First lever for more throughput if the model fits in a single GPU. Scale DP to match the total GPU count after setting TP/PP.
- **Tensor parallel (TP)**: Use when a single GPU cannot hold the model. Pick the largest TP that evenly divides hidden_size and heads (e.g., TP=2 for a 4096-dim, 32-head model).
- **Pipeline parallel (PP)**: Use when the model depth is too large for TP alone. Keep stages balanced; prefer PP that evenly splits layers (e.g., PP=2 for 32 layers → 16/16).

## Memory and throughput levers
- **Micro-batch size**: Main activation memory knob. Decrease if you OOM; then raise `batch_accumulation_per_replica` to keep the effective batch.
- **Sequence length**: Attention cost scales ~`O(seq^2)`. Doubling context length usually requires halving the micro-batch (or adding accumulation) to keep memory constant.
- **Learning rate & warmup**: Keep the same LR when scaling DP if you keep the global batch fixed. If you change global batch, adjust LR proportionally or re-tune.
- **Checkpoint I/O**: Higher PP/TP increases checkpoint shards. Keep `checkpoint_interval` coarse enough to avoid I/O stalls.

## Example scaling paths
- **Baseline (8 GPUs, fits in memory)**  
  `dp=4, tp=2, pp=1, micro_batch_size=4, batch_accumulation_per_replica=1, sequence_length=4096`  
  → `global_batch_tokens = 4096 * 4 * 1 * 4 = 65,536 tokens/step`
- **Scale out to 16 GPUs (same global batch tokens)**  
  `dp=8, tp=2, pp=1, micro_batch_size=2, batch_accumulation_per_replica=1`  
  → `4096 * 2 * 1 * 8 = 65,536 tokens/step` (keeps LR stable while doubling throughput).
- **Longer context (stay on 8 GPUs, move to 8k seq)**  
  `dp=4, tp=2, pp=1, micro_batch_size=2, batch_accumulation_per_replica=1, sequence_length=8192`  
  → `8192 * 2 * 1 * 4 ≈ 65,536 tokens/step` (keeps tokens/step constant while fitting memory).
- **Deeper model (32 layers → PP=2)**  
  `dp=4, tp=2, pp=2, micro_batch_size=4, batch_accumulation_per_replica=1`  
  Keep layers split evenly (e.g., 16/16) and ensure `run_finetune.sh` uses `NPROC = dp * tp * pp`.

## Updating configs
- Regenerate configs with the right topology:  
  ```bash
  python climllama/prepare_training_config.py \
    --checkpoint_path /path/to/checkpoint \
    --data_prefix /path/to/indexed/data \
    --output_config climllama/config_finetune.yaml \
    --dp 8 --tp 2 --pp 1 \
    --micro_batch_size 2 \
    --batch_accumulation 1 \
    --sequence_length 4096 \
    --train_steps 5000
  ```
- Ensure `run_finetune.sh` (or your launch cmd) sets `--nproc_per_node` to `dp * tp * pp` and matches the number of visible GPUs.

## Sanity checks before a long run
- Run a short smoke test (`train_steps=50`) and check for OOM or `nan` losses.
- Verify the logged world size equals `dp * tp * pp`; mismatches usually mean launch/env issues.
- Track tokens/sec; large drops after scaling often indicate I/O or imbalance between PP stages.
