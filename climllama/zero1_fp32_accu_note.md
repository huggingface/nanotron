# ZeRO-1 with FP32 Accumulation: Why reduce-scatter is Not Implemented

## Background

In `src/nanotron/optim/gradient_accumulator.py:355`, the `reduce_scatter` path for ZeRO-1 with FP32 accumulation is marked as `NotImplementedError`. However, ZeRO-1 without FP32 accumulation works fine. This note explains why.

## ZeRO-1 WITHOUT FP32 Accumulation

When using ZeRO-1 without FP32 accumulation, the gradient sync follows an **all-reduce + all-gather** pattern:

1. **Gradient sync**: Uses `all_reduce` to sync gradients across all DP ranks (see `src/nanotron/parallel/data_parallel/utils.py:52` - it does `all_reduce` on `param.grad`)
2. **Optimizer step**: Each DP rank only updates its assigned slice of parameters (sharded optimizer states)
3. **Parameter sync**: After the optimizer step, `_all_gather_params()` (`src/nanotron/optim/zero.py:218-252`) broadcasts the updated parameter slices back to all ranks

This works because gradients on `param.grad` are computed in **half precision (bf16/fp16)**, and each rank has access to the full gradient after all-reduce. The memory savings come from only storing optimizer states for a shard of parameters.

## ZeRO-1 WITH FP32 Accumulation (The Problematic Case)

The FP32 accumulator maintains separate FP32 gradient buffers (`fp32_grad_buffers`) for numerical stability. The issue at line 355 is in the **DDP communication hook** (`get_fp32_accum_hook`).

### The Core Problem: Bucket vs Shard Misalignment

DDP's bucketing mechanism (`GradBucket`) doesn't align with ZeRO's parameter sharding:

- **DDP creates buckets** based on reverse computational graph order
- **ZeRO shards parameters** alphabetically/by name
- The `reduce_scatter` in the hook operates on **DDP buckets**, not on the ZeRO sharding scheme

### Issues in the Incomplete Code (lines 357-378)

```python
grad_buffer_tensor_list = [
    accumulator.get_grad_buffer(param_id_to_name[id(param)]).view(-1) for param in bucket.parameters()
]
# ...
input_tensor_lists = [
    torch.split(grad_buffer, split_size_or_sections=len(grad_buffer) // dp_pg.size())  # BUG: uses dp_pg not dp_cp_pg
    for grad_buffer in grad_buffer_tensor_list
]
```

Problems:
1. `bucket.parameters()` may contain a mix of parameters that belong to different ZeRO shards
2. The split assumes equal division across DP ranks, but ZeRO sharding uses `param_name_to_offsets` which is **per-parameter**, not per-bucket
3. There's a bug in the dead code: it references `dp_pg` which is undefined (should be `dp_cp_pg`)

## Why the Non-Hook Path Works (lines 129-156)

The `sync_gradients_across_dp` method at line 129 works because it operates on the **full FP32 gradient buffer** after backward is complete:

- It iterates through `param_name_to_offsets` which aligns with ZeRO sharding
- It can properly use `reduce_scatter_coalesced` because it controls the iteration order

## Summary

The bucket-based DDP hook doesn't have a clean way to map DDP's bucket structure to ZeRO's parameter sharding, making reduce-scatter implementation complex. The simpler `all_reduce_coalesced` approach works because it doesn't require this alignment - every rank gets the full reduced gradient and then uses only the slice it needs.

## Potential Solutions

To implement reduce-scatter for ZeRO-1 with FP32 accumulation, one would need to either:

1. **Disable DDP bucketing** and process parameters one-by-one (loses communication efficiency)
2. **Post-process after backward**: Don't use the DDP hook for reduce-scatter; instead do reduce-scatter after backward completes (similar to how `sync_gradients_across_dp` works)
3. **Align bucket and shard boundaries**: Ensure DDP buckets match ZeRO sharding (complex and may hurt performance)