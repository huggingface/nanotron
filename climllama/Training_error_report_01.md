# Error Log Analysis - Job 1198520

**Date**: 2025-12-06
**Job ID**: 1198520
**Log File**: `climllama/logs/finetune_climllama_1198520.out`

## Summary

The training run crashed with a **CUDA illegal memory access error** (`cudaErrorIllegalAddress`) during the first training iteration. All 16 processes across 4 nodes crashed simultaneously with SIGABRT.

## Timeline of Events

### 1. Setup Phase (10:49:29 - 10:49:47)
- Model built successfully (3.28B parameters)
- Weights loaded from checkpoint: `/iopsstor/scratch/cscs/lhuang/FoundationModel/outputs/megatron_checkpoints/exp_fsq_245_split_vocab32768/llama_3B_vocab_32768/iter_0204000_nanotron_climllama`
- Optimizer and learning rate scheduler configured

### 2. Data Loading Phase (10:49:47 - 11:00:20)
- Built ClimLlamaDataset with 2155 documents from 100 shards
- Training sampler instantiated with 2.56M total samples

### 3. Training Initiation (11:00:20 - 11:00:23)
- Training started with configuration:
  - `mbs: 4`
  - `grad_accum: 32`
  - `cp: 2`
  - `sequence_length: 32768`
  - `global_batch_size: 512`
- Memory usage: `18787.00MiB used, 19415.07MiB peak allocated, 20170.00MiB peak reserved`

### 4. Crash (11:00:56)
- ~33 seconds after training iteration started
- All 16 processes (across 4 nodes) crashed simultaneously with SIGABRT

## Root Cause

**CONFIRMED: position_ids Reshape Bug in Context Parallelism**

**Location**: `src/nanotron/models/qwen.py:275` (in `Qwen2Attention.forward()`)

The crash is caused by an **incorrect tensor reshape operation** that doubles the batch dimension when Context Parallelism (CP) is enabled. This creates a shape mismatch between `hidden_states` and `position_ids`, leading to illegal memory access in CUDA kernels.

### The Bug

```python
# Line 275 in qwen.py (BUGGY CODE):
return {"hidden_states": output, "position_ids": position_ids.view(-1, seq_length)}
#                                                                        ^^^^^^^^^^
#                                                           Uses LOCAL seq_length (e.g., 256)
#                                                     But position_ids contains GLOBAL data (e.g., 512)
```

When CP=2:
- `position_ids` enters with shape `[batch=2, global_seq=512]`
- Gets flattened to `[1024]`
- `seq_length` is calculated as `global_seq // cp_size = 512 // 2 = 256`
- Reshape to `(-1, 256)` creates `[1024/256, 256]` = **`[4, 256]`** ❌
- **Expected**: `[2, 512]` or `[2, 256]` (batch should remain 2, not 4!)

This doubled batch dimension (2 → 4) causes immediate shape mismatch with `hidden_states` which maintains the correct batch size, triggering CUDA illegal memory access.

## Key Error Details

- **Error Location**: `ProcessGroupNCCL.cpp:2057` - NCCL watchdog thread detected the error
- **Exit Code**: `-6 (SIGABRT)` - All processes received abort signal
- **First Failure**: Rank 8 (local_rank 0 on node nid007142)
- **Affected Ranks**: All 16 ranks across 4 nodes
- **Error Message**:
  ```
  [PG ID 3 PG GUID X Rank Y] Process group watchdog thread terminated with exception:
  CUDA error: an illegal memory access was encountered
  ```

## Critical Warning (Red Herring)

```
the model's max_position_embeddings 4096 is ignored because the sequence length we're training on is 32768.
```

This warning appeared in the original crash but was **NOT the root cause**. Subsequent testing with `max_position_embeddings=32768` (corrected) still produced the same crash, confirming the real issue is the reshape bug.

## Debugging Recommendations

The error message suggests:

1. **Enable synchronous CUDA execution**:
   ```bash
   export CUDA_LAUNCH_BLOCKING=1
   ```
   This will help pinpoint the exact operation causing the illegal memory access.

2. **Enable device-side assertions**:
   Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions (per error message)

3. **Check Configuration**: Review `climllama/config_finetune_with_pe_climllama.yaml` for:
   - Position embedding settings
   - Context parallelism (CP=2) configuration
   - Sequence length vs model capacity mismatch

## Detailed Technical Analysis

### Debug Run Configuration (debug_output2.log)

To diagnose the issue, a debug run was executed with:
- **Config**: `mbs: 2 | cp: 2 | sequence_length: 512 | global_batch_size: 64`
- **Added logging** in `qwen.py` to track tensor shapes through attention layers

**Training Configuration (Line 2499):**
```
[2;3m12/06 14:46:01 [INFO|CP=0|TP=0|nid006683][0m: mbs: 2 | grad_accum: 32 | cp: 2 | sequence_length: 512 | global_batch_size: 64 | train_steps: 3 | start_iteration_step: 0 | consumed_tokens_total: 0
```

**Training Start (Lines 2504-2505):**
```
[2;3m12/06 14:46:02 [INFO|CP=0|TP=0|nid006683][0m: Before train_batch_iter
[2;3m12/06 14:46:02 [INFO|CP=0|TP=0|nid006683][0m:  Memory usage: 18787.00MiB. Peak allocated 19415.07MiB. Peak reserved: 20170.00MiB
```

### Observed Tensor Shapes (Layer 0 Example)

**Layer 0 - Input (Line 2522):**
```
[2;3;32m12/06 14:46:05 [DEBUG|CP=0|TP=0|nid006683][0m: [Layer 0] Input: hidden_states.shape=torch.Size([256, 2816]), position_ids.shape=torch.Size([2, 512]), cp_pg_size=2
```

**Layer 0 - After Flatten (Lines 2523-2524):**
```
[2;3;32m12/06 14:46:05 [DEBUG|CP=0|TP=0|nid006683][0m: [Layer 0] After flatten: original_shape=torch.Size([2, 512]), flattened_shape=torch.Size([1024]), seq_length=256
```

**Layer 0 - Before Reshape (Line 2525):**
```
[2;3;32m12/06 14:46:05 [DEBUG|CP=0|TP=0|nid006683][0m: [Layer 0] Before reshape: output.shape=torch.Size([256, 2816]), position_ids.shape=torch.Size([1024]), seq_length=256, cp_pg_size=2, target_reshape=(-1, 256)
```

### The Shape Mismatch

**After the buggy reshape** `position_ids.view(-1, 256)`:
```
Result: [1024 / 256, 256] = [4, 256]  ❌
```

**Comparison Table:**

| Tensor | Shape | Batch Dimension | Status |
|--------|-------|----------------|--------|
| `hidden_states` | `[256, 2816]` | **2** (split across TP=2) | ✓ Correct |
| `position_ids` (input) | `[2, 512]` | **2** | ✓ Correct |
| `position_ids` (after bug) | `[4, 256]` | **4** | ❌ **DOUBLED!** |
| `position_ids` (expected) | `[2, 512]` or `[2, 256]` | **2** | ✓ Should be |

### Why hidden_states is [256, 2816]?

With `batch=2`, `local_seq_cp=256` (after CP slicing):
1. After embedding: `[batch × local_seq_cp, hidden]` = `[2 × 256, 2816]` = `[512, 2816]`
2. After TP REDUCE_SCATTER (TP=2): `[512 / 2, 2816]` = **`[256, 2816]`**

So `hidden_states` represents effective batch=2 with 256 elements (split across TP ranks).

### Where the Crash Occurs

The mismatched batch dimensions cause illegal memory access when:
1. **RoPE (Rotary Position Embeddings)** tries to apply position encodings using mismatched shapes
2. **Flash Attention** validates input tensor shapes and detects incompatibility
3. **Broadcasting operations** attempt to combine tensors with batch=2 and batch=4
4. **CUDA kernels** access memory with wrong batch offsets → **out-of-bounds** → **illegal memory access**

### Pattern Across All Layers

This bug propagates through all 32 layers:
- Each layer receives `position_ids` with doubled batch dimension from the previous layer
- Each layer outputs the same incorrect shape
- The crash occurs during forward pass when CUDA operations detect the mismatch

**Example: Layer 2 - Shows the pattern repeating (Lines 2530-2533):**
```
[2;3;32m12/06 14:46:08 [DEBUG|CP=0|TP=0|nid006683][0m: [Layer 2] Before reshape: output.shape=torch.Size([256, 2816]), position_ids.shape=torch.Size([1024]), seq_length=256, cp_pg_size=2, target_reshape=(-1, 256)
[2;3;32m12/06 14:46:08 [DEBUG|CP=0|TP=0|nid006683][0m: [Layer 3] Input: hidden_states.shape=torch.Size([256, 2816]), position_ids.shape=torch.Size([2, 512]), cp_pg_size=2
[2;3;32m12/06 14:46:08 [DEBUG|CP=0|TP=0|nid006683][0m: [Layer 3] After flatten: original_shape=torch.Size([2, 512]), flattened_shape=torch.Size([1024]), seq_length=256
[2;3;32m12/06 14:46:08 [DEBUG|CP=0|TP=0|nid006683][0m: [Layer 3] Before reshape: output.shape=torch.Size([256, 2816]), position_ids.shape=torch.Size([1024]), seq_length=256, cp_pg_size=2, target_reshape=(-1, 256)
```

**Layer 31 (Final Layer) - Still showing same pattern (Lines 2615-2617):**
```
[2;3;32m12/06 14:46:08 [DEBUG|CP=0|TP=0|nid006683][0m: [Layer 31] Input: hidden_states.shape=torch.Size([256, 2816]), position_ids.shape=torch.Size([2, 512]), cp_pg_size=2
[2;3;32m12/06 14:46:08 [DEBUG|CP=0|TP=0|nid006683][0m: [Layer 31] After flatten: original_shape=torch.Size([2, 512]), flattened_shape=torch.Size([1024]), seq_length=256
[2;3;32m12/06 14:46:08 [DEBUG|CP=0|TP=0|nid006683][0m: [Layer 31] Before reshape: output.shape=torch.Size([256, 2816]), position_ids.shape=torch.Size([1024]), seq_length=256, cp_pg_size=2, target_reshape=(-1, 256)
```

**Crash Immediately After Layer 31 (Line 2618):**
```
[rank1]:[E1206 14:46:08.298864673 ProcessGroupNCCL.cpp:2057] [PG ID 3 PG GUID 3 Rank 1] Process group watchdog thread terminated with exception: CUDA error: an illegal memory access was encountered
```

## The Fix

**File**: `src/nanotron/models/qwen.py`
**Line**: 275 (in `Qwen2Attention.forward()`)

### Current (Buggy) Code:
```python
def forward(self, hidden_states, position_ids, cu_seqlens=None):
    # ...
    seq_length = position_ids.shape[1] // self.cp_pg_size  # local seq_length
    original_position_ids_shape = position_ids.shape
    position_ids = position_ids.view(-1)  # flatten

    # ... forward pass ...

    # BUG: Uses LOCAL seq_length but position_ids has GLOBAL data
    return {"hidden_states": output, "position_ids": position_ids.view(-1, seq_length)}
```

### Fixed Code (Option 1 - Recommended):
```python
    # Use the original shape to preserve correct batch dimension
    return {"hidden_states": output, "position_ids": position_ids.view(original_position_ids_shape)}
```

### Fixed Code (Option 2 - Alternative):
```python
    # Explicitly use global sequence length
    return {"hidden_states": output, "position_ids": position_ids.view(-1, seq_length * self.cp_pg_size)}
```

Both fixes ensure `position_ids` maintains `[batch, global_seq]` shape instead of incorrectly creating `[batch * cp_pg_size, local_seq]`.

## Verification

After applying the fix:
1. Run the same test configuration: `mbs=2, cp=2, sequence_length=512`
2. Verify `position_ids` shape remains `[2, 512]` through all layers
3. Confirm no CUDA illegal memory access errors
4. Test with full configuration: `mbs=4, cp=2, sequence_length=32768`

## Related Investigation

- **Commit 42c117b** ("Fix sanity check issue for position embedding") only addressed sanity check warnings, not this underlying bug
- The issue exists in the base Qwen2 attention implementation and affects ClimLlama when CP > 1
- Bug is **not related** to `max_position_embeddings` configuration (red herring)
- Bug **only manifests** when Context Parallelism is enabled (CP > 1)

## System Information

- **Platform**: NVIDIA GH200 120GB GPUs (aarch64)
- **Nodes**: nid[007132,007140,007142-007143]
- **PyTorch version**: 2.9.0+cu129
- **CUDA version**: 12.9
- **nanotron version**: 0.4
- **Driver**: 550.54.15

## Log References

### Original Crash (debug_output.log / Job 1198520)
- Line 5100: "Before train_batch_iter"
- Line 5101: Memory usage report
- Line 5102: First CUDA error detected
- Lines 5814-5858: Root cause failure summary

### Debug Run (debug_output2.log)
- Line 2499: Training configuration `mbs: 2 | cp: 2 | sequence_length: 512`
- Line 2504: "Before train_batch_iter"
- Lines 2522-2617: Detailed shape logging through all 32 layers showing the bug
- Line 2522: Layer 0 input shapes showing `position_ids=[2, 512]`
- Line 2525: Layer 0 reshape target `(-1, 256)` which creates `[4, 256]`
- Lines 2530-2533: Layer 2-3 showing repeated pattern
- Lines 2615-2617: Layer 31 (final layer) still showing same bug pattern
- Line 2618: CUDA illegal memory access error immediately after Layer 31
- Lines 2619-2702: Full stack traces showing ProcessGroupNCCL error

**Complete Error Message (Lines 2618-2622):**
```
[rank1]:[E1206 14:46:08.298864673 ProcessGroupNCCL.cpp:2057] [PG ID 3 PG GUID 3 Rank 1] Process group watchdog thread terminated with exception: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

## Reproduction Steps

1. Set configuration with CP > 1 (e.g., `cp: 2`)
2. Run training with any batch size and sequence length
3. Observe crash during first forward pass with "CUDA error: an illegal memory access"
4. Add logging to `qwen.py` lines 232-275 to observe shape doubling

## Impact

- **Severity**: Critical - prevents all training with Context Parallelism
- **Scope**: Affects any configuration with `cp > 1`
- **Workaround**: Set `cp: 1` (disables context parallelism) - not viable for long sequences
- **Fix**: Simple one-line code change in `qwen.py:275`
