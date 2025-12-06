# Training Error Report 02

**Date:** 2025-12-06
**Time:** 18:42:22
**Log File:** `debug_output3.log`
**Status:** RESOLVED (root cause identified)

---

## Executive Summary

Training initially failed with a CUDA CUBLAS error during the forward pass of the first training step. The issue was traced to using flash-attn with context parallelism (CP) > 1; switching to ring-attn for CP > 1 resolved the failure. The error occurred in the attention output projection layer (`o_proj`) during a tensor parallel row-wise linear operation on rank 2.

---

## Error Details

### Primary Error

**Error Type:** `RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED`

**Error Message:**
```
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasGemmEx( handle, opa, opb, m, n, k, &falpha, a, CUDA_R_16BF, lda, b, CUDA_R_16BF, ldb, &fbeta, c, std::is_same_v<C_Dtype, float> ? CUDA_R_32F : CUDA_R_16BF, ldc, compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`
```
*Reference: debug_output3.log, end of file*

### Failure Context

**Failed Rank:** 2 (local_rank: 0)
**Host:** nid006752
**Exit Code:** 1
**Timestamp:** 2025-12-06_18:42:22

---

## Stack Trace Analysis

### Full Call Stack

The error propagated through the following call stack (from top to bottom):

1. **Training Loop** (`trainer.py:621`)
   ```
   [rank2]: File ".../src/nanotron/trainer.py", line 621, in training_step
   [rank2]:   outputs = self.pipeline_engine.train_batch_iter(
   ```

2. **Pipeline Engine** (`engine.py:295`)
   ```
   [rank2]: File ".../src/nanotron/parallel/pipeline_parallel/engine.py", line 295, in train_batch_iter
   [rank2]:   output = self.forward(context=context, state=state, micro_batch=micro_batch, model=model)
   ```

3. **Model Forward Pass** (`climllama.py:493`)
   ```
   [rank2]: File ".../src/nanotron/models/climllama.py", line 493, in forward
   [rank2]:   sharded_logits = self.model(
   ```

4. **Decoder Layer** (`climllama.py:401`)
   ```
   [rank2]: File ".../src/nanotron/models/climllama.py", line 401, in forward
   [rank2]:   decoder_states = decoder_layer(**decoder_states)
   ```

5. **Pipeline Parallel Block** (`block.py:151`)
   ```
   [rank2]: File ".../src/nanotron/parallel/pipeline_parallel/block.py", line 151, in forward
   [rank2]:   output = self.pp_block(**new_kwargs)
   ```

6. **Qwen Core Forward** (`qwen.py:654` → `qwen.py:624`)
   ```
   [rank2]: File ".../src/nanotron/models/qwen.py", line 654, in forward
   [rank2]:   hidden_states, position_ids, cu_seqlens = self._core_forward(hidden_states, position_ids, cu_seqlens)
   [rank2]: File ".../src/nanotron/models/qwen.py", line 624, in _core_forward
   [rank2]:   output = self.attn(hidden_states=hidden_states, position_ids=position_ids, cu_seqlens=cu_seqlens)
   ```

7. **Attention Output Projection** (`qwen.py:263`) ⚠️ **Critical Location**
   ```
   [rank2]: File ".../src/nanotron/models/qwen.py", line 263, in forward
   [rank2]:   output = self.o_proj(attn_output)
   ```

8. **Tensor Parallel Row Linear** (`tensor_parallel/nn.py:171`)
   ```
   [rank2]: File ".../src/nanotron/parallel/tensor_parallel/nn.py", line 171, in forward
   [rank2]:   return row_linear(
   ```

9. **Linear Operation** (`tensor_parallel/functional.py:582`) ❌ **FAILURE POINT**
   ```
   [rank2]: File ".../src/nanotron/parallel/tensor_parallel/functional.py", line 582, in forward
   [rank2]:   out = F.linear(tensor, weight, bias)
   [rank2]: RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED
   ```

---

## System Configuration

### Model Configuration

From log initialization (debug_output3.log, lines ~6-20):

```
Building model
Total number of parameters: 3.28B (6253.04MiB)
Local number of parameters: 1.64B (3126.52MiB)
[After model building] Memory usage: 3152.48MiB. Peak allocated: 3152.49MiB Peak reserved: 3318.00MiB
```

**Model Checkpoint:**
```
Loading weights from /iopsstor/scratch/cscs/lhuang/FoundationModel/outputs/megatron_checkpoints/exp_fsq_245_split_vocab32768/llama_3B_vocab_32768/iter_0204000_nanotron_climllama
Loading weights: 100%|██████████| 198/198 [00:01<00:00, 103.39it/s]
```

### Training Configuration

From log (debug_output3.log, training setup section):

```
micro_batch_size=4,
batch_accumulation_per_replica=32,
sequence_length=8192
global_batch_size=128
train_steps=3
start_iteration_step=0
consumed_tokens_total=0
```

**Parallelism Setup:**
- Context Parallel (CP): 2
- Tensor Parallel (TP): 2
- Pipeline Parallel: Enabled
- Total Ranks: 4 (CP × TP)

### Hardware Configuration

From system info (debug_output3.log, environment section):

```
GPU models and configuration:
GPU 0: NVIDIA GH200 120GB
GPU 1: NVIDIA GH200 120GB
GPU 2: NVIDIA GH200 120GB
GPU 3: NVIDIA GH200 120GB

Nvidia driver version: 550.54.15
CUDA runtime version: 12.9.86
PyTorch version: 2.9.0+cu129
cuDNN version: 9.10.2
```

**Precision:** BFloat16 (CUDA_R_16BF) as indicated in the error message

### Software Versions

From log (debug_output3.log, libraries section):

```
nanotron version: 0.4
torch version: 2.9.0+cu129
transformers version: 4.57.1
datasets version: 4.4.1
flash-attn version: 2.8.3
numpy version: 2.3.4
```

---

## Training Progress Before Failure

### Initialization Status

From log (debug_output3.log, training start section):

```
[INFO] Instantiating MegatronPretrainingSampler with total_samples: 384 and consumed_samples: 0
[INFO] mbs: 4 | grad_accum: 32 | cp: 2 | sequence_length: 8192 | global_batch_size: 128 | train_steps: 3 | start_iteration_step: 0 | consumed_tokens_total: 0
[INFO] Before train_batch_iter
```

### Micro-batch Processing

From log (debug_output3.log, batch processing section):

```
[DEBUG] DP 0 batch 0 [0, 1, 2, 3] self.consumed_samples 0
[DEBUG] DP 0 batch 1 [4, 5, 6, 7] self.consumed_samples 4
[DEBUG] DP 0 batch 2 [8, 9, 10, 11] self.consumed_samples 8
[DEBUG] Forward micro batch id: 1
```

**Observation:** The failure occurred during the forward pass of micro batch 1 (the second micro-batch in the gradient accumulation loop).

---

## Root Cause Analysis

### Root Cause: flash-attn with CP > 1

- The CUBLAS failure was triggered when using flash-attn while context parallelism was enabled with CP > 1.
- Switching the attention backend to ring-attn for CP > 1 removes the failure; training proceeds normally with the same configuration otherwise unchanged.

---

## Recommendations

### Immediate Actions

1. **Enable Gradient/Activation Checking**
   - Add NaN/Inf detection hooks in forward pass
   - Focus on attention layer outputs
   - **Location to modify:** `qwen.py:263` (before `self.o_proj`)

2. **Inspect Training Data**
   - Check first few batches for extreme values
   - Validate data loading and preprocessing
   - **Data path:** `/capstor/store/cscs/swissai/a122/ycheng/ClimLlama/token_pred_6/245-fsq_2025-02-07-14-06-33-00230000/`

3. **Reduce Precision Sensitivity**
   - Consider mixed precision with loss scaling
   - Try FP32 for attention computations
   - Enable gradient clipping

### Testing Steps

1. **Reduce Batch Size**
   ```yaml
   micro_batch_size: 2  # Currently 4
   ```

2. **Add Debugging Output**
   - Print tensor statistics before `o_proj` layer
   - Log min/max/mean values of attention outputs

3. **Validate Checkpoint**
   - Verify loaded weights don't contain NaN/Inf
   - Test with random initialization to isolate checkpoint issues

4. **Run GPU Diagnostics**
   ```bash
   nvidia-smi -q
   nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total --format=csv
   ```

### Code Modifications to Consider

**Location:** `src/nanotron/models/qwen.py:263`

Add validation before output projection:
```python
# Before: output = self.o_proj(attn_output)
assert not torch.isnan(attn_output).any(), "NaN detected in attention output"
assert not torch.isinf(attn_output).any(), "Inf detected in attention output"
output = self.o_proj(attn_output)
```

**Location:** `src/nanotron/parallel/tensor_parallel/functional.py:582`

Add tensor statistics logging:
```python
# Before: out = F.linear(tensor, weight, bias)
if torch.isnan(tensor).any() or torch.isinf(tensor).any():
    print(f"Invalid values in input tensor: min={tensor.min()}, max={tensor.max()}")
out = F.linear(tensor, weight, bias)
```

---

## Related Errors

### NCCL Warning

From log (end of file):
```
[W1206 18:42:22] Warning: destroy_process_group() was not called before program exit, which can leak resources.
```

**Status:** Secondary issue caused by abrupt termination. Not root cause.

### Process Termination

From log (end of file):
```
W1206 18:42:22.964000 174632 torch/distributed/elastic/multiprocessing/api.py:908] Sending process 174663 closing signal SIGTERM
W1206 18:42:22.964000 174632 torch/distributed/elastic/multiprocessing/api.py:908] Sending process 174664 closing signal SIGTERM
W1206 18:42:22.964000 174632 torch/distributed/elastic/multiprocessing/api.py:908] Sending process 174665 closing signal SIGTERM
```

**Status:** Clean shutdown after error detection. Expected behavior.

---

## Next Steps

1. ✅ Use ring-attn when CP > 1 (fix applied; training passes first step)

---

## Appendix: File References

- **Error occurred:** `src/nanotron/parallel/tensor_parallel/functional.py:582`
- **Critical layer:** `src/nanotron/models/qwen.py:263`
- **Training loop:** `src/nanotron/trainer.py:621`
- **Pipeline engine:** `src/nanotron/parallel/pipeline_parallel/engine.py:295`
- **Log file:** `debug_output3.log` (514.2KB)
