# Megatron Sampler Implementation Guide

## Overview

This guide explains the implementation of Megatron-style samplers for use with IndexedDataset and BlendableDataset in Nanotron. The Megatron samplers provide better sample tracking, deterministic resumption, and are optimized for large-scale distributed training.

## What Was Implemented

### 1. `get_megatron_sampler()` function
**Location**: [src/nanotron/data/samplers.py:114-184](src/nanotron/data/samplers.py#L114-L184)

A unified interface to create three types of Megatron samplers:
- **`sequential`**: `MegatronPretrainingSampler` - Deterministic, sequential sampling
- **`random`**: `MegatronPretrainingRandomSampler` - Epoch-based shuffling
- **`cyclic`**: `MegatronPretrainingCyclicSampler` - Cyclic without shuffling

### 2. `get_train_dataloader_with_megatron_sampler()` function
**Location**: [src/nanotron/data/dataloader.py:379-490](src/nanotron/data/dataloader.py#L379-L490)

A DataLoader factory function specifically designed for IndexedDataset/BlendableDataset that uses Megatron samplers instead of standard PyTorch DistributedSampler.

### 3. Updated `run_train.py`
**Location**: [run_train.py:219-236](run_train.py#L219-L236)

Modified the IndexedDataset dataloader creation to use the new Megatron sampler function.

### 4. Configuration Support
**Location**: [src/nanotron/config/config.py:224-225](src/nanotron/config/config.py#L224-L225)

Added two new fields to `IndexedDatasetsArgs`:
- `sampler_type`: Choose sampler type ("sequential", "random", or "cyclic")
- `pad_samples_to_global_batch_size`: Whether to pad the last batch

## Usage

### Using prepare_training_config.py (Recommended)

The easiest way to create a configuration with Megatron samplers is using the `climllama/prepare_training_config.py` script:

```bash
python climllama/prepare_training_config.py \
    --checkpoint_path /path/to/checkpoint \
    --data_prefix "0.6,/data/fineweb,0.4,/data/code" \
    --output_config config_train.yaml \
    --mode pretrain \
    --sampler_type sequential \
    --pad_samples_to_global_batch_size \
    --train_steps 10000 \
    --dp 4 --tp 2 --pp 1
```

Options for `--sampler_type`:
- `sequential`: Deterministic, sequential sampling (default)
- `random`: Epoch-based shuffling
- `cyclic`: Cyclic without shuffling

### YAML Configuration Example

```yaml
data_stages:
  - name: indexed_stage
    start_training_step: 1
    data:
      dataset:
        data_prefix:
          - 0.6
          - /data/fineweb/fw_edu
          - 0.4
          - /data/code/stack_mix
        splits_string: "999,1,0"
        validation_drop_last: true
        eod_mask_loss: false
        index_mapping_dir: /scratch/nanotron_index_cache
        fim_rate: 0.0
        skip_warmup: false
        # NEW: Megatron sampler configuration
        sampler_type: "sequential"  # Options: "sequential", "random", "cyclic"
        pad_samples_to_global_batch_size: true
      num_loading_workers: 8
      seed: 6
```

### Programmatic Usage

```python
from nanotron.data.nemo_dataset import build_dataset
from nanotron.data.dataloader import get_train_dataloader_with_megatron_sampler

# Build IndexedDataset or BlendableDataset
train_dataset = build_dataset(
    cfg=dataset_cfg,
    tokenizer=tokenizer,
    data_prefix=[0.6, "/data/fineweb", 0.4, "/data/code"],
    num_samples=10000,
    seq_length=2048,
    seed=42,
    skip_warmup=False,
    name="train",
    parallel_context=parallel_context,
)

# Create dataloader with Megatron sampler
dataloader = get_train_dataloader_with_megatron_sampler(
    train_dataset=train_dataset,
    sequence_length=2048,
    parallel_context=parallel_context,
    input_pp_rank=0,
    output_pp_rank=0,
    micro_batch_size=4,
    global_batch_size=128,  # Must be micro_batch_size * dp_size * gradient_accumulation_steps
    consumed_train_samples=0,
    dataloader_num_workers=2,
    seed_worker=42,
    sampler_type="sequential",  # Choose: "sequential", "random", or "cyclic"
    pad_samples_to_global_batch_size=False,
)

# Use dataloader for training
for batch in dataloader:
    # batch contains 'input_ids', 'input_mask', 'label_ids', 'label_mask'
    ...
```

## Key Differences from Standard Sampler

| Feature | Standard (`get_sampler`) | Megatron (`get_megatron_sampler`) |
|---------|-------------------------|-----------------------------------|
| **Sampler Class** | `DistributedSampler` or `DistributedSamplerWithLoop` | `MegatronPretrainingSampler` variants |
| **DataLoader Parameter** | `sampler=...` + `batch_size=...` | `batch_sampler=...` (no batch_size) |
| **Global Batch Size** | Not directly specified | Explicitly required parameter |
| **Sample Tracking** | Uses `SkipBatchSampler` wrapper | Built-in `consumed_samples` tracking |
| **Resume Behavior** | May have slight variations | Deterministic with exact same order |
| **Best For** | HuggingFace datasets | IndexedDataset / BlendableDataset |

## How BlendableDataset Works with Megatron Samplers

The integration is seamless:

1. **BlendableDataset** pre-computes blending indices during initialization
2. **MegatronSampler** generates sequential indices: `[0, 1, 2, 3, ...]`
3. **BlendableDataset.__getitem__(idx)** maps each index to:
   - A specific dataset: `dataset_idx = self.dataset_index[idx]`
   - A sample within that dataset: `sample_idx = self.dataset_sample_index[idx]`

```python
# BlendableDataset internals
def __getitem__(self, idx):
    dataset_idx = self.dataset_index[idx]        # Which dataset? (pre-computed)
    sample_idx = self.dataset_sample_index[idx]  # Which sample? (pre-computed)
    return self.datasets[dataset_idx][sample_idx + offset]
```

This means:
- Blending is **transparent** to the sampler
- Sampler only needs to generate indices from `[0, len(dataset))`
- Dataset mixing ratios are handled by BlendableDataset's pre-computed indices

## Sampler Types Explained

### Sequential Sampler (`sampler_type="sequential"`)
- **Behavior**: Deterministic, processes samples in order
- **Resumption**: Exact same data order on resume
- **Use Case**: Standard pretraining, reproducible experiments
- **Class**: `MegatronPretrainingSampler`

### Random Sampler (`sampler_type="random"`)
- **Behavior**: Shuffles data with epoch-based seeding
- **Resumption**: Reproducible with same seed and epoch
- **Use Case**: Training that benefits from randomization
- **Class**: `MegatronPretrainingRandomSampler`

### Cyclic Sampler (`sampler_type="cyclic"`)
- **Behavior**: Cycles through dataset without shuffling
- **Resumption**: Deterministic cycling
- **Use Case**: Multi-epoch training without randomization
- **Class**: `MegatronPretrainingCyclicSampler`
- **Note**: Does NOT support `pad_samples_to_global_batch_size`

## Important Parameters

### `global_batch_size`
Must satisfy: `global_batch_size = micro_batch_size × dp_size × gradient_accumulation_steps`

Where:
- `micro_batch_size`: Batch size per DP rank
- `dp_size`: Number of data parallel ranks
- `gradient_accumulation_steps`: Number of micro-batches per global batch

### `consumed_samples`
Tracks the total number of samples consumed across **all** data parallel ranks. This is crucial for:
- Deterministic resumption
- Correct checkpoint recovery
- Avoiding duplicate/skipped samples

### `pad_samples_to_global_batch_size`
- When `True`: Pads the last batch to match `global_batch_size` (uses -1 padding)
- When `False`: Last batch may be smaller
- **Not supported** by `MegatronPretrainingCyclicSampler`

## Benefits

1. **Deterministic Resumption**: Sequential sampler guarantees exact same data order on resume
2. **Proper Sample Accounting**: Built-in tracking of consumed samples across DP ranks
3. **Optimized for Scale**: Designed for multi-billion parameter models
4. **BlendableDataset Integration**: Works seamlessly with weighted dataset blending
5. **Flexible Strategies**: Three sampler types for different training scenarios
6. **Global Batch Size Awareness**: Ensures correct batch formation across distributed setup

## Testing

To verify the implementation works:

```bash
# Run training with IndexedDataset and Megatron sampler
python run_train.py --config-file your_config.yaml
```

Look for the log message:
```
Using Megatron sampler type: sequential
```

## Files Modified

1. **`src/nanotron/data/samplers.py`**: Added `get_megatron_sampler()` function
2. **`src/nanotron/data/dataloader.py`**: Added `get_train_dataloader_with_megatron_sampler()` function
3. **`src/nanotron/config/config.py`**: Added `sampler_type` and `pad_samples_to_global_batch_size` fields
4. **`run_train.py`**: Updated IndexedDataset dataloader creation to use Megatron sampler

## Backward Compatibility

The changes are **fully backward compatible**:
- Old configs without `sampler_type` will default to `"sequential"`
- Old configs without `pad_samples_to_global_batch_size` will default to `False`
- HuggingFace datasets continue to use the standard `get_train_dataloader()` function
- Nanoset datasets continue to use their specialized dataloader

## Common Issues & Troubleshooting

### Issue: `global_batch_size` doesn't match micro_batch_size × dp_size
**Error**: Assertion error in MegatronSampler initialization
**Solution**: Ensure `global_batch_size = micro_batch_size × dp_size × gradient_accumulation_steps`

### Issue: Last batch is dropped unexpectedly
**Solution**: Set `drop_last=False` or enable `pad_samples_to_global_batch_size=True`

### Issue: Different data on resume
**Solution**: Use `sampler_type="sequential"` for deterministic behavior and ensure `consumed_samples` is correctly loaded from checkpoint

## Example Configurations

### Small-Scale Experiment
```yaml
sampler_type: "sequential"
pad_samples_to_global_batch_size: false
```

### Large-Scale Training with Randomization
```yaml
sampler_type: "random"
pad_samples_to_global_batch_size: false
```

### Multi-Epoch Cyclic Training
```yaml
sampler_type: "cyclic"
pad_samples_to_global_batch_size: false  # Must be false for cyclic
```

## References

- Original Megatron-LM samplers: https://github.com/NVIDIA/Megatron-LM
- MegatronPretrainingSampler: [src/nanotron/data/samplers.py:188](src/nanotron/data/samplers.py#L188)
- BlendableDataset: [src/nanotron/data/nemo_dataset/blendable_dataset.py:43](src/nanotron/data/nemo_dataset/blendable_dataset.py#L43)
