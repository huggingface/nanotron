# Timestamp Tracking for Weaved Token Dataset

## Overview

This document specifies how timestamps are tracked and stored for weaved token datasets to enable temporal positional encoding in ClimLlama.

---

## Timestamp Array Format

### Global Timestamp File

**File**: `timestamps.npy` (stored alongside indexed dataset `.idx` and `.bin` files)

**Format**:
```python
timestamps.npy: (N_docs,) datetime64[s] or int64
```

Where:
- `N_docs`: Total number of documents in the indexed dataset
- Each element represents the **base timestamp** of the corresponding document
- Array index matches document index in the indexed dataset

**Storage in datetime64[s]**:

```python
timestamps = np.array([
    np.datetime64('2020-01-01T00:00:00'),
    np.datetime64('2020-01-01T06:00:00'),
    np.datetime64('2020-01-01T12:00:00'),
    ...
], dtype='datetime64[s]')
```

---

## Document Boundary Strategy

### Per-Sample Documents

Each time sample (variable at one timestamp) becomes one document:

```
Document 0: temperature_850 at 2020-01-01T00:00:00
Document 1: u_wind_850 at 2020-01-01T00:00:00
Document 2: v_wind_850 at 2020-01-01T00:00:00
...
Document M-1: msl at 2020-01-01T00:00:00
Document M: temperature_850 at 2020-01-01T06:00:00
...
```

**Timestamp array**:
```python
timestamps[0] = datetime64('2020-01-01T00:00:00')  # temperature_850
timestamps[1] = datetime64('2020-01-01T00:00:00')  # u_wind_850
timestamps[2] = datetime64('2020-01-01T00:00:00')  # v_wind_850
...
timestamps[M] = datetime64('2020-01-01T06:00:00')  # temperature_850
```

**Characteristics**:
- Same timestamp repeated for all variables at that time
- Fine-grained document boundaries (good for flexible batching)
- Variable information embedded in special tokens within document

### Per-Timestep Documents (Alternative)

All variables at one timestamp form one document:

```
Document 0: [temp_850, u_850, v_850, ..., msl] at 2020-01-01T00:00:00
Document 1: [temp_850, u_850, v_850, ..., msl] at 2020-01-01T06:00:00
Document 2: [temp_850, u_850, v_850, ..., msl] at 2020-01-01T12:00:00
...
```

**Timestamp array**:
```python
timestamps[0] = datetime64('2020-01-01T00:00:00')
timestamps[1] = datetime64('2020-01-01T06:00:00')
timestamps[2] = datetime64('2020-01-01T12:00:00')
...
```

**Characteristics**:
- One timestamp per document (1:1 mapping)
- Coarser document boundaries (less batching flexibility)
- All variables for a timestep kept together

**Recommended**: Per-sample documents for flexibility in variable subsampling during training.

---

## Creation During Weaving

### In `scripts/weave_and_index.py`

```python
from atmtokenizer.utils.weaver import TokenWeaver
from megatron.data.indexed_dataset import IndexedDatasetBuilder
import numpy as np

def weave_and_index(
    npz_path: str,
    output_prefix: str,
    tokenizer_config: dict,
    weaving_strategy: str = "split"
):
    """Convert NPZ tokens to indexed dataset with timestamps."""

    # Initialize weaver
    weaver = TokenWeaver(
        npz_path=npz_path,
        tokenizer_config=tokenizer_config,
        strategy=weaving_strategy
    )

    # Initialize dataset builder
    builder = IndexedDatasetBuilder(f"{output_prefix}.bin")

    # Track timestamps for each document
    timestamps = []

    # Iterate through all samples
    for idx in range(len(weaver)):
        # Get weaved token sequence with metadata
        result = weaver[idx]  # Returns dict with 'tokens' and metadata
        tokens = result['tokens']  # 1D int32 array
        timestamp = result['timestamp']  # datetime64 or datetime object

        # Add document to indexed dataset
        builder.add_document(tokens)

        # Record timestamp
        timestamps.append(timestamp)

    # Finalize indexed dataset
    builder.finalize(f"{output_prefix}.idx")

    # Save timestamp array
    timestamps_array = np.array(timestamps, dtype='datetime64[s]')
    np.save(f"{output_prefix}_timestamps.npy", timestamps_array)

    print(f"Created indexed dataset with {len(timestamps)} documents")
    print(f"Timestamp range: {timestamps_array[0]} to {timestamps_array[-1]}")
```

---

## Loading in ClimLlamaDataset

### Timestamp Lookup

```python
class ClimLlamaDataset(GPTDataset):
    """Dataset for ClimLlama with temporal positional encoding."""

    def __init__(self, data_prefix, indexed_dataset, ...):
        super().__init__(...)

        # Load global timestamp array
        timestamp_path = f"{data_prefix}_timestamps.npy"
        if os.path.exists(timestamp_path):
            self.timestamps = np.load(timestamp_path)
            print(f"Loaded {len(self.timestamps)} timestamps from {timestamp_path}")
        else:
            print(f"Warning: No timestamp file found at {timestamp_path}")
            self.timestamps = None

    def _get_document_timestamp(self, doc_idx: int) -> np.datetime64:
        """Get timestamp for a document."""
        if self.timestamps is not None:
            return self.timestamps[doc_idx]
        else:
            # Fallback: return dummy timestamp
            return np.datetime64('1970-01-01T00:00:00')

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Return item with temporal features computed from document timestamp."""

        # Get base item from GPTDataset
        base_item = super().__getitem__(idx)

        # Determine which document(s) this sample spans
        # Note: GPTDataset samples may span multiple documents
        # We need to track document boundaries within the sample

        # For simplicity, assume each sample comes from a single document
        # (This requires careful handling in the sampling strategy)
        doc_idx = self._get_document_index_for_sample(idx)
        timestamp = self._get_document_timestamp(doc_idx)

        # Compute temporal features
        temporal_features = self._compute_temporal_features(timestamp)
        # Returns: (cos_hour, sin_hour, cos_day, sin_day)

        # Broadcast temporal features to all tokens in sequence
        seq_len = len(base_item["input_ids"])
        spatial_temporal_features = np.zeros((seq_len, 7), dtype=np.float32)

        # Parse tokens to get spatial positions (x, y, z) per token
        var_idx, res_idx, spatial_features = self._parse_special_tokens(
            base_item["input_ids"]
        )

        # Fill spatial features (x, y, z)
        spatial_temporal_features[:, 0:3] = spatial_features

        # Fill temporal features (same for all tokens in this sample)
        spatial_temporal_features[:, 3] = temporal_features[0]  # cos_hour
        spatial_temporal_features[:, 4] = temporal_features[1]  # sin_hour
        spatial_temporal_features[:, 5] = temporal_features[2]  # cos_day
        spatial_temporal_features[:, 6] = temporal_features[3]  # sin_day

        return {
            "input_ids": base_item["input_ids"],
            "var_idx": var_idx,
            "res_idx": res_idx,
            "leadtime_idx": np.zeros(seq_len, dtype=np.int64),  # TODO: forecast lead time
            "spatial_temporal_features": spatial_temporal_features,
        }
```

### Handling Multi-Document Samples

If training samples can span multiple documents (e.g., long context windows), we need to track document boundaries:

```python
def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
    """Handle samples that may span multiple documents."""

    base_item = super().__getitem__(idx)
    seq_len = len(base_item["input_ids"])

    # Get document indices for each token in the sample
    doc_indices = self._get_document_indices_for_tokens(idx, seq_len)

    # Compute temporal features per token based on document
    spatial_temporal_features = np.zeros((seq_len, 7), dtype=np.float32)

    for token_idx, doc_idx in enumerate(doc_indices):
        timestamp = self._get_document_timestamp(doc_idx)
        temporal_features = self._compute_temporal_features(timestamp)

        spatial_temporal_features[token_idx, 3:7] = temporal_features

    # ... rest of the processing
```

---

## Metadata File

### Additional Dataset Metadata

Store timestamp range and frequency in dataset metadata:

**File**: `{output_prefix}_metadata.json`

```json
{
  "model_config": {
    "resolutions": [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 32],
    "vocab_size": 32768,
    "special_tokens": {...}
  },
  "temporal_info": {
    "start_timestamp": "2020-01-01T00:00:00",
    "end_timestamp": "2020-12-31T18:00:00",
    "time_step_hours": 6,
    "total_timesteps": 1460,
    "timestamps_per_document": "one",
    "comment": "Each document represents one variable at one timestep"
  },
  "document_info": {
    "total_documents": 21900,
    "documents_per_timestep": 15,
    "variables": ["temperature_850", "u_850", "v_850", ...]
  },
  "created_at": "2025-01-15T10:30:00"
}
```

---

## Temporal Feature Computation

### Implementation in ClimLlamaDataset

```python
def _compute_temporal_features(self, timestamp: np.datetime64) -> np.ndarray:
    """Compute temporal features from timestamp.

    Args:
        timestamp: numpy datetime64 object

    Returns:
        np.ndarray: (4,) array with [cos_hour, sin_hour, cos_day, sin_day]
    """
    # Convert to Python datetime for easier manipulation
    dt = timestamp.astype('datetime64[s]').astype(datetime.datetime)

    # Hour of day (0-23) + fractional minutes
    hour_of_day = dt.hour + dt.minute / 60.0
    hour_angle = hour_of_day / 24.0 * 2 * np.pi

    # Day of year (1-365 or 366)
    day_of_year = dt.timetuple().tm_yday
    days_in_year = 366 if self._is_leap_year(dt.year) else 365
    day_angle = (day_of_year - 1) / days_in_year * 2 * np.pi

    return np.array([
        np.cos(hour_angle),
        np.sin(hour_angle),
        np.cos(day_angle),
        np.sin(day_angle),
    ], dtype=np.float32)

@staticmethod
def _is_leap_year(year: int) -> bool:
    """Check if year is a leap year."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
```

---

## File Structure Summary

After running `weave_and_index.py`, the output directory contains:

```
/path/to/output/
├── train.bin              # Indexed dataset binary (token sequences)
├── train.idx              # Indexed dataset index (document pointers)
├── train_timestamps.npy   # Timestamp array (N_docs,) datetime64[s]
├── train_metadata.json    # Dataset metadata including temporal info
├── tokenizer_config.json  # Tokenizer configuration with special tokens
└── README.md              # Dataset documentation
```

---

## Usage Example

### Creating Dataset

```bash
python scripts/weave_and_index.py \
  --input-npz=/path/to/tokens/202001.npz \
  --output-prefix=/path/to/indexed/train \
  --tokenizer-config=config/tokenizer_config.json \
  --strategy=split
```

### Loading in Training

```python
from nanotron.data.climllama_dataset import ClimLlamaDataset

# Dataset automatically loads timestamps
dataset = ClimLlamaDataset(
    data_prefix="/path/to/indexed/train",
    indexed_dataset=indexed_ds,
    ...
)

# Each sample includes temporal features
sample = dataset[0]
# sample["spatial_temporal_features"][:, 3:7] contains temporal encodings
```

---

## Implementation Checklist

- [ ] Modify `scripts/weave_and_index.py` to track and save timestamps
- [ ] Implement `TokenWeaver.get_metadata()` to return timestamp
- [ ] Add timestamp loading in `ClimLlamaDataset.__init__()`
- [ ] Implement `_compute_temporal_features()` in `ClimLlamaDataset`
- [ ] Handle multi-document samples (if needed)
- [ ] Add timestamp range to dataset metadata
- [ ] Write unit tests for temporal feature computation
- [ ] Document timestamp format in dataset README

---

## Design Rationale

### Why Separate Timestamp File?

1. **Efficiency**: Don't duplicate timestamps in the binary token file
2. **Flexibility**: Easy to update or regenerate timestamps without touching tokens
3. **Memory**: Timestamps only loaded once, not per sample
4. **Compatibility**: Doesn't require changes to indexed dataset format

### Why Document-Level Timestamps?

1. **Simplicity**: Clear 1:1 or N:1 mapping between documents and timestamps
2. **Assumption**: All tokens in a document share the same base timestamp
3. **Forecast Lead Time**: Can be added as a separate attribute (per token or per document)

### Future Extensions

- [ ] Add forecast lead time tracking (separate array or embedded in metadata)
- [ ] Support for time-varying temporal features within a document
- [ ] Efficient handling of very long sequences spanning multiple days
