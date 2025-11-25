# Implementation Plan: ClimLlama with Hybrid Positional Embeddings

## Overview
Create a ClimLlama model that extends Qwen2 architecture with **hybrid positional embeddings**:
- **Learned & Fixed Absolute PE**: For climate-specific spatial-temporal information
- **RoPE**: For relative position encoding (already present in Qwen2)

## Architecture Background

### Current Qwen2 Architecture
- **Location**: [src/nanotron/models/qwen.py](../src/nanotron/models/qwen.py)
- **Key Classes**:
  - `Qwen2ForTraining` (lines 896-1021): Main training model
  - `Qwen2Model` (lines 680-851): Core model with pipeline blocks
  - `Qwen2Attention` (lines 196-330): Attention with RoPE support
  - `Embedding` (line 662): Token embedding layer

### Position Encoding Strategy
Based on [note_collator_position_id.md](note_collator_position_id.md):
- `position_ids` currently used **only for label masking** at document boundaries
- RoPE uses implicit sequential positions during training (via flash attention)
- To support custom absolute PE, we need to:
  1. Pass position embeddings through the model
  2. Add them to token embeddings at input layer
  3. Keep RoPE unchanged in attention layers

---

## Implementation Plan

### Phase 1: Configuration and Model Structure

#### 1.1 Create ClimLlamaConfig
**File**: `src/nanotron/config/models_config.py`

Add a new configuration class extending `Qwen2Config`:

```python
@dataclass
class ClimLlamaConfig(Qwen2Config):
    """Configuration for ClimLlama with climate-specific positional embeddings."""

    # Absolute positional embedding parameters
    # When setting to false, it should be compatible with Qwen2/Llama Model
    use_absolute_position_embeddings: bool = True

    # Embedding dimensions for different position types
    var_vocab_size: int = 100  # Number of climate variables
    res_vocab_size: int = 10   # Number of resolution levels
    leadtime_vocab_size: int = 240  # Lead times in hours (0-239 for 10 days)

    # Spatial-temporal continuous position encoding
    use_spatial_temporal_encoding: bool = True
    spatial_temporal_encoding_dim: int = 128  # Dimension for encoding x,y,z,time features
```

**Key Design Decisions**:
- Inherit from `Qwen2Config` to maintain compatibility
- Separate discrete embeddings (var, res, leadtime) from continuous encodings (spatial-temporal)
- Configurable dimensions for flexibility
- Grid size (resolutions) is NOT in model config - it's loaded from the dataset's `metadata.json` file which contains the resolution hierarchy from the original tokenizer training configuration

#### 1.2 Create ClimLlamaEmbedding
**File**: `src/nanotron/models/climllama.py` (new file)

Custom embedding layer that combines:
1. Token embeddings (from vocabulary)
2. Variable index embeddings
3. Resolution level embeddings
4. Lead time embeddings
5. Spatial-temporal continuous position encoding (x, y, z, time)

```python
class ClimLlamaEmbedding(nn.Module, AttachableStore):
    def __init__(self, tp_pg: dist.ProcessGroup, config: ClimLlamaConfig, parallel_config: Optional[ParallelismArgs]):
        super().__init__()

        # Token embeddings (standard)
        self.token_embedding = TensorParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            pg=tp_pg,
            mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
        )

        # Discrete position embeddings
        if config.use_absolute_position_embeddings:
            self.var_embedding = TensorParallelEmbedding(
                num_embeddings=config.var_vocab_size,
                embedding_dim=config.hidden_size,
                pg=tp_pg,
                mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
            )

            self.res_embedding = TensorParallelEmbedding(
                num_embeddings=config.res_vocab_size,
                embedding_dim=config.hidden_size,
                pg=tp_pg,
                mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
            )

            self.leadtime_embedding = TensorParallelEmbedding(
                num_embeddings=config.leadtime_vocab_size,
                embedding_dim=config.hidden_size,
                pg=tp_pg,
                mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
            )

        # Continuous spatial-temporal encoding
        if config.use_spatial_temporal_encoding:
            # MLP to project 7D spatial-temporal features to hidden_size
            # Input: [x, y, z, cos_hour, sin_hour, cos_day, sin_day]
            self.spatial_temporal_proj = TensorParallelColumnLinear(
                in_features=7,
                out_features=config.spatial_temporal_encoding_dim,
                pg=tp_pg,
                mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
                bias=True,
            )
            self.spatial_temporal_proj2 = TensorParallelColumnLinear(
                in_features=config.spatial_temporal_encoding_dim,
                out_features=config.hidden_size,
                pg=tp_pg,
                mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
                bias=True,
            )
            self.activation = nn.GELU()

        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,  # [batch, seq_len]
        input_mask: torch.Tensor,  # [batch, seq_len]
        var_idx: Optional[torch.Tensor] = None,  # [batch, seq_len]
        res_idx: Optional[torch.Tensor] = None,  # [batch, seq_len]
        leadtime_idx: Optional[torch.Tensor] = None,  # [batch, seq_len]
        spatial_temporal_features: Optional[torch.Tensor] = None,  # [batch, seq_len, 7]
    ):
        # Token embeddings
        input_ids = input_ids.transpose(0, 1)  # [seq_len, batch]
        token_embeds = self.token_embedding(input_ids)  # [seq_len, batch, hidden]

        # Add discrete position embeddings
        if self.config.use_absolute_position_embeddings:
            var_idx = var_idx.transpose(0, 1)
            res_idx = res_idx.transpose(0, 1)
            leadtime_idx = leadtime_idx.transpose(0, 1)

            var_embeds = self.var_embedding(var_idx)
            res_embeds = self.res_embedding(res_idx)
            leadtime_embeds = self.leadtime_embedding(leadtime_idx)

            token_embeds = token_embeds + var_embeds + res_embeds + leadtime_embeds

        # Add continuous spatial-temporal encoding
        if self.config.use_spatial_temporal_encoding and spatial_temporal_features is not None:
            spatial_temporal_features = spatial_temporal_features.transpose(0, 1)  # [seq_len, batch, 7]
            spatial_embeds = self.activation(self.spatial_temporal_proj(spatial_temporal_features))
            spatial_embeds = self.spatial_temporal_proj2(spatial_embeds)
            token_embeds = token_embeds + spatial_embeds

        return {"input_embeds": token_embeds}
```

#### 1.3 Create ClimLlamaModel and ClimLlamaForTraining
**File**: `src/nanotron/models/climllama.py`

```python
class ClimLlamaModel(Qwen2Model):
    """ClimLlama model with custom positional embeddings."""

    def __init__(self, config: ClimLlamaConfig, parallel_context: ParallelContext, ...):
        # Initialize with custom embedding
        super().__init__(config, parallel_context, ...)

        # Replace the embedding layer with ClimLlamaEmbedding
        self.token_position_embeddings = PipelineBlock(
            p2p=self.p2p,
            module_builder=ClimLlamaEmbedding,
            module_kwargs={
                "tp_pg": parallel_context.tp_pg,
                "config": config,
                "parallel_config": parallel_config,
            },
            module_input_keys={"input_ids", "input_mask", "var_idx", "res_idx", "leadtime_idx", "spatial_temporal_features"},
            module_output_keys={"input_embeds"},
        )

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        position_ids: Union[torch.Tensor, TensorPointer],
        var_idx: Union[torch.Tensor, TensorPointer],
        res_idx: Union[torch.Tensor, TensorPointer],
        leadtime_idx: Union[torch.Tensor, TensorPointer],
        spatial_temporal_features: Union[torch.Tensor, TensorPointer],
    ):
        # Forward pass with additional position information
        # ... (similar to Qwen2Model but with extra inputs)
        pass


class ClimLlamaForTraining(NanotronModel):
    """Training wrapper for ClimLlama."""

    def __init__(self, config: ClimLlamaConfig, ...):
        self.model = ClimLlamaModel(config, ...)
        self.loss = PipelineBlock(...)

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        position_ids: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
        var_idx: Union[torch.Tensor, TensorPointer],
        res_idx: Union[torch.Tensor, TensorPointer],
        leadtime_idx: Union[torch.Tensor, TensorPointer],
        spatial_temporal_features: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Training forward pass
        pass
```

---

### Phase 2: Dataset Implementation

#### 2.1 Create ClimLlamaDataset
**File**: `src/nanotron/data/climllama_dataset.py` (new file)

**Reference**: `GPTDataset` in [src/nanotron/data/nemo_dataset/__init__.py](../src/nanotron/data/nemo_dataset/__init__.py) (lines 359-579)

```python
class ClimLlamaDataset(GPTDataset):
    """Dataset for ClimLlama that returns climate-specific position information."""

    def __init__(
        self,
        cfg,
        tokenizer,
        name,
        data_prefix,
        documents,
        indexed_dataset,
        num_samples,
        seq_length,
        seed,
        parallel_context,
        drop_last=True,
    ):
        super().__init__(
            cfg, tokenizer, name, data_prefix, documents,
            indexed_dataset, num_samples, seq_length, seed,
            parallel_context, drop_last
        )

        # Extract grid size (resolutions) from dataset metadata
        # The dataset folder contains metadata.json with resolution information
        # in model_config["resolutions"] (e.g., [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 32])
        self.resolutions = self._load_resolutions_from_metadata(data_prefix)

        # Precompute lat/lon grids for each resolution level
        self._precompute_spatial_grids()

    def _load_resolutions_from_metadata(self, data_prefix):
        """Load resolution information from dataset metadata.json.

        The dataset folder contains metadata.json with structure:
        {
            "model_config": {
                "resolutions": [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 32],
                ...
            }
        }

        These resolution values should be expanded to (n_lat, n_lon) tuples:
        [(1, 2), (2, 4), (3, 6), ..., (32, 64)]

        Args:
            data_prefix: Path to the dataset (can be a prefix or full path)

        Returns:
            List of (n_lat, n_lon) tuples for each resolution level
        """
        import json
        from pathlib import Path

        # Find metadata.json in the dataset directory
        # data_prefix might be like "/path/to/dataset/train" or "/path/to/dataset"
        data_path = Path(data_prefix)
        if data_path.is_file():
            data_path = data_path.parent

        # Search for metadata.json
        metadata_path = data_path / "metadata.json"
        if not metadata_path.exists():
            # Try parent directory
            metadata_path = data_path.parent / "metadata.json"

        if not metadata_path.exists():
            # Fallback to default resolutions
            print(f"Warning: metadata.json not found at {data_path}, using default resolutions")
            return [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10), (6, 12), (8, 16), (10, 20), (13, 26), (16, 32), (32, 64)]

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Extract resolutions from model_config
        resolution_values = metadata.get("model_config", {}).get("resolutions", [])

        if not resolution_values:
            print(f"Warning: No resolutions found in metadata.json, using default")
            return [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10), (6, 12), (8, 16), (10, 20), (13, 26), (16, 32), (32, 64)]

        # Expand to (n_lat, n_lon) tuples where n_lon = 2 * n_lat
        resolutions = [(n, 2 * n) for n in resolution_values]

        print(f"Loaded {len(resolutions)} resolution levels from metadata: {resolutions}")
        return resolutions

    def _precompute_spatial_grids(self):
        """Precompute latitude and longitude grids for each resolution level.

        For climate data, we have a hierarchy of resolutions stored as (n_lat, n_lon) tuples.
        For example: [(1, 2), (2, 4), (3, 6), ..., (32, 64)]
        """
        self.spatial_grids = {}

        for res_idx, (n_lat, n_lon) in enumerate(self.resolutions):
            # Longitude: linspace(w, 2π - w, m) where w = 2π / (2m)
            w = 2 * np.pi / (2 * n_lon)
            lon_centers = np.linspace(w, 2 * np.pi - w, n_lon)

            # Latitude: linspace(π/2 - h, -π/2 + h, n) where h = π / (2n)
            h = np.pi / (2 * n_lat)
            lat_centers = np.linspace(np.pi / 2 - h, -np.pi / 2 + h, n_lat)

            # Create meshgrid
            lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

            # Precompute x, y, z coordinates
            # x = cos(lat) * sin(lon)
            # y = cos(lat) * cos(lon)
            # z = sin(lat)
            x_grid = np.cos(lat_grid) * np.sin(lon_grid)
            y_grid = np.cos(lat_grid) * np.cos(lon_grid)
            z_grid = np.sin(lat_grid)

            # Store grids for this resolution
            self.spatial_grids[res_idx] = {
                'lat_centers': lat_centers,
                'lon_centers': lon_centers,
                'x_grid': x_grid,
                'y_grid': y_grid,
                'z_grid': z_grid,
                'n_lat': n_lat,
                'n_lon': n_lon,
            }

    def _compute_temporal_features(self, timestamp):
        """Compute temporal features from timestamp.

        Args:
            timestamp: datetime or unix timestamp

        Returns:
            (cos_hour_of_day, sin_hour_of_day, cos_day_of_year, sin_day_of_year)
        """
        # Extract hour of day and day of year
        # Normalize to [0, 2π]
        hour_angle = (timestamp.hour + timestamp.minute / 60.0) / 24.0 * 2 * np.pi
        day_angle = timestamp.timetuple().tm_yday / 365.0 * 2 * np.pi

        return (
            np.cos(hour_angle),
            np.sin(hour_angle),
            np.cos(day_angle),
            np.sin(day_angle),
        )

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Return item with climate-specific position information."""

        # Get base item from GPTDataset
        base_item = super().__getitem__(idx)

        # TODO: Extract metadata from tokens or external source
        # For now, create placeholder arrays
        seq_len = len(base_item["input_ids"])

        # Discrete position indices
        var_idx = np.zeros(seq_len, dtype=np.int64)  # TODO: extract from metadata
        res_idx = np.zeros(seq_len, dtype=np.int64)  # TODO: extract from metadata
        leadtime_idx = np.zeros(seq_len, dtype=np.int64)  # TODO: extract from metadata

        # Spatial-temporal features [seq_len, 7]
        # TODO: Extract actual spatial positions and timestamps from metadata
        spatial_temporal_features = np.zeros((seq_len, 7), dtype=np.float32)
        # spatial_temporal_features[:, 0] = x (from grid position)
        # spatial_temporal_features[:, 1] = y
        # spatial_temporal_features[:, 2] = z
        # spatial_temporal_features[:, 3] = cos_hour_of_day
        # spatial_temporal_features[:, 4] = sin_hour_of_day
        # spatial_temporal_features[:, 5] = cos_day_of_year
        # spatial_temporal_features[:, 6] = sin_day_of_year

        return {
            "input_ids": base_item["input_ids"],
            "var_idx": var_idx,
            "res_idx": res_idx,
            "leadtime_idx": leadtime_idx,
            "spatial_temporal_features": spatial_temporal_features,
        }
```

**Key Features**:
- Inherits from `GPTDataset` for base functionality
- Loads resolution hierarchy from dataset's `metadata.json` file
- Precomputes spatial grids for all resolution levels
- Computes x, y, z from lat/lon: `x=cos(lat)sin(lon)`, `y=cos(lat)cos(lon)`, `z=sin(lat)`
- Computes temporal features with proper normalization to [0, 2π]
- Returns 7D spatial-temporal features + discrete indices

#### 2.2 Resolution Data Source
The resolution hierarchy is stored in the dataset's `metadata.json` file:
```json
{
  "model_config": {
    "resolutions": [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 32],
    ...
  }
}
```

These values are expanded to (n_lat, n_lon) tuples: `[(1, 2), (2, 4), (3, 6), ..., (32, 64)]`

This ensures that the spatial grids match exactly the resolution levels used during tokenizer training.

#### 2.3 Position Metadata Extraction Strategy
The dataset needs to extract position metadata for each token. Two approaches:

**Option A: Embed metadata in tokens**
- Store special tokens or patterns that encode position information
- Parse these during `__getitem__`

**Option B: External metadata file**
- Maintain a separate index mapping sample_idx → position metadata
- Load metadata alongside the indexed dataset

---

### Phase 3: Data Collator Implementation

#### 3.1 Create DataCollatorForClimLlama
**File**: `src/nanotron/data/climllama_collator.py` (new file)

**Reference**: `DataCollatorForCLMWithPositionIds` in [src/nanotron/data/clm_collator.py](../src/nanotron/data/clm_collator.py) (lines 138-282)

```python
@dataclasses.dataclass
class DataCollatorForClimLlama:
    """Collator for ClimLlama that handles climate-specific position information."""

    sequence_length: int
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext
    use_doc_masking: bool = True

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        """Collate examples into batch with all position information."""

        batch_size = len(examples)
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)

        # Return TensorPointers for non-participating ranks
        if current_pp_rank not in [self.input_pp_rank, self.output_pp_rank]:
            return {
                "input_ids": TensorPointer(...),
                "position_ids": TensorPointer(...),
                "var_idx": TensorPointer(...),
                "res_idx": TensorPointer(...),
                "leadtime_idx": TensorPointer(...),
                "spatial_temporal_features": TensorPointer(...),
                "label_ids": TensorPointer(...),
                "label_mask": TensorPointer(...),
            }

        result = {}

        # Process input side (first PP rank)
        if current_pp_rank == self.input_pp_rank:
            # Stack all fields
            input_ids = np.vstack([ex["input_ids"] for ex in examples])
            var_idx = np.vstack([ex["var_idx"] for ex in examples])
            res_idx = np.vstack([ex["res_idx"] for ex in examples])
            leadtime_idx = np.vstack([ex["leadtime_idx"] for ex in examples])
            spatial_temporal_features = np.stack([ex["spatial_temporal_features"] for ex in examples])

            # Create position_ids if available (for document masking)
            if "positions" in examples[0] and self.use_doc_masking:
                position_ids = np.vstack([ex["positions"] for ex in examples])
            else:
                # Default sequential positions
                position_ids = np.tile(np.arange(self.sequence_length + 1), (batch_size, 1))

            # Drop last token for input
            result["input_ids"] = torch.from_numpy(input_ids[:, :-1])
            result["position_ids"] = torch.from_numpy(position_ids[:, :-1])
            result["var_idx"] = torch.from_numpy(var_idx[:, :-1])
            result["res_idx"] = torch.from_numpy(res_idx[:, :-1])
            result["leadtime_idx"] = torch.from_numpy(leadtime_idx[:, :-1])
            result["spatial_temporal_features"] = torch.from_numpy(spatial_temporal_features[:, :-1, :])

        # Process output side (last PP rank)
        if current_pp_rank == self.output_pp_rank:
            # Create labels (shifted by 1)
            input_ids = np.vstack([ex["input_ids"] for ex in examples])
            label_ids = input_ids[:, 1:]

            # Create label mask using position_ids
            if "positions" in examples[0] and self.use_doc_masking:
                position_ids = np.vstack([ex["positions"] for ex in examples])
                shifted_positions = position_ids[:, 1:]

                # Mask where position resets to 0
                label_mask = np.ones_like(shifted_positions, dtype=bool)
                label_mask &= ~(shifted_positions == 0)
            else:
                label_mask = np.ones_like(label_ids, dtype=bool)

            result["label_ids"] = torch.from_numpy(label_ids)
            result["label_mask"] = torch.from_numpy(label_mask)

        return result
```

**Key Features**:
- Handles all climate-specific position fields
- Maintains compatibility with pipeline parallelism
- Supports document masking like `DataCollatorForCLMWithPositionIds`
- Properly shifts labels for next-token prediction

---

### Phase 4: Integration and Testing

#### 4.1 Register ClimLlamaConfig
**File**: `src/nanotron/config/models_config.py`

Add to model config registry:
```python
from nanotron.models.climllama import ClimLlamaForTraining

# In model loading logic
MODEL_CONFIG_MAP = {
    ...
    "ClimLlamaConfig": ClimLlamaForTraining,
}
```

#### 4.2 Update Training Script
**File**: Training configuration YAML

```yaml
model:
  model_config:
    # Inherit Qwen2 base config
    hidden_size: 4096
    num_hidden_layers: 32
    num_attention_heads: 32
    ...

    # ClimLlama-specific config
    use_absolute_position_embeddings: true
    var_vocab_size: 100
    res_vocab_size: 10
    leadtime_vocab_size: 240
    use_spatial_temporal_encoding: true
    spatial_temporal_encoding_dim: 128

# Note: Grid resolutions are loaded from the dataset's metadata.json,
# not from the model config. The dataset metadata contains the resolution
# hierarchy used during tokenizer training.
```

#### 4.3 Testing Checklist
- [ ] Test ClimLlamaEmbedding forward pass with all position types
- [ ] Test ClimLlamaDataset returns correct shapes
- [ ] Test DataCollatorForClimLlama batching
- [ ] Test end-to-end training with small model
- [ ] Verify RoPE still works in attention layers
- [ ] Check tensor parallelism compatibility
- [ ] Validate position encoding values (x, y, z in [-1, 1], temporal in [-1, 1])

---

## Implementation Checklist

### Core Components
- [ ] Create `ClimLlamaConfig` in `src/nanotron/config/models_config.py`
- [ ] Create `ClimLlamaEmbedding` in `src/nanotron/models/climllama.py`
- [ ] Create `ClimLlamaModel` in `src/nanotron/models/climllama.py`
- [ ] Create `ClimLlamaForTraining` in `src/nanotron/models/climllama.py`

### Data Pipeline
- [ ] Create `ClimLlamaDataset` in `src/nanotron/data/climllama_dataset.py`
  - [ ] Implement spatial grid precomputation
  - [ ] Implement temporal feature computation
  - [ ] Design metadata extraction strategy
- [ ] Create `DataCollatorForClimLlama` in `src/nanotron/data/climllama_collator.py`

### Integration
- [ ] Register ClimLlama in model config registry
- [ ] Create example training configuration YAML
- [ ] Update `prepare_training_config.py` to support ClimLlama

### Testing
- [ ] Unit tests for embedding layer
- [ ] Unit tests for dataset
- [ ] Unit tests for collator
- [ ] Integration test for full forward pass
- [ ] Verify hybrid PE (absolute + RoPE) working together

---

## Key Design Principles

1. **Inheritance Over Reimplementation**: Inherit from Qwen2 to reuse existing infrastructure
2. **Additive Design**: Add positional embeddings without breaking RoPE
3. **Flexibility**: Make all position encodings optional via config
4. **Efficiency**: Precompute spatial grids, use TensorParallel layers
5. **Compatibility**: Maintain compatibility with pipeline/tensor/data parallelism

---

## References

- **Qwen2 Model**: [src/nanotron/models/qwen.py](../src/nanotron/models/qwen.py)
- **GPTDataset**: [src/nanotron/data/nemo_dataset/__init__.py](../src/nanotron/data/nemo_dataset/__init__.py)
- **DataCollatorForCLMWithPositionIds**: [src/nanotron/data/clm_collator.py](../src/nanotron/data/clm_collator.py)
- **Position IDs Analysis**: [climllama/note_collator_position_id.md](note_collator_position_id.md)
- **Hybrid PE Guide**: [climllama/positional_embedding_guide.md](positional_embedding_guide.md)

---

## Notes

### Spatial Position Computation
For a grid of size `n × m` (latitude × longitude) spanning the globe:

**Longitude centers**: `lon = linspace(w, 2π - w, m)` where `w = 2π / (2m)`
- Ensures centers are offset from edges by half a cell width

**Latitude centers**: `lat = linspace(π/2 - h, -π/2 + h, n)` where `h = π / (2n)`
- π/2 = 90° (North pole), -π/2 = -90° (South pole)
- Centers offset from poles by half a cell height

**Cartesian coordinates**:
- `x = cos(lat) × sin(lon)`
- `y = cos(lat) × cos(lon)`
- `z = sin(lat)`

All values in range [-1, 1], providing normalized spatial encoding.

### Temporal Position Computation
**Hour of day**: `hour_angle = (hour + minute/60) / 24 × 2π`
- Normalized to [0, 2π]
- Use cos/sin for periodic encoding

**Day of year**: `day_angle = day_of_year / 365 × 2π`
- Normalized to [0, 2π]
- Use cos/sin for periodic encoding

---

## Next Steps

1. Start with Phase 1: Create configuration and model structure
2. Implement Phase 2: Dataset with placeholder metadata
3. Implement Phase 3: Data collator
4. Test each component incrementally
5. Design metadata strategy based on data format
6. Full integration testing