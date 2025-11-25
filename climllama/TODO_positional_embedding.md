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

        # Load global timestamp array
        # See TODO_weavedtokens_timestamp.md for specification
        timestamp_path = f"{data_prefix}_timestamps.npy"
        if os.path.exists(timestamp_path):
            self.timestamps = np.load(timestamp_path)
            print(f"Loaded {len(self.timestamps)} timestamps from {timestamp_path}")
        else:
            print(f"Warning: No timestamp file found at {timestamp_path}")
            print(f"Temporal features will use fallback epoch time")
            self.timestamps = None

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

    def _parse_special_tokens(self, input_ids: np.ndarray) -> tuple:
        """Parse special tokens to extract position metadata.

        Args:
            input_ids: Token sequence with special tokens from TokenWeaver

        Returns:
            Tuple of (var_idx, res_idx, spatial_features):
                - var_idx: (seq_len,) int64 array of variable indices
                - res_idx: (seq_len,) int64 array of resolution level indices
                - spatial_features: (seq_len, 3) float32 array of (x, y, z) coordinates
        """
        seq_len = len(input_ids)
        var_idx = np.zeros(seq_len, dtype=np.int64)
        res_idx = np.zeros(seq_len, dtype=np.int64)
        spatial_features = np.zeros((seq_len, 3), dtype=np.float32)

        # Special token IDs (must match TokenWeaver config)
        VAR_TOKEN_START = 32800
        RES_TOKEN_START = 32787
        EOR_TOKEN = 32785

        current_var = 0
        current_res = 0
        token_position = 0  # Position within current resolution level
        output_idx = 0

        for tok in input_ids:
            if tok >= VAR_TOKEN_START and tok < VAR_TOKEN_START + 256:
                # Variable marker: extract variable index
                current_var = tok - VAR_TOKEN_START
                # Don't increment output_idx; special tokens not in output

            elif tok >= RES_TOKEN_START and tok < RES_TOKEN_START + 13:
                # Resolution marker: extract resolution level
                current_res = tok - RES_TOKEN_START
                token_position = 0  # Reset position counter for new resolution
                # Don't increment output_idx

            elif tok == EOR_TOKEN:
                # End of resolution marker
                # Don't increment output_idx
                pass

            else:
                # Regular codebook token: assign metadata
                if output_idx < seq_len:
                    var_idx[output_idx] = current_var
                    res_idx[output_idx] = current_res

                    # Compute spatial coordinates from token position
                    n_lat, n_lon = self.resolutions[current_res]
                    lat_idx = token_position // n_lon
                    lon_idx = token_position % n_lon

                    # Lookup spatial coordinates from precomputed grids
                    grid = self.spatial_grids[current_res]
                    spatial_features[output_idx, 0] = grid['x_grid'][lat_idx, lon_idx]
                    spatial_features[output_idx, 1] = grid['y_grid'][lat_idx, lon_idx]
                    spatial_features[output_idx, 2] = grid['z_grid'][lat_idx, lon_idx]

                    token_position += 1
                    output_idx += 1

        # If special tokens are stripped before this point, return full arrays
        # Otherwise, return only filled portion
        return var_idx[:output_idx], res_idx[:output_idx], spatial_features[:output_idx]

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Return item with climate-specific position information."""

        # Get base item from GPTDataset
        base_item = super().__getitem__(idx)

        # Note: input_ids may contain special tokens from TokenWeaver
        # Or special tokens may already be stripped by GPTDataset
        # This implementation assumes special tokens are present

        # Parse special tokens to extract position metadata
        var_idx, res_idx, spatial_features = self._parse_special_tokens(
            base_item["input_ids"]
        )

        seq_len = len(var_idx)  # Length after special token filtering

        # Get document index and timestamp
        # (Simplified: assumes one document per sample)
        doc_idx = self._get_document_index_for_sample(idx)
        timestamp = self._get_document_timestamp(doc_idx)

        # Compute temporal features (same for all tokens)
        temporal_features = self._compute_temporal_features(timestamp)

        # Combine spatial and temporal features
        spatial_temporal_features = np.zeros((seq_len, 7), dtype=np.float32)
        spatial_temporal_features[:, 0:3] = spatial_features  # x, y, z
        spatial_temporal_features[:, 3] = temporal_features[0]  # cos_hour_of_day
        spatial_temporal_features[:, 4] = temporal_features[1]  # sin_hour_of_day
        spatial_temporal_features[:, 5] = temporal_features[2]  # cos_day_of_year
        spatial_temporal_features[:, 6] = temporal_features[3]  # sin_day_of_year

        # Lead time: for now, assume analysis data (lead time = 0)
        # TODO: Extract lead time from metadata if using forecast data
        leadtime_idx = np.zeros(seq_len, dtype=np.int64)

        return {
            "input_ids": base_item["input_ids"],
            "var_idx": var_idx,
            "res_idx": res_idx,
            "leadtime_idx": leadtime_idx,
            "spatial_temporal_features": spatial_temporal_features,
        }

    def _get_document_index_for_sample(self, sample_idx: int) -> int:
        """Get document index for a given sample index.

        This is a simplified implementation assuming one sample = one document.
        For more complex sampling strategies (e.g., samples spanning multiple documents),
        this needs to be more sophisticated.
        """
        # TODO: Implement proper document index tracking
        # For now, assume 1:1 mapping
        return sample_idx

    def _get_document_timestamp(self, doc_idx: int) -> np.datetime64:
        """Get timestamp for a document.

        Args:
            doc_idx: Document index in the indexed dataset

        Returns:
            Timestamp as numpy datetime64[s]
        """
        if self.timestamps is not None and doc_idx < len(self.timestamps):
            return self.timestamps[doc_idx]
        else:
            # Fallback: return epoch
            return np.datetime64('1970-01-01T00:00:00')

    def _compute_temporal_features(self, timestamp: np.datetime64) -> np.ndarray:
        """Compute temporal features from timestamp.

        Args:
            timestamp: numpy datetime64[s] object

        Returns:
            np.ndarray: (4,) array with [cos_hour, sin_hour, cos_day, sin_day]
        """
        import datetime

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

**Key Features**:
- Inherits from `GPTDataset` for base functionality
- Loads resolution hierarchy from dataset's `metadata.json` file
- Loads global timestamp array from `{data_prefix}_timestamps.npy`
- Precomputes spatial grids for all resolution levels
- Parses special tokens from TokenWeaver to extract var_idx, res_idx
- Computes spatial positions (x, y, z) from resolution grids
- Computes temporal features from timestamps with proper normalization to [0, 2π]
- Returns 7D spatial-temporal features + discrete indices

**Important Notes**:
- Special token IDs (32768+) must be coordinated with TokenWeaver configuration
- See [TODO_weavedtokens_timestamp.md](TODO_weavedtokens_timestamp.md) for timestamp format
- Input tokens may contain special tokens; parser filters them out
- Spatial coordinates computed on-the-fly from flattened token positions

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

**Selected Approach: Option C - Hybrid (Special Token Parsing + Spatial Grid Lookup)**

The dataset extracts position metadata using a combination of:

1. **Special Token Parsing**: Parse weaved token sequences from TokenWeaver to extract discrete indices
   - Variable index (`var_idx`): Extracted from `<VAR:...>` special tokens
   - Resolution index (`res_idx`): Extracted from `<RES_N>` special tokens
   - Spatial position within resolution level: Track token position between `<RES_N>` and `<EOR>` markers

2. **Spatial Grid Precomputation**: Convert flattened positions to (lat, lon) coordinates
   - Precompute lat/lon grids for each resolution level (done in `_precompute_spatial_grids()`)
   - Map flattened token position to 2D grid coordinates
   - Compute (x, y, z) from (lat, lon) using spherical projection

3. **Global Timestamp Array**: Load timestamps from separate file
   - See [TODO_weavedtokens_timestamp.md](TODO_weavedtokens_timestamp.md) for detailed specification
   - One timestamp per document in indexed dataset
   - Compute temporal features (cos_hour, sin_hour, cos_day, sin_day) from timestamp

**Advantages**:
- Minimal changes to TokenWeaver (only needs special tokens, no extra metadata)
- Spatial information computed on-the-fly (no storage overhead)
- Temporal information efficiently stored in separate array
- Clear separation of concerns between tokenizer and model

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
  - [ ] Implement `_load_resolutions_from_metadata()` to load resolution hierarchy
  - [ ] Implement `_precompute_spatial_grids()` for all resolution levels
  - [ ] Implement `_parse_special_tokens()` to extract var_idx, res_idx, spatial positions
  - [ ] Implement timestamp loading from `{data_prefix}_timestamps.npy`
  - [ ] Implement `_compute_temporal_features()` for temporal encoding
  - [ ] Implement `_get_document_timestamp()` for timestamp lookup
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

## Coordination with TokenWeaver

### Data Format Contract

ClimLlama depends on specific data formats from the TokenWeaver pipeline:

#### 1. **Indexed Dataset Files**
Generated by `scripts/weave_and_index.py` in atmtokenizer:
- `{prefix}.bin`: Binary file with token sequences
- `{prefix}.idx`: Index file with document pointers
- Format: Standard Megatron indexed dataset format

#### 2. **Timestamp Array**
Generated alongside indexed dataset:
- `{prefix}_timestamps.npy`: Global timestamp array
- Shape: `(N_docs,)` where N_docs = number of documents
- Type: `datetime64[s]` or `int64` (Unix timestamps)
- See [TODO_weavedtokens_timestamp.md](TODO_weavedtokens_timestamp.md) for specification

#### 3. **Metadata File**
JSON file with tokenizer and dataset metadata:
- `{prefix}_metadata.json`: Dataset metadata
- Must include:
  ```json
  {
    "model_config": {
      "resolutions": [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 32],
      "vocab_size": 32768
    },
    "special_tokens": {
      "VAR_TOKEN_START": 32800,
      "RES_TOKEN_START": 32787,
      "EOR_TOKEN": 32785,
      "PAD_TOKEN": 32784
    }
  }
  ```

#### 4. **Token Sequence Format**
Weaved 1D sequences with special tokens (from TokenWeaver):
```
<VAR:temp:850> <RES_0> tok_0 tok_1 <EOR> <RES_1> tok_0 tok_1 ... <EOR> ...
```

Special token ID allocation:
- `0-32767`: Codebook tokens (from VQ-VAR)
- `32768-32783`: Reserved
- `32784`: `<PAD>`
- `32785`: `<EOR>` (End of Resolution)
- `32786`: `<VAR>` (Variable marker)
- `32787-32799`: `<RES_0>` to `<RES_12>` (13 resolution levels)
- `32800-33055`: Variable name tokens (256 slots)
- `33056-33311`: Pressure level tokens (256 slots)

### Vocabulary Coordination

**ClimLlamaConfig.vocab_size** must account for special tokens:

```python
@dataclass
class ClimLlamaConfig(Qwen2Config):
    # Vocabulary size INCLUDES special tokens
    vocab_size: int = 33312  # 32768 (codebook) + special tokens

    # This should match TokenWeaver's total vocabulary
    # = codebook_size + num_special_tokens
```

**TokenWeaver tokenizer config** must match:
```json
{
  "vocab_size": 33312,
  "codebook_size": 32768,
  "special_tokens": {
    "pad_token_id": 32784,
    "eor_token_id": 32785,
    ...
  }
}
```

### Implementation Dependencies

1. **TokenWeaver must implement** (in atmtokenizer):
   - `TokenWeaver.get_metadata(idx)` returning timestamp
   - `scripts/weave_and_index.py` saving `_timestamps.npy`
   - Metadata JSON with resolution and special token info

2. **ClimLlamaDataset must implement** (in nanotron):
   - Load timestamps from `_timestamps.npy`
   - Parse special tokens matching TokenWeaver's ID allocation
   - Load resolutions from `_metadata.json`

3. **Validation** (both projects):
   - Unit test: TokenWeaver output → ClimLlamaDataset input
   - Integration test: End-to-end from NPZ → indexed dataset → training batch
   - Verify special token IDs match between tokenizer config and dataset parser

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

1. **Phase 1 - Model Architecture** (nanotron):
   - Create `ClimLlamaConfig` extending `Qwen2Config`
   - Create `ClimLlamaEmbedding` with discrete + continuous position encodings
   - Create `ClimLlamaModel` and `ClimLlamaForTraining`

2. **Phase 2 - Data Pipeline** (nanotron):
   - Create `ClimLlamaDataset` with:
     - Resolution loading from metadata
     - Spatial grid precomputation
     - Special token parsing
     - Timestamp loading and temporal feature computation
   - Create `DataCollatorForClimLlama`

3. **Phase 3 - TokenWeaver Integration** (atmtokenizer):
   - Implement timestamp tracking in `scripts/weave_and_index.py`
   - Save `_timestamps.npy` alongside indexed dataset
   - Update metadata JSON with special token info
   - See [TODO_weavedtokens_timestamp.md](TODO_weavedtokens_timestamp.md)

4. **Phase 4 - Testing**:
   - Unit tests for each component
   - Integration test: TokenWeaver output → ClimLlamaDataset
   - End-to-end training test with small model
   - Validation: special token parsing, position encoding values

5. **Phase 5 - Documentation**:
   - Document data format contract
   - Create example training configuration
   - Update CLAUDE.md in both repositories