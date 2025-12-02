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

    # Always allocate one extra size for position embedding, index 0 for unknown
    # Embedding dimensions for different position types
    var_vocab_size: int = 13  # Number of pressure-level and surface-level atmosphere/climate variables
    variables: List[str] = ["unk", "z", "t", "q", "u", "v", "w", "t2m", "msl", "u10", "v10", "tp_1h", "tp_6h"]

    res_vocab_size: int = 12   # Number of resolution levels
    leadtime_vocab_size: int = 13  # Embed 12 possible lead times, e.g., 0h, 6h, ..., 72h
    leadtime_step: str = "6h"

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

#### 2.1 Position Generation Strategy

**Selected Approach: On-the-fly Position Generation using `atmtokenizer.eval.weaved_tokens_position`**

We use the `WeavedTokensPositionVisitor` from atmtokenizer to generate positional metadata on-the-fly during training. This ensures consistency between tokenizer and model while avoiding the need for precomputed position arrays.

**Reference**: `../atmtokenizer/atmtokenizer/eval/weaved_tokens_position.py`

The `WeavedTokensPositionVisitor` traverses the Lark parse tree and assigns each token:
- `leadtime`: Lead time ID (hours from timestamp_0)
- `resolution`: Resolution level ID
- `variable`: Variable ID
- `grid_i`, `grid_j`: Grid row/column indices (for CODEBOOK_TOKEN only)
- `grid_lat`, `grid_lon`: Latitude/longitude in degrees
- `grid_x`, `grid_y`, `grid_z`: Cartesian coordinates on unit sphere
- `timestamp`: Absolute timestamp (timestamp_0 + leadtime_hours)
- `cos_hour_of_day`, `sin_hour_of_day`: Temporal encoding for hour
- `cos_day_of_year`, `sin_day_of_year`: Temporal encoding for day

#### 2.2 ClimLlamaDataset

**File**: `src/nanotron/data/climllama_dataset.py` (new file)

**Reference**: `GPTDataset` in [src/nanotron/data/nemo_dataset/__init__.py](../src/nanotron/data/nemo_dataset/__init__.py)

```python
from atmtokenizer.eval.weaved_tokens_position import get_leaf_positions
from atmtokenizer.eval.special_tokens import create_special_tokens
from atmtokenizer.eval.lark_grammar import WEAVED_TOKENS_GRAMMAR
from lark import Lark
from datetime import datetime


class ClimLlamaDataset(GPTDataset):
    """Dataset for ClimLlama that generates position arrays on-the-fly."""

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
        codebook_size: int = 32768,
    ):
        super().__init__(
            cfg, tokenizer, name, data_prefix, documents,
            indexed_dataset, num_samples, seq_length, seed,
            parallel_context, drop_last
        )

        # Setup Lark parser and special tokens for position generation
        self.parser = Lark(WEAVED_TOKENS_GRAMMAR, parser='lalr')
        self.special_tokens = create_special_tokens(codebook_size=codebook_size)

        # Load resolution shapes from metadata
        self.resolution_shapes = self._load_resolution_shapes_from_metadata(data_prefix)

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

    def _load_resolution_shapes_from_metadata(self, data_prefix) -> Dict[int, Tuple[int, int]]:
        """Load resolution shapes from dataset metadata.json.

        The dataset folder contains metadata.json with structure:
        {
            "model_config": {
                "resolutions": [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 32],
                ...
            }
        }

        Returns:
            Dict mapping resolution ID to (height, width) tuple
            E.g., {0: (1, 2), 1: (2, 4), 2: (3, 6), ...}
        """
        import json
        from pathlib import Path

        data_path = Path(data_prefix)
        if data_path.is_file():
            data_path = data_path.parent

        metadata_path = data_path / "metadata.json"
        if not metadata_path.exists():
            metadata_path = data_path.parent / "metadata.json"

        if not metadata_path.exists():
            print(f"Warning: metadata.json not found, using default resolutions")
            default_res = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 32]
            return {i: (n, 2 * n) for i, n in enumerate(default_res)}

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        resolution_values = metadata.get("model_config", {}).get("resolutions", [])
        if not resolution_values:
            default_res = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 32]
            return {i: (n, 2 * n) for i, n in enumerate(default_res)}

        # Map resolution ID to (height, width) tuple
        return {i: (n, 2 * n) for i, n in enumerate(resolution_values)}

    def _generate_positions(self, input_ids: np.ndarray, timestamp_0: datetime) -> Dict[str, np.ndarray]:
        """Generate position arrays on-the-fly using WeavedTokensPositionVisitor.

        Args:
            input_ids: Token sequence from indexed dataset
            timestamp_0: Initial timestamp for the document

        Returns:
            Dict with var_idx, res_idx, leadtime_idx, spatial_temporal_features
        """
        # Parse token sequence
        tree = self.parser.parse(input_ids.tolist())

        # Get positions for all tokens
        leaves = get_leaf_positions(
            tree,
            self.resolution_shapes,
            self.special_tokens,
            timestamp_0
        )

        # Convert to arrays
        n_tokens = len(leaves)
        var_idx = np.zeros(n_tokens, dtype=np.int64)
        res_idx = np.zeros(n_tokens, dtype=np.int64)
        leadtime_idx = np.zeros(n_tokens, dtype=np.int64)
        spatial_temporal_features = np.zeros((n_tokens, 7), dtype=np.float32)

        for i, (token, pos) in enumerate(leaves):
            var_idx[i] = pos["variable"] or 0
            res_idx[i] = pos["resolution"] or 0
            leadtime_idx[i] = pos["leadtime"] or 0

            # Spatial features (x, y, z)
            spatial_temporal_features[i, 0] = pos["grid_x"] or 0.0
            spatial_temporal_features[i, 1] = pos["grid_y"] or 0.0
            spatial_temporal_features[i, 2] = pos["grid_z"] or 0.0

            # Temporal features (cos_hour, sin_hour, cos_day, sin_day)
            spatial_temporal_features[i, 3] = pos["cos_hour_of_day"]
            spatial_temporal_features[i, 4] = pos["sin_hour_of_day"]
            spatial_temporal_features[i, 5] = pos["cos_day_of_year"]
            spatial_temporal_features[i, 6] = pos["sin_day_of_year"]

        return {
            "var_idx": var_idx,
            "res_idx": res_idx,
            "leadtime_idx": leadtime_idx,
            "spatial_temporal_features": spatial_temporal_features,
        }

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Return item with position information generated on-the-fly."""

        # Get base item from GPTDataset
        base_item = super().__getitem__(idx)

        # Get document index and timestamp
        doc_idx = self._get_document_index_for_sample(idx)
        timestamp_0 = self._get_document_timestamp(doc_idx)

        # Generate positions on-the-fly
        positions = self._generate_positions(base_item["input_ids"], timestamp_0)

        return {
            "input_ids": base_item["input_ids"],
            "var_idx": positions["var_idx"],
            "res_idx": positions["res_idx"],
            "leadtime_idx": positions["leadtime_idx"],
            "spatial_temporal_features": positions["spatial_temporal_features"],
        }

    def _get_document_index_for_sample(self, sample_idx: int) -> int:
        """Get document index for a given sample index."""
        # Implementation depends on sampling strategy
        # For simple 1:1 mapping:
        return sample_idx

    def _get_document_timestamp(self, doc_idx: int) -> datetime:
        """Get timestamp for a document.

        Args:
            doc_idx: Document index in the indexed dataset

        Returns:
            datetime object for the document's initial timestamp
        """
        if self.timestamps is not None and doc_idx < len(self.timestamps):
            # Convert numpy datetime64 or unix timestamp to datetime
            ts = self.timestamps[doc_idx]
            if isinstance(ts, np.datetime64):
                return ts.astype('datetime64[s]').astype(datetime)
            else:
                return datetime.fromtimestamp(ts)
        else:
            # Fallback: return epoch
            return datetime(1970, 1, 1, 0, 0, 0)
```

**Key Features**:
- Uses `WeavedTokensPositionVisitor` from atmtokenizer for on-the-fly position generation
- No precomputation required - positions generated during `__getitem__`
- Consistent with tokenizer's Lark grammar and special token definitions
- Single source of truth in `weaved_tokens_position.py`

**Advantages of On-the-fly Generation**:
- No additional preprocessing step required
- No extra storage for position arrays
- Simpler data pipeline
- Positions always in sync with token sequences

**Trade-offs**:
- Slightly higher CPU overhead during training (Lark parsing per sample)
- Can be mitigated with DataLoader workers

#### 2.3 Position Metadata Fields

The position arrays contain the following fields per token:

| Field | Type | Description |
|-------|------|-------------|
| var_idx | int | Variable index (0=unknown, 1=z, 2=t, ...) |
| res_idx | int | Resolution level (0=coarsest, N=finest) |
| leadtime_idx | int | Lead time in hours from timestamp_0 |
| spatial_temporal_features[:, 0] | float | grid_x = cos(lat)sin(lon), range [-1, 1] |
| spatial_temporal_features[:, 1] | float | grid_y = cos(lat)cos(lon), range [-1, 1] |
| spatial_temporal_features[:, 2] | float | grid_z = sin(lat), range [-1, 1] |
| spatial_temporal_features[:, 3] | float | cos_hour_of_day, range [-1, 1] |
| spatial_temporal_features[:, 4] | float | sin_hour_of_day, range [-1, 1] |
| spatial_temporal_features[:, 5] | float | cos_day_of_year, range [-1, 1] |
| spatial_temporal_features[:, 6] | float | sin_day_of_year, range [-1, 1] |

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
- [x] Test ClimLlamaEmbedding forward pass with all position types
- [ ] Test ClimLlamaDataset returns correct shapes
- [ ] Test DataCollatorForClimLlama batching
- [x] Test end-to-end training with small model
- [x] Verify RoPE still works in attention layers
- [x] Check tensor parallelism compatibility
- [ ] Validate position encoding values (x, y, z in [-1, 1], temporal in [-1, 1])

---

## Implementation Checklist

### Core Components
- [x] Create `ClimLlamaConfig` in `src/nanotron/config/models_config.py`
- [x] Create `ClimLlamaEmbedding` in `src/nanotron/models/climllama.py`
- [x] Create `ClimLlamaModel` in `src/nanotron/models/climllama.py`
- [x] Create `ClimLlamaForTraining` in `src/nanotron/models/climllama.py`

### Data Pipeline
- [x] Create `ClimLlamaDataset` in `src/nanotron/data/climllama_dataset.py`
  - [x] Implement `_load_resolutions_from_metadata()` to load resolution hierarchy
  - [x] Implement on-the-fly position generation using `atmtokenizer.eval.weaved_tokens_position`
  - [x] Implement timestamp loading from `{data_prefix}_timestamps.npy`
  - [x] Implement `_get_document_timestamp()` for timestamp lookup
- [x] Create `DataCollatorForClimLlama` in `src/nanotron/data/climllama_collator.py`

### Integration
- [x] Register ClimLlama in model config registry (in `src/nanotron/trainer.py`)
- [x] Create example training configuration YAML
- [x] Update `prepare_training_config.py` to support ClimLlama

### Testing
- [x] Unit tests for embedding layer
- [x] Unit tests for dataset
- [x] Unit tests for collator
- [x] Integration test for full forward pass
- [x] Verify hybrid PE (absolute + RoPE) working together

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