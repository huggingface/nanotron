# Hybrid Positional Embedding: Learned + RoPE

This guide explains how to implement a hybrid positional embedding approach that combines **learned absolute positional embeddings** with **RoPE (Rotary Position Embeddings)** in the nanotron LLaMA model.

## Why Hybrid Approach?

Using both types of positional encodings provides complementary benefits:
- **Learned Positional Embeddings**: Provide absolute position information (helps with position-aware tasks)
- **RoPE**: Provides relative position information (helps with length extrapolation and attention computation)

## Current Architecture

Currently, the LLaMA model uses **only RoPE**:
- Token embeddings are in [Embedding class](../src/nanotron/models/llama.py#L785-L812)
- RoPE is applied in the [attention layers](../src/nanotron/models/llama.py#L470-L518)
- **No separate positional embeddings** are used

## Implementation Steps

### Step 1: Update LlamaConfig

Add configuration parameters for learned positional embeddings.

**File**: `src/nanotron/config/models_config.py`

```python
@dataclass
class LlamaConfig(NanotronConfigs):
    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    max_position_embeddings: int = 4096  # ADD THIS - Maximum sequence length
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: Optional[int] = None
    pad_token_id: Optional[int] = None
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[dict] = None
    rope_theta: float = 10000.0
    rope_interleaved: bool = False
    tie_word_embeddings: bool = False
    use_cache: bool = True
    vocab_size: int = 32000
    use_learned_position_embeddings: bool = True  # ADD THIS - Enable hybrid approach
```

### Step 2: Modify Embedding Class

Update the Embedding class to include learned positional embeddings alongside token embeddings.

**File**: `src/nanotron/models/llama.py` (lines 785-812)

```python
class Embedding(nn.Module, AttachableStore):
    def __init__(self, tp_pg: dist.ProcessGroup, config: LlamaConfig, parallel_config: Optional[ParallelismArgs]):
        super().__init__()

        # Token embeddings (existing)
        self.token_embedding = TensorParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            pg=tp_pg,
            mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
        )

        # ADD: Learned positional embeddings (NEW)
        if config.use_learned_position_embeddings:
            self.position_embedding = TensorParallelEmbedding(
                num_embeddings=config.max_position_embeddings,
                embedding_dim=config.hidden_size,
                pg=tp_pg,
                mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
            )
        else:
            self.position_embedding = None

        self.pg = tp_pg
        self.config = config

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor):
        store = self.get_local_store()
        batch_size, seq_length = input_ids.shape

        # Handle position tracking for inference
        if store is not None:
            if "past_length" in store:
                past_length = store["past_length"]
            else:
                past_length = torch.zeros(1, dtype=torch.long, device=input_ids.device).expand(batch_size)

            cumsum_mask = input_mask.cumsum(-1, dtype=torch.long)
            store["past_length"] = past_length + cumsum_mask[:, -1]

            # Position IDs for inference (with KV cache)
            if self.position_embedding is not None:
                position_ids = past_length[:, None] + torch.arange(seq_length, device=input_ids.device)
        else:
            # Position IDs for training
            if self.position_embedding is not None:
                position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Format input in [seq_length, batch_size]
        input_ids = input_ids.transpose(0, 1)
        token_embeds = self.token_embedding(input_ids)

        # ADD: Apply positional embeddings if enabled
        if self.position_embedding is not None:
            position_ids = position_ids.transpose(0, 1)
            position_embeds = self.position_embedding(position_ids)
            # Combine token + learned position embeddings
            input_embeds = token_embeds + position_embeds
        else:
            input_embeds = token_embeds

        return {"input_embeds": input_embeds}
```

### Step 3: RoPE Remains in Attention Layers

**No changes needed** - RoPE is automatically applied in the attention layers.

**File**: `src/nanotron/models/llama.py` (lines 499-504)

```python
# RoPE is still applied in attention layers (keeps relative position info)
# In CoreAttention._forward_inference()
if self.rope_interleaved:
    query_states = self.rotary_embedding(query_states, position_ids=position_ids)
    key_states = self.rotary_embedding(key_states, position_ids=position_ids)
else:
    cos, sin = self.rotary_embedding(value_states, position_ids)
    query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

### Step 4: Update Your Training Config

**File**: Your training config YAML (e.g., `config.yaml`)

```yaml
model:
  model_config:
    # ... existing config ...
    max_position_embeddings: 4096  # Set to your max sequence length
    use_learned_position_embeddings: true  # Enable hybrid approach

    # RoPE config (existing)
    rope_theta: 10000.0
    rope_interleaved: false
```

## How It Works

```
Input Token IDs
      ↓
Token Embeddings ──┐
                   ├─→ Add ─→ Combined Embeddings ─→ Transformer Layers
Learned Pos Emb ───┘                                          ↓
                                                    Attention Layer (Q, K, V)
                                                              ↓
                                                    Apply RoPE to Q and K
                                                              ↓
                                                    Attention Computation
```

1. **Input Level**: Token embeddings + Learned positional embeddings are added
2. **Attention Level**: RoPE is applied to Query and Key tensors
3. **Result**: Model has both absolute position info (learned) and relative position info (RoPE)

## Key Files to Modify

| File | Lines | What to Change |
|------|-------|----------------|
| `src/nanotron/config/models_config.py` | LlamaConfig class | Add `max_position_embeddings` and `use_learned_position_embeddings` |
| `src/nanotron/models/llama.py` | 785-812 | Modify `Embedding` class to include positional embeddings |
| Your training config YAML | model.model_config | Set the new config parameters |

## Important Considerations

### 1. Tensor Parallelism
Use `TensorParallelEmbedding` to ensure the positional embeddings work correctly with distributed training across multiple GPUs.

### 2. Inference with KV Cache
The implementation properly tracks `past_length` to compute correct position IDs when using KV caching during inference.

### 3. Context Length
Ensure `max_position_embeddings >= max_sequence_length` in your config. The model cannot handle sequences longer than `max_position_embeddings`.

### 4. Initialization
Learned positional embeddings are initialized with small random values (default PyTorch behavior). This allows the model to learn optimal positional representations during training.

### 5. Memory Overhead
Adding learned positional embeddings increases model parameters by:
```
Parameters added = max_position_embeddings × hidden_size
Example: 4096 × 4096 = 16,777,216 parameters (~64MB in fp32)
```

### 6. RoPE Still Applied
RoPE is applied separately in the attention layers, so both types of positional encoding coexist and provide complementary information.

## Testing Your Implementation

After implementation, add these checks to verify correctness:

```python
# In the Embedding.forward() method, add assertions:
if self.position_embedding is not None:
    # Check embedding shapes match
    assert position_embeds.shape == token_embeds.shape, \
        f"Shape mismatch: position {position_embeds.shape} vs token {token_embeds.shape}"

    # Check final output shape
    assert input_embeds.shape == (seq_length, batch_size, self.config.hidden_size), \
        f"Invalid output shape: {input_embeds.shape}"

    # Verify position IDs are within bounds
    assert position_ids.max() < self.config.max_position_embeddings, \
        f"Position ID {position_ids.max()} exceeds max {self.config.max_position_embeddings}"
```

## RoPE Configuration Reference

For reference, here are the RoPE-related configuration options that work alongside learned embeddings:

```python
# In src/nanotron/config/models_config.py
rope_scaling: Optional[dict] = None  # For context length extension
rope_theta: float = 10000.0  # Base for frequency computation
rope_interleaved: bool = False  # LLaMA3 uses False
no_rope_layer: Optional[int] = None  # Skip RoPE every N layers (advanced)
rope_seq_len_interpolation_factor: Optional[float] = None  # For sequence length interpolation
```

### Advanced RoPE Features

1. **Sequence Length Interpolation**: Extend context beyond training length
   - Set `rope_seq_len_interpolation_factor` parameter
   - Implements technique from arXiv:2306.15595

2. **No-RoPE Layers**: Skip RoPE in specific layers (research feature)
   - Set `no_rope_layer` to skip every N layers
   - References: arXiv:2501.18795, arXiv:2305.19466

3. **Interleaved vs Non-Interleaved**:
   - **Non-interleaved** (default): More efficient, used by LLaMA3
   - **Interleaved**: Original implementation

## Benefits of Hybrid Approach

1. **Better Position Awareness**: Absolute positions help with tasks requiring exact position information
2. **Improved Extrapolation**: RoPE provides better length extrapolation capabilities
3. **Rich Positional Information**: Combines learned and geometric positional encodings
4. **Backward Compatible**: Can be toggled on/off via config without code changes

## Example Usage

```bash
# Train with hybrid embeddings
python train.py --config config.yaml

# In config.yaml, ensure:
# model.model_config.use_learned_position_embeddings: true
# model.model_config.max_position_embeddings: 4096
```

## Summary

After implementing these changes, your model will use:
- **Learned positional embeddings** added at the input level (absolute position info)
- **RoPE** applied in attention layers (relative position info)

This hybrid approach provides the best of both worlds for positional encoding!
