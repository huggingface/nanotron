# How `position_ids` are Consumed with `DataCollatorForCLMWithPositionIds`

Based on analysis of the nanotron codebase, here's an important finding: **The current implementation does NOT consume `position_ids` during training when using `DataCollatorForCLMWithPositionIds`**.

Here's what actually happens:

## 1. Data Collation

Location: [src/nanotron/data/clm_collator.py:138-282](src/nanotron/data/clm_collator.py#L138-L282)

The `DataCollatorForCLMWithPositionIds` processes the data as follows:

```python
# Input from dataset
{"input_ids": np.ndarray, "positions": np.ndarray}

# For input_pp_rank (lines 206-223)
if "positions" in examples[0] and self.use_doc_masking:
    position_ids = np.vstack([examples[i]["positions"] for i in range(len(examples))])
    result["positions"] = position_ids[:, :-1]  # Drop last position
    # Later renamed to position_ids
    result["position_ids"] = result.pop("positions")
```

**Key outputs**:
- `input_ids`: `[batch, seq_len]` - tokens 0 to seq_len-1
- `position_ids`: `[batch, seq_len]` - custom position IDs from dataset
- `label_ids`: `[batch, seq_len]` - tokens 1 to seq_len
- `label_mask`: `[batch, seq_len]` - **Uses position_ids to create mask** (masks tokens where position resets to 0)

## 2. Label Masking with Position IDs

Location: [src/nanotron/data/clm_collator.py:230-241](src/nanotron/data/clm_collator.py#L230-L241)

This is where `position_ids` IS actively used:

```python
# Create mask: True for all tokens except where position_id == 0
result["label_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

# Find where position_ids is 0 (document boundaries)
zeros = position_ids == 0
# Mask those tokens (don't compute loss on document start tokens)
result["label_mask"] &= ~zeros
```

**Purpose**: When packing multiple documents in one sequence, `position_ids` resets to 0 at each document boundary. The label mask prevents computing loss on the first token of each new document.

### Example from code comments (lines 166-181):

```python
# input_ids[0,:20]
# array([  198,    50,    30, 12532,  3589,   198,    51,    30, 30618,
#         198,    52,    30,  8279, 11274,   198, 21350,    42,   340,
#         0,  1780])

# position_ids[0,:20]
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18,  0])
#                  ^ position resets here (new document)

# result["label_ids"][0,:20]
# array([   50,    30, 12532,  3589,   198,    51,    30, 30618,   198,
#         52,    30,  8279, 11274,   198, 21350,    42,   340,     0,
#         1780,   314])

# result["label_mask"][0,:20]
# array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
#         True,  True,  True,  True,  True,  True,  True,  True,  True,
#     False,  True])
#         ^ label for position_id=0 is masked (token 1780)
```

## 3. Model Forward Pass - The Problem

Location: [src/nanotron/models/llama.py:1060-1066](src/nanotron/models/llama.py#L1060-L1066)

The `LlamaForTraining.forward()` method signature:

```python
def forward(
    self,
    input_ids: Union[torch.Tensor, TensorPointer],
    input_mask: Union[torch.Tensor, TensorPointer],
    label_ids: Union[torch.Tensor, TensorPointer],
    label_mask: Union[torch.Tensor, TensorPointer],
) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
```

**Notice**: There's **NO `position_ids` parameter** in the forward signature! This means the `position_ids` from the collator are being **silently ignored**.

## 4. Rotary Position Embeddings in Training

Location: [src/nanotron/models/llama.py:675-683](src/nanotron/models/llama.py#L675-L683)

In the `_forward_training` method:

```python
# Uses flash_attn's RotaryEmbedding which does NOT accept position_ids
query_states, key_value_states = self.flash_rotary_embedding(query_states, kv=key_value_states)
```

The `flash_rotary_embedding` automatically computes positions as `[0, 1, 2, ..., seq_len-1]` for each sequence, **ignoring custom position_ids**.

## 5. Inference vs Training Difference

### Inference

Location: [src/nanotron/models/llama.py:489-504](src/nanotron/models/llama.py#L489-L504)

```python
if "position_offsets" in store:
    position_ids = old_position_offsets[:, None] + sequence_mask
else:
    position_ids = torch.cumsum(sequence_mask, dim=-1, dtype=torch.int32) - 1

# Uses position_ids for rotary embeddings
query_states = self.rotary_embedding(query_states, position_ids=position_ids)
```

### Training

Location: [src/nanotron/models/llama.py:683](src/nanotron/models/llama.py#L683)

```python
# Does NOT use position_ids - uses implicit sequential positions
query_states, key_value_states = self.flash_rotary_embedding(query_states, kv=key_value_states)
```

## 6. Flash Attention's RotaryEmbedding Signature

The `flash_attn.layers.rotary.RotaryEmbedding.forward()` method signature:

```python
def forward(
    self,
    qkv: torch.Tensor,
    kv: Optional[torch.Tensor] = None,
    seqlen_offset: Union[int, torch.Tensor] = 0,
    max_seqlen: Optional[int] = None,
    num_heads_q: Optional[int] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
```

**Key Parameters**:
- `seqlen_offset`: Supports either an integer or tensor. Used to shift sequences, particularly useful during inference with KV cache.
- **Position IDs**: The method does **NOT** include a `position_ids` parameter.

## Summary

When using `DataCollatorForCLMWithPositionIds`:

1. ✅ **`position_ids` ARE used** to create `label_mask` - masking document boundaries
2. ❌ **`position_ids` are NOT used** for rotary position embeddings during training
3. ⚠️ **Limitation**: Flash attention's `RotaryEmbedding.forward()` doesn't accept `position_ids`, only `seqlen_offset`
4. 📝 **Comment in code** (line 413): "NOTE: Only supported for training (TODO(fmom): position_ids not supported yet)"

## To Properly Support Custom `position_ids` in Training

You would need to:

1. **Modify `LlamaForTraining.forward()`** to accept `position_ids` parameter
2. **Modify `LlamaModel.forward_with_hidden_states()`** to pass `position_ids` through to the attention layers
3. **Update `CausalSelfAttention._forward_training()`** to use custom position logic compatible with flash attention's `seqlen_offset` parameter
4. **Handle the flash_attn API limitation** - since `flash_rotary_embedding` doesn't accept `position_ids`, you would need to either:
   - Use `seqlen_offset` if your position pattern fits (e.g., all sequences start from same offset)
   - Fall back to the non-flash rotary embedding implementation that accepts `position_ids`
   - Manually apply rotary embeddings with custom positions before calling flash attention

## Current Use Case

The current implementation of `DataCollatorForCLMWithPositionIds` is primarily useful for:
- **Document packing** with proper loss masking at document boundaries
- Preventing cross-document attention leakage through the label mask
- **Not** for custom positional encoding during training (positions are always sequential within each sequence)
