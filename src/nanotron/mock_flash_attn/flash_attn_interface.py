"""
Mock flash_attn.flash_attn_interface module.
Replaces flash attention functions with PyTorch's native scaled_dot_product_attention.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def _make_causal_mask(q_len: int, kv_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a causal attention mask."""
    mask = torch.ones(q_len, kv_len, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=kv_len - q_len + 1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


def _expand_mask_for_gqa(mask: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    """Expand attention mask for grouped query attention."""
    if num_heads == num_kv_heads:
        return mask
    # For GQA, we need to expand the kv heads
    num_groups = num_heads // num_kv_heads
    return mask.repeat_interleave(num_groups, dim=1)


def flash_attn_func(
    q: torch.Tensor,  # [batch_size, seqlen_q, num_heads, head_dim]
    k: torch.Tensor,  # [batch_size, seqlen_k, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, seqlen_k, num_kv_heads, head_dim]
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Mock implementation of flash_attn_func using PyTorch's scaled_dot_product_attention.

    Args:
        q: Query tensor [batch_size, seqlen_q, num_heads, head_dim]
        k: Key tensor [batch_size, seqlen_k, num_kv_heads, head_dim]
        v: Value tensor [batch_size, seqlen_k, num_kv_heads, head_dim]
        dropout_p: Dropout probability
        softmax_scale: Softmax scale (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        window_size: Sliding window size (not fully supported in mock)
        alibi_slopes: ALiBi slopes (not supported in mock)
        deterministic: Whether to use deterministic algorithms
        return_attn_probs: Whether to return attention probabilities

    Returns:
        Output tensor [batch_size, seqlen_q, num_heads, head_dim]
    """
    batch_size, seqlen_q, num_heads, head_dim = q.shape
    _, seqlen_k, num_kv_heads, _ = k.shape

    # Transpose to [batch_size, num_heads, seqlen, head_dim] for SDPA
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Handle GQA by expanding k and v
    if num_kv_heads != num_heads:
        num_groups = num_heads // num_kv_heads
        k = k.repeat_interleave(num_groups, dim=1)
        v = v.repeat_interleave(num_groups, dim=1)

    # Scale
    scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(head_dim))

    # Use PyTorch SDPA
    output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=dropout_p if q.requires_grad else 0.0,
        is_causal=causal,
        scale=scale,
    )

    # Transpose back to [batch_size, seqlen_q, num_heads, head_dim]
    output = output.transpose(1, 2)

    if return_attn_probs:
        # Return dummy attention probs for compatibility
        return output, None, None
    return output


def flash_attn_varlen_func(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    k: torch.Tensor,  # [total_k, num_kv_heads, head_dim]
    v: torch.Tensor,  # [total_k, num_kv_heads, head_dim]
    cu_seqlens_q: torch.Tensor,  # [batch_size + 1]
    cu_seqlens_k: torch.Tensor,  # [batch_size + 1]
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Mock implementation of flash_attn_varlen_func for variable length sequences.

    This implementation pads sequences to max length and uses SDPA.
    """
    device = q.device
    dtype = q.dtype
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = k.shape[1]
    batch_size = len(cu_seqlens_q) - 1

    # Pad to batch format
    q_padded = torch.zeros(batch_size, max_seqlen_q, num_heads, head_dim, device=device, dtype=dtype)
    k_padded = torch.zeros(batch_size, max_seqlen_k, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_padded = torch.zeros(batch_size, max_seqlen_k, num_kv_heads, head_dim, device=device, dtype=dtype)

    # Create attention mask for variable length sequences
    attn_mask = torch.zeros(batch_size, 1, max_seqlen_q, max_seqlen_k, device=device, dtype=dtype)

    for i in range(batch_size):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_start, k_end = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()
        q_len = q_end - q_start
        k_len = k_end - k_start

        q_padded[i, :q_len] = q[q_start:q_end]
        k_padded[i, :k_len] = k[k_start:k_end]
        v_padded[i, :k_len] = v[k_start:k_end]

        # Create mask: 1 for valid positions, -inf for padding
        if causal:
            # Causal mask within the valid region
            for qi in range(q_len):
                for ki in range(k_len):
                    if ki <= qi:  # Can attend to current and past positions
                        attn_mask[i, 0, qi, ki] = 0.0
                    else:
                        attn_mask[i, 0, qi, ki] = float("-inf")
            # Mask out padding
            attn_mask[i, 0, q_len:, :] = float("-inf")
            attn_mask[i, 0, :, k_len:] = float("-inf")
        else:
            # Non-causal: attend to all valid positions
            attn_mask[i, 0, :q_len, :k_len] = 0.0
            attn_mask[i, 0, q_len:, :] = float("-inf")
            attn_mask[i, 0, :q_len, k_len:] = float("-inf")
            attn_mask[i, 0, q_len:, :] = float("-inf")

    # Transpose for SDPA: [batch, heads, seq, dim]
    q_padded = q_padded.transpose(1, 2)
    k_padded = k_padded.transpose(1, 2)
    v_padded = v_padded.transpose(1, 2)

    # Handle GQA
    if num_kv_heads != num_heads:
        num_groups = num_heads // num_kv_heads
        k_padded = k_padded.repeat_interleave(num_groups, dim=1)
        v_padded = v_padded.repeat_interleave(num_groups, dim=1)

    # Expand mask for all heads
    attn_mask = attn_mask.expand(-1, num_heads, -1, -1)

    scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(head_dim))

    # Use SDPA
    output = F.scaled_dot_product_attention(
        q_padded, k_padded, v_padded,
        attn_mask=attn_mask,
        dropout_p=dropout_p if q.requires_grad else 0.0,
        is_causal=False,  # We handle causal in the mask
        scale=scale,
    )

    # Transpose back: [batch, seq, heads, dim]
    output = output.transpose(1, 2)

    # Unpad to variable length format
    outputs = []
    for i in range(batch_size):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        q_len = q_end - q_start
        outputs.append(output[i, :q_len])

    result = torch.cat(outputs, dim=0)

    if return_attn_probs:
        return result, None, None
    return result


def flash_attn_with_kvcache(
    q: torch.Tensor,  # [batch_size, seqlen_q, num_heads, head_dim]
    k_cache: torch.Tensor,  # [batch_size, max_seqlen_k, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [batch_size, max_seqlen_k, num_kv_heads, head_dim]
    k: Optional[torch.Tensor] = None,  # [batch_size, seqlen_k, num_kv_heads, head_dim]
    v: Optional[torch.Tensor] = None,  # [batch_size, seqlen_k, num_kv_heads, head_dim]
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,  # [batch_size]
    cache_batch_idx: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    rotary_interleaved: bool = True,
    alibi_slopes: Optional[torch.Tensor] = None,
    num_splits: int = 0,
) -> torch.Tensor:
    """
    Mock implementation of flash_attn_with_kvcache for inference with KV cache.
    """
    batch_size, seqlen_q, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]

    # Update cache with new k, v if provided
    if k is not None and v is not None and cache_seqlens is not None:
        for i in range(batch_size):
            cache_len = cache_seqlens[i].item()
            new_len = k.shape[1]
            k_cache[i, cache_len:cache_len + new_len] = k[i]
            v_cache[i, cache_len:cache_len + new_len] = v[i]

    # Get effective k, v from cache
    if cache_seqlens is not None:
        max_cache_len = cache_seqlens.max().item() + (k.shape[1] if k is not None else 0)
        k_eff = k_cache[:, :max_cache_len]
        v_eff = v_cache[:, :max_cache_len]
    else:
        k_eff = k_cache
        v_eff = v_cache

    # Use flash_attn_func
    return flash_attn_func(
        q, k_eff, v_eff,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        return_attn_probs=False,
    )


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv: torch.Tensor,  # [total_k, 2, num_kv_heads, head_dim]
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Mock implementation of flash_attn_varlen_kvpacked_func."""
    k = kv[:, 0]  # [total_k, num_kv_heads, head_dim]
    v = kv[:, 1]  # [total_k, num_kv_heads, head_dim]

    return flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )


def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,  # [batch_size, seqlen, 3, num_heads, head_dim]
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Mock implementation of flash_attn_qkvpacked_func."""
    q = qkv[:, :, 0]  # [batch_size, seqlen, num_heads, head_dim]
    k = qkv[:, :, 1]
    v = qkv[:, :, 2]

    return flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )
