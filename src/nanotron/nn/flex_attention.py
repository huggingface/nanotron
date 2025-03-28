"""
Modular masking utilities for FlexAttention implementation.
"""
from typing import Callable, Optional, Tuple

import torch
from torch.nn.attention.flex_attention import create_block_mask


def create_softcapped_causal_score_mod(
    softcap: Optional[float],
    causal_mask: Optional[torch.Tensor],
) -> Optional[Callable]:
    """Creates a score modifier function that applies softcapping and causal masking.

    Args:
        softcap: Optional softcap value for attention scores
        causal_mask: Optional causal mask tensor [batch, 1, seq_len, seq_len]

    Returns:
        Callable score modifier function or None if no modifications needed
    """
    if softcap is None and causal_mask is None:
        return None

    def score_mod(score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int) -> torch.Tensor:
        if softcap is not None:
            score = softcap * torch.tanh(score / softcap)
        if causal_mask is not None:
            score = score + causal_mask[b][0][q_idx][kv_idx]
        return score

    return score_mod


def create_document_mask_func(
    query: torch.Tensor,
    document_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> Optional[Callable]:
    """Creates a document masking function based on either document_ids or position_ids.

    Args:
        query: Query tensor for shape information [batch, num_heads, seq_len, head_dim]
        document_ids: Optional tensor marking document boundaries [seq_len]
        position_ids: Optional tensor of position IDs [batch, seq_len] or [seq_len]

    Returns:
        Document masking function or None if no document masking needed
    """
    if document_ids is not None:
        document_ids = document_ids.to(query.device)

        def document_mask_direct(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
            seq_len = query.size(2)
            q_flat_idx = b * seq_len + q_idx
            kv_flat_idx = b * seq_len + kv_idx

            q_doc = document_ids[q_flat_idx] if q_flat_idx < len(document_ids) else -1
            kv_doc = document_ids[kv_flat_idx] if kv_flat_idx < len(document_ids) else -1

            return q_doc == kv_doc

        return document_mask_direct

    elif position_ids is not None:
        seq_len = query.size(2)

        if position_ids.dim() == 2:
            position_ids = position_ids.view(-1)

        doc_starts = torch.where(position_ids == 0)[0]

        def document_mask_from_positions(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
            q_flat_idx = b * seq_len + q_idx
            kv_flat_idx = b * seq_len + kv_idx

            q_doc_idx = torch.searchsorted(doc_starts, q_flat_idx) - 1
            kv_doc_idx = torch.searchsorted(doc_starts, kv_flat_idx) - 1

            return q_doc_idx == kv_doc_idx

        return document_mask_from_positions

    return None


def create_attention_mask(
    query: torch.Tensor,
    sliding_window: Optional[int] = None,
    doc_mask_func: Optional[Callable] = None,
) -> Optional[torch.Tensor]:
    """Creates an attention mask combining sliding window and document masking.

    Args:
        query: Query tensor for shape information [batch, num_heads, seq_len, head_dim]
        sliding_window: Optional sliding window size
        doc_mask_func: Optional document masking function

    Returns:
        Block mask tensor or None if no masking needed
    """
    if sliding_window is None and doc_mask_func is None:
        return None

    seq_len = query.size(2)

    if sliding_window is not None:
        def sliding_window_causal(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
            causal_mask = q_idx >= kv_idx
            window_mask = q_idx - kv_idx <= sliding_window
            if doc_mask_func is not None:
                doc_mask = doc_mask_func(b, h, q_idx, kv_idx)
                return causal_mask & window_mask & doc_mask
            return causal_mask & window_mask

        mask_func = sliding_window_causal
    else:
        def doc_causal_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
            causal_mask = q_idx >= kv_idx
            doc_mask = doc_mask_func(b, h, q_idx, kv_idx)
            return causal_mask & doc_mask

        mask_func = doc_causal_mask

    return create_block_mask(
        mask_func,
        B=query.size(0) if doc_mask_func is not None else None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
    )


def get_block_mask_from_type(
    flex_attention_mask: Optional[str],
    query: torch.Tensor,
    key: torch.Tensor,
    sliding_window: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Get a specific block mask based on mask type.
    
    Args:
        flex_attention_mask: String identifier for mask type
        query: Query states tensor [batch_size, num_heads, seq_len, head_dim]
        key: Key states tensor [batch_size, num_kv_heads, seq_len, head_dim]
        sliding_window: Optional sliding window size
        
    Returns:
        Block mask tensor or None
    """
    if flex_attention_mask is None:
        return None
        
    seq_len = query.size(2)
    
    if flex_attention_mask == "causal":
        # Simple causal mask
        def causal_mask_func(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
            return q_idx >= kv_idx
            
        return create_block_mask(
            causal_mask_func,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
        )
        
    elif flex_attention_mask == "sliding_window_causal" and sliding_window is not None:
        # Sliding window causal mask
        def sliding_window_mask_func(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
            causal = q_idx >= kv_idx
            window = q_idx - kv_idx <= sliding_window
            return causal and window
            
        return create_block_mask(
            sliding_window_mask_func,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
        )
        
    elif flex_attention_mask == "local_attention":
        # Local attention with equal window on both sides (non-causal)
        window_size = sliding_window if sliding_window is not None else 128
        
        def local_attention_mask_func(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
            return abs(q_idx - kv_idx) <= window_size
            
        return create_block_mask(
            local_attention_mask_func,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
        )
    
    return None


def get_attention_mod_from_type(
    flex_attention_mask: Optional[str],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> Optional[Callable]:
    """Get a specific attention modifier based on mask type.
    
    Args:
        flex_attention_mask: String identifier for mask type
        query: Query states tensor [batch_size, num_heads, seq_len, head_dim]
        key: Key states tensor [batch_size, num_kv_heads, seq_len, head_dim]
        value: Value states tensor [batch_size, num_kv_heads, seq_len, head_dim]
        
    Returns:
        Callable attention modifier function or None
    """
    # We're setting this to None as specified in the requirements
    return None
