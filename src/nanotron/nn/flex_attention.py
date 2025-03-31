"""
Modular masking utilities for FlexAttention implementation.
"""
from typing import Callable, Optional, Tuple

import torch
from torch.nn.attention.flex_attention import create_block_mask
import functools


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


@functools.lru_cache(maxsize=32)
def create_block_mask_cached(mask_func, B, H, Q_LEN, KV_LEN, device="cuda"):
    """Cached version of create_block_mask for better performance."""
    block_mask = create_block_mask(mask_func, B, H, Q_LEN, KV_LEN, device=device)
    return block_mask


def _offsets_to_doc_ids_tensor(offsets):
    """Converts offsets to document IDs tensor.
    
    Args:
        offsets: Tensor of shape [num_docs + 1] containing cumulative token counts
        
    Returns:
        Tensor of document IDs for each token position
    """
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )


def lengths_to_offsets(lengths, device):
    """Converts a list of document lengths to offsets.
    
    Args:
        lengths: List or tensor of document lengths
        device: Device for the resulting tensor
        
    Returns:
        Tensor of cumulative offsets
    """
    if not isinstance(lengths, torch.Tensor):
        offsets = [0]
        offsets.extend(lengths)
        offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    else:
        offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.int32), lengths])
    
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets


def generate_doc_mask_mod(inner_mask_func, offsets):
    """Generates a document-aware mask function.
    
    Args:
        inner_mask_func: Base mask function to apply within each document
        offsets: Tensor of shape [num_docs + 1] containing cumulative token counts
        
    Returns:
        Document-aware mask function
    """
    document_ids = _offsets_to_doc_ids_tensor(offsets)
    
    def doc_mask_mod(b, h, q_idx, kv_idx):
        # Only attend within the same document
        same_doc = document_ids[q_idx] == document_ids[kv_idx]
        
        # Convert to document-local indices
        q_local_idx = q_idx - offsets[document_ids[q_idx]]
        kv_local_idx = kv_idx - offsets[document_ids[kv_idx]]
        
        # Apply the inner mask function using document-local indices
        inner_mask = inner_mask_func(b, h, q_local_idx, kv_local_idx)
        
        return same_doc & inner_mask
    
    return doc_mask_mod


def causal_mask_func(b, h, q_idx, kv_idx):
    """Simple causal masking function."""
    return q_idx >= kv_idx


def sliding_window_causal_mask_func(window_size, b, h, q_idx, kv_idx):
    """Sliding window causal masking function."""
    causal = q_idx >= kv_idx
    window = q_idx - kv_idx <= window_size
    return causal & window


def get_block_mask_from_type(
    flex_attention_mask: Optional[str],
    query: torch.Tensor,
    key: torch.Tensor,
    sliding_window: Optional[int] = None,
    position_ids: Optional[torch.Tensor] = None,
    document_ids: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Get a specific block mask based on mask type.
    
    Args:
        flex_attention_mask: String identifier for mask type
        query: Query states tensor [batch_size, num_heads, seq_len, head_dim]
        key: Key states tensor [batch_size, num_kv_heads, seq_len, head_dim]
        sliding_window: Optional sliding window size
        position_ids: Optional tensor of position IDs for document masking
        document_ids: Optional explicit document IDs tensor
        
    Returns:
        Block mask tensor or None
    """
    if flex_attention_mask is None:
        return None
        
    seq_len = query.size(2)
    device = query.device
    
    if flex_attention_mask == "causal":
        # Simple causal mask - flex_attention handles this efficiently by default
        return None
        
    elif flex_attention_mask == "sliding_window":
        # Assert that sliding_window is provided and valid
        assert sliding_window is not None and sliding_window > 0, \
            f"For 'sliding_window' mask type, sliding_window must be provided and > 0, got {sliding_window}"
            
        # Use partial to bind the window size parameter
        mask_func = functools.partial(sliding_window_causal_mask_func, sliding_window)
        
        return create_block_mask_cached(
            mask_func,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )
        
    elif flex_attention_mask == "document":
        # Assert that at least one of document_ids or position_ids is provided
        assert document_ids is not None or position_ids is not None, \
            "For 'document' mask type, either document_ids or position_ids must be provided"
        
        # Handle document masking
        if document_ids is not None:
            # Ensure document_ids is on the right device
            document_ids = document_ids.to(device)
            
            # Validate document_ids
            assert document_ids.numel() > 0, "document_ids tensor cannot be empty"
            
            # If explicit document_ids are provided, convert to offsets format
            unique_docs = torch.unique(document_ids, sorted=True)
            offsets = []
            current_offset = 0
            
            offsets.append(current_offset)
            for doc_id in unique_docs:
                doc_length = (document_ids == doc_id).sum().item()
                current_offset += doc_length
                offsets.append(current_offset)
                
            offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
            
        elif position_ids is not None:
            # Ensure position_ids is on the right device
            position_ids = position_ids.to(device)
            
            # Validate position_ids
            assert position_ids.numel() > 0, "position_ids tensor cannot be empty"
            
            # If position_ids are provided (where 0s indicate document starts)
            if position_ids.dim() == 2:
                position_ids = position_ids.view(-1)
                
            # Find document boundaries (positions where position_id is 0)
            doc_starts = torch.where(position_ids == 0)[0]
            
            # Validate document boundaries
            assert doc_starts.numel() > 0, "No document boundaries found in position_ids (no zeros found)"
            
            # Add the total sequence length as the final boundary
            doc_ends = torch.cat([doc_starts[1:], torch.tensor([len(position_ids)], device=device)])
            
            # Calculate document lengths
            doc_lengths = doc_ends - doc_starts
            
            # Convert to offsets
            offsets = lengths_to_offsets(doc_lengths, device)
            
            # Adjust offsets to account for actual starting positions
            if doc_starts[0] != 0:
                offsets = offsets + doc_starts[0]
        
        # Generate document-aware causal mask
        doc_mask_mod = generate_doc_mask_mod(causal_mask_func, offsets)
        
        return create_block_mask_cached(
            doc_mask_mod,
            B=None,  # Not needed since we handle batching in the mask function
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )
    else:
        raise ValueError(f"Unknown flex_attention_mask type: {flex_attention_mask}. " 
                         f"Supported types are: 'causal', 'sliding_window', 'document'")
    
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


def validate_attention_args(
    flex_attention_mask: Optional[str],
    sliding_window: Optional[int] = None,
    position_ids: Optional[torch.Tensor] = None, 
    document_ids: Optional[torch.Tensor] = None
) -> None:
    """Validates arguments for attention mask creation.
    
    Args:
        flex_attention_mask: String identifier for mask type
        sliding_window: Optional sliding window size
        position_ids: Optional position IDs tensor
        document_ids: Optional document IDs tensor
        
    Raises:
        ValueError: If arguments are not valid for the specified mask type
    """
    if flex_attention_mask is None:
        return
        
    if flex_attention_mask == "causal":
        # No additional parameters needed for causal mask
        return
        
    elif flex_attention_mask == "sliding_window":
        if sliding_window is None or sliding_window <= 0:
            raise ValueError(f"For 'sliding_window' mask type, sliding_window must be a positive integer, got {sliding_window}")
            
    elif flex_attention_mask == "document":
        if document_ids is None and position_ids is None:
            raise ValueError("For 'document' mask type, either document_ids or position_ids must be provided")
            
        if position_ids is not None and position_ids.numel() == 0:
            raise ValueError("position_ids tensor cannot be empty")
            
        if document_ids is not None and document_ids.numel() == 0:
            raise ValueError("document_ids tensor cannot be empty")
            
    else:
        raise ValueError(f"Unknown flex_attention_mask type: {flex_attention_mask}. "
                         f"Supported types are: 'causal', 'sliding_window', 'document'")
