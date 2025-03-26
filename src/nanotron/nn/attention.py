from typing import Literal, Optional, Tuple

import torch
from packaging import version
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from nanotron.nn.ring_attention import ring_flash_attn_varlen_func


# Replace direct import with a function for lazy loading
def get_ring_flash_attn_cuda():
    """Lazily import ring_flash_attn_cuda to avoid early Triton dependency."""
    from nanotron.nn.ring_attention_lucidrain import ring_flash_attn_cuda

    return ring_flash_attn_cuda


def is_torch_flex_attn_available():
    # TODO check if some bugs cause push backs on the exact version
    # NOTE: We require torch>=2.5.0 as it is the first release
    return version.parse(torch.__version__) >= version.parse("2.5.0")


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention


# adapted from transformers.integrations.flex_attention.flex_attention_forward
def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    sliding_window: Optional[int] = None,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    position_ids: Optional[torch.Tensor] = None,
    document_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implementation of attention using PyTorch's FlexAttention.

    Args:
        module: The module calling this function
        query: Query states tensor [batch_size, num_heads, seq_len, head_dim]
        key: Key states tensor [batch_size, num_kv_heads, seq_len, head_dim]
        value: Value states tensor [batch_size, num_kv_heads, seq_len, head_dim]
        attention_mask: Optional attention mask tensor
        sliding_window: Optional sliding window size for efficient attention
        scaling: Optional scaling factor for attention scores
        softcap: Optional softcap for softmax stability
        position_ids: Optional tensor of position IDs used for document masking
        document_ids: Optional tensor explicitly marking document boundaries [seq_len]
                     (e.g., [0,0,0,1,1,2,2,2,2,2,2] for seqs of length 3,2,6)

    Returns:
        Tuple of (attention_output, attention_weights)
    """
    # Initialize parameters for FlexAttention
    score_mod = None
    block_mask = None

    # Handle causal mask with softcapping
    if softcap is not None and attention_mask is not None:
        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]

        def causal_score_mod(score, b, h, q_idx, kv_idx):
            # Apply softcap if specified
            if softcap is not None:
                score = softcap * torch.tanh(score / softcap)
            # Apply attention mask if provided
            if causal_mask is not None:
                score = score + causal_mask[b][0][q_idx][kv_idx]
            return score

        score_mod = causal_score_mod

    # Document masking for packed sequences
    doc_mask_func = None

    # Direct document_ids approach (preferred if available)
    if document_ids is not None:
        # Ensure document_ids is on the right device
        document_ids = document_ids.to(query.device)

        def document_mask_direct(b, h, q_idx, kv_idx):
            # Calculate global indices accounting for batch dimension
            seq_len = query.size(2)
            q_flat_idx = b * seq_len + q_idx
            kv_flat_idx = b * seq_len + kv_idx

            # Check if both tokens are in the same document
            q_doc = document_ids[q_flat_idx] if q_flat_idx < len(document_ids) else -1
            kv_doc = document_ids[kv_flat_idx] if kv_flat_idx < len(document_ids) else -1

            # Allow attention only within the same document
            return q_doc == kv_doc

        doc_mask_func = document_mask_direct

    # Position_ids based approach (fallback)
    elif position_ids is not None:
        # Extract document boundaries from position_ids
        # position_ids resets to 0 at the start of each document
        query.size(0)
        seq_len = query.size(2)

        # If position_ids is [batch_size, seq_length], we need to reshape to match attention indices
        if position_ids.dim() == 2:
            position_ids = position_ids.view(-1)

        # Find document boundaries from position resets
        doc_starts = torch.where(position_ids == 0)[0]

        # Create document masking function
        def document_mask_from_positions(b, h, q_idx, kv_idx):
            # Calculate the absolute positions in the flattened sequence
            q_flat_idx = b * seq_len + q_idx
            kv_flat_idx = b * seq_len + kv_idx

            # Find which document each token belongs to
            # A token at index i belongs to document j if doc_starts[j] <= i < doc_starts[j+1]
            q_doc_idx = torch.searchsorted(doc_starts, q_flat_idx) - 1
            kv_doc_idx = torch.searchsorted(doc_starts, kv_flat_idx) - 1

            # Allow attention only within the same document
            return q_doc_idx == kv_doc_idx

        doc_mask_func = document_mask_from_positions

    # Handle sliding window attention using block_mask (more efficient)
    if sliding_window is not None:
        # Define sliding window causal mask function
        def sliding_window_causal(b, h, q_idx, kv_idx):
            # Apply causal masking (only attend to past tokens)
            causal_mask = q_idx >= kv_idx
            # Apply sliding window constraint
            window_mask = q_idx - kv_idx <= sliding_window
            # Apply document masking if available
            if doc_mask_func is not None:
                doc_mask = doc_mask_func(b, h, q_idx, kv_idx)
                return causal_mask & window_mask & doc_mask
            return causal_mask & window_mask

        # Create a block mask for more efficient processing
        seq_len = query.size(2)
        block_mask = create_block_mask(
            sliding_window_causal,
            B=query.size(0) if doc_mask_func is not None else None,  # Need batch dim for doc masking
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
        )
    elif doc_mask_func is not None:
        # If we only have document masking without sliding window
        def doc_causal_mask(b, h, q_idx, kv_idx):
            # Apply causal masking (only attend to past tokens)
            causal_mask = q_idx >= kv_idx
            # Apply document masking
            doc_mask = doc_mask_func(b, h, q_idx, kv_idx)
            return causal_mask & doc_mask

        # Create a block mask for document masking
        seq_len = query.size(2)
        block_mask = create_block_mask(
            doc_causal_mask, B=query.size(0), H=None, Q_LEN=seq_len, KV_LEN=seq_len  # Need batch dim for doc masking
        )

    # Call PyTorch's flex_attention with the appropriate parameters
    attn_output, attention_weights = flex_attention(
        query,
        key,
        value,
        enable_gqa=True,  # Enable grouped query attention
        score_mod=score_mod,
        block_mask=block_mask,  # Efficient for both sliding window and document masking
        scale=scaling,
        return_lse=True,  # FlexAttention always computes log-sum-exp anyway
    )

    # FlexAttention returns weights in float32, convert to match value dtype
    attention_weights = attention_weights.to(value.dtype)

    # Transpose output to match expected format [batch_size, seq_len, num_heads, head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attention_weights


ALL_ATTENTION_FUNCTIONS = {
    "flash_attention_2": flash_attention_forward,
    "flex_attention": flex_attention_forward,
    "sdpa": sdpa_attention_forward,
    "ring_flash_triton": lambda *args, **kwargs: get_ring_flash_attn_cuda()(*args, **kwargs),
    "ring": ring_flash_attn_varlen_func,
}

AttentionImplementation = Literal[tuple(ALL_ATTENTION_FUNCTIONS.keys())]


# TODO @nouamane: optimize this, and make sure it works with flashattn and flexattn
@torch.jit.script
def get_attention_mask(position_ids, seq_length):
    attention_mask = torch.zeros(seq_length, seq_length, device=position_ids.device)
    start_indices = torch.where(position_ids == 0)[0]
    cu_seqlens = torch.cat(
        [start_indices, torch.tensor([seq_length], dtype=torch.int32, device=start_indices.device)]
    ).to(torch.int32)
    # make trius for each document
    for i in range(len(cu_seqlens) - 1):
        attention_mask[cu_seqlens[i] : cu_seqlens[i + 1], cu_seqlens[i] : cu_seqlens[i + 1]] = torch.tril(
            torch.ones(cu_seqlens[i + 1] - cu_seqlens[i], cu_seqlens[i + 1] - cu_seqlens[i])
        )
    return attention_mask.to(torch.bool), cu_seqlens  # [seq_length, seq_length]
