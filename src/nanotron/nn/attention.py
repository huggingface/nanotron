from functools import lru_cache
from typing import Literal, Optional, Tuple

import torch
from packaging import version

from nanotron.nn.ring_attention import ring_flash_attn_varlen_func
from nanotron.nn.llama3_ring_attention import llama3_flash_attn_varlen_qkvpacked_func

# Replace direct import with a function for lazy loading
def get_ring_flash_attn_cuda():
    """Lazily import ring_flash_attn_cuda to avoid early Triton dependency."""
    from nanotron.nn.ring_attention_lucidrain import ring_flash_attn_cuda

    return ring_flash_attn_cuda


@lru_cache()
def is_torch_flex_attn_available():
    # TODO check if some bugs cause push backs on the exact version
    # NOTE: We require torch>=2.5.0 as it is the first release
    return version.parse(torch.__version__) >= version.parse("2.5.0")


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention


@lru_cache()
def is_flash_attn_greater_or_equal_2_10():
    try:
        import flash_attn

        return version.parse(flash_attn.__version__) >= version.parse("2.1.0")
    except ImportError:
        return False


if is_flash_attn_greater_or_equal_2_10():
    from flash_attn.flash_attn_interface import flash_attn_func
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
    flex_attention_mask: Optional[str] = None,
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
        flex_attention_mask: Optional string specifying a custom mask type
        
    Returns:
        Tuple of (attention_output, attention_weights)
    """
    from nanotron.nn.flex_attention import (
        create_softcapped_causal_score_mod,
        create_document_mask_func,
        create_attention_mask,
        get_attention_mod_from_type,
        get_block_mask_from_type,
        validate_attention_args,
    )

    # Validate arguments if a flex_attention_mask is specified
    validate_attention_args(
        flex_attention_mask=flex_attention_mask,
        sliding_window=sliding_window,
        position_ids=position_ids,
        document_ids=document_ids,
    )

    # We're setting score_mod to None as requested
    score_mod = None

    # Determine which block mask to use
    if flex_attention_mask is not None:
        # Use the mask type specified by flex_attention_mask
        block_mask = get_block_mask_from_type(
            flex_attention_mask=flex_attention_mask,
            query=query,
            key=key,
            sliding_window=sliding_window,
            position_ids=position_ids,
            document_ids=document_ids,
        )
    else:
        # Use the existing document/sliding window masking logic
        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]
            
        # Create document masking function if needed
        doc_mask_func = create_document_mask_func(query, document_ids, position_ids)

        # Create combined attention mask
        block_mask = create_attention_mask(query, sliding_window, doc_mask_func)

    # Call PyTorch's flex_attention with the appropriate parameters
    attn_output, attention_weights = flex_attention(
        query,
        key,
        value,
        enable_gqa=True,  # Enable grouped query attention
        score_mod=score_mod,
        block_mask=block_mask,  # Efficient mask based on type
        scale=scaling,
        return_lse=True,  # FlexAttention always computes log-sum-exp anyway
    )

    # FlexAttention returns weights in float32, convert to match value dtype
    attention_weights = attention_weights.to(value.dtype)

    # Transpose output to match expected format [batch_size, seq_len, num_heads, head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attention_weights


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,  # [b, num_heads, seq_len, head_dim]
    key: torch.Tensor,  # [b, num_kv_heads, seq_len, head_dim]
    value: torch.Tensor,  # [b, num_kv_heads, seq_len, head_dim]
    attention_mask: Optional[torch.Tensor],  # [b, num_heads, seq_len, seq_len]
    max_seqlen: Optional[int],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    query = query.view(-1, max_seqlen, module.local_num_heads, module.head_dim)
    key = key.view(-1, max_seqlen, module.local_num_kv_heads, module.head_dim)
    value = value.view(-1, max_seqlen, module.local_num_kv_heads, module.head_dim)

    if attention_mask is None:
        is_causal = True
    else:
        is_causal = False

    if sliding_window is not None:
        window_size = (sliding_window, sliding_window)
    else:
        window_size = (-1, -1)

    attn_output = flash_attn_func(
        q=query,
        k=key,
        v=value,
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=is_causal,
        window_size=window_size,
        return_attn_probs=False,
    )
    attn_output = attn_output.contiguous()
    return attn_output, None


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,  # [b, num_heads, seq_len, head_dim]
    key: torch.Tensor,  # [b, num_kv_heads, seq_len, head_dim]
    value: torch.Tensor,  # [b, num_kv_heads, seq_len, head_dim]
    attention_mask: Optional[torch.Tensor],  # [b, num_heads, seq_len, seq_len]
    max_seqlen: int,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    if attention_mask is None:
        is_causal = True
    else:
        is_causal = False
    query = query.view(-1, max_seqlen, module.local_num_heads, module.head_dim).transpose(
        1, 2
    )  # [b, num_heads, seq_length, head_dim]
    key = key.view(-1, max_seqlen, module.local_num_kv_heads, module.head_dim).transpose(1, 2)
    value = value.view(-1, max_seqlen, module.local_num_kv_heads, module.head_dim).transpose(1, 2)
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        enable_gqa=query.shape[1] != key.shape[1],
    )
    attn_output = attn_output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
    return attn_output, None


ALL_ATTENTION_FUNCTIONS = {
    "flash_attention_2": flash_attention_forward,
    "flex_attention": flex_attention_forward,
    "sdpa": sdpa_attention_forward,
    "ring_flash_triton": lambda *args, **kwargs: get_ring_flash_attn_cuda()(*args, **kwargs),
    "ring": ring_flash_attn_varlen_func,
    "llama3_ring_attention": llama3_flash_attn_varlen_qkvpacked_func,
}

AttentionImplementation = Literal[tuple(ALL_ATTENTION_FUNCTIONS.keys())]


# TODO @nouamane: optimize this, and make sure it works with flashattn and flexattn
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
