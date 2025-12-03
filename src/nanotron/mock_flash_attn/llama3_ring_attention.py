"""
Mock llama3_ring_attention module for local debugging without CUDA/Triton.
Provides stub implementations that raise errors when called, since ring attention
requires distributed communication that doesn't work without a proper GPU setup.
"""

import torch
from typing import Optional, Tuple


def llama3_flash_attn_prepare_cu_seqlens(
    cu_seqlens: torch.Tensor, causal: bool, rank: int, world_size: int
) -> Tuple[torch.Tensor, torch.Tensor, slice]:
    """
    Mock llama3_flash_attn_prepare_cu_seqlens.

    This function requires distributed GPU communication which is not available in mock mode.
    """
    raise NotImplementedError(
        "llama3_flash_attn_prepare_cu_seqlens is not available in mock mode. "
        "Ring attention requires CUDA and distributed GPU communication. "
        "Please use 'flash_attention_2', 'sdpa', or 'flex_attention' instead."
    )


def llama3_flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    local_k_slice: slice,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Mock llama3 ring flash attention varlen kvpacked function.

    Ring attention requires distributed communication across GPUs,
    which is not available in the mock environment.
    """
    raise NotImplementedError(
        "llama3_flash_attn_varlen_kvpacked_func is not available in mock mode. "
        "Ring attention requires CUDA and distributed GPU communication. "
        "Please use 'flash_attention_2', 'sdpa', or 'flex_attention' instead."
    )


def llama3_flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    local_k_slice: slice,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Mock llama3 ring flash attention varlen function.

    Ring attention requires distributed communication across GPUs,
    which is not available in the mock environment.
    """
    raise NotImplementedError(
        "llama3_flash_attn_varlen_func is not available in mock mode. "
        "Ring attention requires CUDA and distributed GPU communication. "
        "Please use 'flash_attention_2', 'sdpa', or 'flex_attention' instead."
    )


def llama3_flash_attn_varlen_qkvpacked_func(
    module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    ring_pg=None,
    **kwargs,
) -> Tuple[torch.Tensor]:
    """
    Mock llama3 ring flash attention varlen qkvpacked function.

    Ring attention requires distributed communication across GPUs,
    which is not available in the mock environment.
    """
    raise NotImplementedError(
        "llama3_flash_attn_varlen_qkvpacked_func is not available in mock mode. "
        "Ring attention requires CUDA and distributed GPU communication. "
        "Please use 'flash_attention_2', 'sdpa', or 'flex_attention' instead."
    )


class Llama3RingFlashAttnFunc(torch.autograd.Function):
    """Mock Llama3RingFlashAttnFunc class."""

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError("Llama3RingFlashAttnFunc requires CUDA and Triton.")

    @staticmethod
    def backward(ctx, *args):
        raise NotImplementedError("Llama3RingFlashAttnFunc requires CUDA and Triton.")


def llama3_ring_flash_attn_varlen_forward(*args, **kwargs):
    """Mock llama3_ring_flash_attn_varlen_forward."""
    raise NotImplementedError("llama3_ring_flash_attn_varlen_forward requires CUDA and Triton.")


def llama3_ring_flash_attn_varlen_backward(*args, **kwargs):
    """Mock llama3_ring_flash_attn_varlen_backward."""
    raise NotImplementedError("llama3_ring_flash_attn_varlen_backward requires CUDA and Triton.")
