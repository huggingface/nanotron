"""
Mock ring_attention module for local debugging without CUDA/Triton.
Provides stub implementations that raise errors when called, since ring attention
requires distributed communication that doesn't work without a proper GPU setup.
"""

import torch
from typing import Optional, Tuple


def ring_flash_attn_varlen_func(
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
    Mock ring flash attention varlen function.

    Ring attention requires distributed communication across GPUs,
    which is not available in the mock environment.
    """
    raise NotImplementedError(
        "ring_flash_attn_varlen_func is not available in mock mode. "
        "Ring attention requires CUDA and distributed GPU communication. "
        "Please use 'flash_attention_2', 'sdpa', or 'flex_attention' instead."
    )


class RingComm:
    """Mock RingComm class."""
    def __init__(self, process_group):
        raise NotImplementedError("RingComm requires distributed GPU communication.")


class RingFlashAttnVarlenFunc(torch.autograd.Function):
    """Mock RingFlashAttnVarlenFunc class."""
    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError("RingFlashAttnVarlenFunc requires CUDA and Triton.")

    @staticmethod
    def backward(ctx, *args):
        raise NotImplementedError("RingFlashAttnVarlenFunc requires CUDA and Triton.")


def ring_flash_attn_varlen_forward(*args, **kwargs):
    """Mock ring_flash_attn_varlen_forward."""
    raise NotImplementedError("ring_flash_attn_varlen_forward requires CUDA and Triton.")


def ring_flash_attn_varlen_backward(*args, **kwargs):
    """Mock ring_flash_attn_varlen_backward."""
    raise NotImplementedError("ring_flash_attn_varlen_backward requires CUDA and Triton.")


# Triton utility stubs
def flatten_varlen_lse(*args, **kwargs):
    """Mock flatten_varlen_lse."""
    raise NotImplementedError("flatten_varlen_lse requires Triton.")


def unflatten_varlen_lse(*args, **kwargs):
    """Mock unflatten_varlen_lse."""
    raise NotImplementedError("unflatten_varlen_lse requires Triton.")


def update_out_and_lse(*args, **kwargs):
    """Mock update_out_and_lse."""
    raise NotImplementedError("update_out_and_lse requires Triton.")


def get_default_args(func):
    """Mock get_default_args - returns empty dict."""
    return {}


class AllGatherComm:
    """Mock AllGatherComm class."""
    def __init__(self, group=None):
        raise NotImplementedError("AllGatherComm requires distributed GPU communication.")
