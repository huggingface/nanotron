"""
Mock flash_attn.ops.triton.layer_norm module.
Provides layer normalization functions compatible with flash_attn API using PyTorch native ops.
"""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F


def layer_norm_fn(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    eps: float = 1e-6,
    dropout_p: float = 0.0,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = False,
    return_dropout_mask: bool = False,
) -> Union[Tensor, Tuple[Tensor, ...], Tuple[Tensor, Tensor, Tensor]]:
    """
    Layer normalization with optional residual connection and dropout.

    Args:
        x: Input tensor of shape [..., hidden_size]
        weight: Weight tensor of shape [hidden_size]
        bias: Optional bias tensor of shape [hidden_size]
        residual: Optional residual tensor to add before normalization
        eps: Epsilon for numerical stability
        dropout_p: Dropout probability
        prenorm: If True, return both normalized output and residual
        residual_in_fp32: If True, compute residual in fp32
        is_rms_norm: If True, use RMS normalization instead of layer norm
        return_dropout_mask: If True, return dropout mask

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if prenorm=True,
        or tuple of (normalized, residual, dropout_mask) if return_dropout_mask=True
    """
    original_dtype = x.dtype
    hidden_size = x.shape[-1]

    # Handle residual connection
    if residual is not None:
        if residual_in_fp32:
            residual = residual.to(torch.float32)
            x = x.to(torch.float32)
        x = x + residual

    # Apply dropout if needed
    dropout_mask = None
    if dropout_p > 0.0 and x.requires_grad:
        dropout_mask = torch.bernoulli(torch.full_like(x, 1 - dropout_p))
        x = x * dropout_mask / (1 - dropout_p)

    # Store residual for prenorm
    residual_out = x.to(torch.float32) if residual_in_fp32 else x.clone()

    # Compute normalization
    if is_rms_norm:
        # RMS normalization
        x_float = x.to(torch.float32)
        variance = x_float.pow(2).mean(-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + eps)
        output = (weight * x_normed).to(original_dtype)
    else:
        # Standard layer normalization
        output = F.layer_norm(x.to(torch.float32), (hidden_size,), weight.float(), bias.float() if bias is not None else None, eps)
        output = output.to(original_dtype)

    # Return appropriate outputs
    if return_dropout_mask:
        if dropout_mask is None:
            dropout_mask = torch.ones_like(x, dtype=torch.bool)
        return output, residual_out, dropout_mask
    elif prenorm:
        return output, residual_out
    else:
        return output


def rms_norm_fn(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    eps: float = 1e-6,
    dropout_p: float = 0.0,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    return_dropout_mask: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """
    RMS normalization with optional residual connection and dropout.

    This is a convenience wrapper around layer_norm_fn with is_rms_norm=True.
    """
    return layer_norm_fn(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        eps=eps,
        dropout_p=dropout_p,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        is_rms_norm=True,
        return_dropout_mask=return_dropout_mask,
    )
