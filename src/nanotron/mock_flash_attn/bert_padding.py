"""
Mock flash_attn.bert_padding module.
Provides functions for padding/unpadding sequences for efficient attention computation.
"""

from typing import Tuple

import torch
from torch import Tensor


def unpad_input(
    hidden_states: Tensor,
    attention_mask: Tensor,
    unused_mask: Tensor = None,
) -> Tuple[Tensor, Tensor, Tensor, int]:
    """
    Remove padding from hidden states based on attention mask.

    Args:
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
        attention_mask: Boolean mask of shape [batch_size, seq_len], True for valid tokens
        unused_mask: Unused parameter for API compatibility

    Returns:
        hidden_states_unpad: Unpadded hidden states [total_tokens, hidden_size]
        indices: Indices of valid tokens in the flattened batch
        cu_seqlens: Cumulative sequence lengths [batch_size + 1]
        max_seqlen: Maximum sequence length in the batch
    """
    # Get sequence lengths from attention mask
    seqlens = attention_mask.sum(dim=1, dtype=torch.int32)
    max_seqlen = seqlens.max().item()

    # Calculate cumulative sequence lengths
    cu_seqlens = torch.zeros(attention_mask.shape[0] + 1, dtype=torch.int32, device=attention_mask.device)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)

    # Get indices of valid tokens
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()

    # Unpad hidden states
    hidden_states_unpad = hidden_states.view(-1, hidden_states.shape[-1])[indices]

    return hidden_states_unpad, indices, cu_seqlens, max_seqlen


def unpad_input_for_concatenated_sequences(
    hidden_states: Tensor,
    attention_mask: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, int]:
    """
    Unpad input for concatenated sequences (same as unpad_input).

    Args:
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
        attention_mask: Boolean mask of shape [batch_size, seq_len]

    Returns:
        Same as unpad_input
    """
    return unpad_input(hidden_states, attention_mask)


def pad_input(
    hidden_states: Tensor,
    indices: Tensor,
    batch_size: int,
    seq_len: int,
) -> Tensor:
    """
    Pad hidden states back to original shape.

    Args:
        hidden_states: Unpadded hidden states [total_tokens, hidden_size] or [total_tokens, num_heads, head_dim]
        indices: Indices of valid tokens in the flattened batch
        batch_size: Original batch size
        seq_len: Original sequence length

    Returns:
        Padded hidden states [batch_size, seq_len, hidden_size] or [batch_size, seq_len, num_heads, head_dim]
    """
    if hidden_states.dim() == 2:
        # [total_tokens, hidden_size]
        output = torch.zeros(
            batch_size * seq_len,
            hidden_states.shape[-1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        output[indices] = hidden_states
        return output.view(batch_size, seq_len, hidden_states.shape[-1])
    elif hidden_states.dim() == 3:
        # [total_tokens, num_heads, head_dim]
        output = torch.zeros(
            batch_size * seq_len,
            hidden_states.shape[1],
            hidden_states.shape[2],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        output[indices] = hidden_states
        return output.view(batch_size, seq_len, hidden_states.shape[1], hidden_states.shape[2])
    else:
        raise ValueError(f"Unexpected hidden_states dimension: {hidden_states.dim()}")


def index_first_axis(x: Tensor, indices: Tensor) -> Tensor:
    """
    Index the first axis of a tensor.

    Args:
        x: Input tensor [total, ...]
        indices: Indices to select

    Returns:
        Selected tensor [len(indices), ...]
    """
    return x[indices]


def index_first_axis_residual(x: Tensor, indices: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Index the first axis and also return residual indices.

    Args:
        x: Input tensor
        indices: Indices to select

    Returns:
        Tuple of (selected tensor, original tensor)
    """
    return x[indices], x


class IndexFirstAxis(torch.autograd.Function):
    """Autograd function for indexing first axis."""

    @staticmethod
    def forward(ctx, x: Tensor, indices: Tensor) -> Tensor:
        ctx.save_for_backward(indices)
        ctx.first_axis_dim = x.shape[0]
        return x[indices]

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        (indices,) = ctx.saved_tensors
        grad_input = torch.zeros(
            ctx.first_axis_dim,
            *grad_output.shape[1:],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input[indices] = grad_output
        return grad_input, None


class IndexPutFirstAxis(torch.autograd.Function):
    """Autograd function for index_put on first axis."""

    @staticmethod
    def forward(ctx, values: Tensor, indices: Tensor, first_axis_dim: int) -> Tensor:
        ctx.save_for_backward(indices)
        output = torch.zeros(
            first_axis_dim,
            *values.shape[1:],
            device=values.device,
            dtype=values.dtype,
        )
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None]:
        (indices,) = ctx.saved_tensors
        return grad_output[indices], None, None
