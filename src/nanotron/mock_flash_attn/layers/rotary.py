"""
Mock flash_attn.layers.rotary module.
Provides rotary positional embedding implementation compatible with flash_attn API.
"""

from typing import Optional, Tuple, Union

import torch
from torch import nn, Tensor


def rotate_half(x: Tensor, interleaved: bool = False) -> Tensor:
    """Rotate half the hidden dims of the input."""
    if interleaved:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)
    else:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    interleaved: bool = False,
    inplace: bool = False,
    seqlen_offsets: Union[int, Tensor] = 0,
    cu_seqlens: Optional[Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> Tensor:
    """
    Apply rotary positional embedding to input tensor.

    Args:
        x: Input tensor of shape [..., seq_len, num_heads, head_dim] or [total, num_heads, head_dim]
        cos: Cosine tensor of shape [seq_len, head_dim//2] or [seq_len, head_dim]
        sin: Sine tensor of shape [seq_len, head_dim//2] or [seq_len, head_dim]
        interleaved: If True, use interleaved rotary embedding
        inplace: If True, modify x in place (ignored in mock, always creates new tensor)
        seqlen_offsets: Offset for sequence positions
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        max_seqlen: Maximum sequence length

    Returns:
        Tensor with rotary embedding applied
    """
    # Handle different cos/sin shapes
    if cos.dim() == 2 and cos.shape[-1] == x.shape[-1] // 2:
        # cos/sin are [seq_len, head_dim//2], need to expand
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

    # Get sequence length from x
    if x.dim() == 4:
        # [batch, seq_len, num_heads, head_dim]
        seq_len = x.shape[1]
        # Expand cos/sin for broadcasting: [1, seq_len, 1, head_dim]
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)
    elif x.dim() == 3:
        # [total, num_heads, head_dim] - variable length format
        if cu_seqlens is not None:
            # Handle variable length sequences
            total_len = x.shape[0]
            cos = cos[:total_len].unsqueeze(1)  # [total, 1, head_dim]
            sin = sin[:total_len].unsqueeze(1)
        else:
            seq_len = x.shape[0]
            cos = cos[:seq_len].unsqueeze(1)  # [seq_len, 1, head_dim]
            sin = sin[:seq_len].unsqueeze(1)
    else:
        raise ValueError(f"Unexpected input dimension: {x.dim()}")

    # Apply rotary embedding
    if interleaved:
        x_rotated = rotate_half(x, interleaved=True)
    else:
        x_rotated = rotate_half(x, interleaved=False)

    return x * cos + x_rotated * sin


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding module compatible with flash_attn API.

    This is a mock implementation that uses standard PyTorch operations.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: Optional[float] = None,
        pos_idx_in_fp32: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize RotaryEmbedding.

        Args:
            dim: Dimension of the rotary embedding (usually head_dim)
            base: Base for computing frequencies
            interleaved: If True, use interleaved rotary embedding
            scale_base: Base for scaling (for NTK-aware scaling)
            pos_idx_in_fp32: Whether to compute position indices in fp32
            device: Device for computation
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.pos_idx_in_fp32 = pos_idx_in_fp32

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Scale for dynamic NTK
        self.scale = None
        if scale_base is not None:
            scale = (
                (torch.arange(0, dim, 2, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            ) ** scale_base
            self.register_buffer("scale", scale, persistent=False)

        # Cache for cos/sin values
        self._seq_len_cached = 0
        self._cos_cached: Optional[Tensor] = None
        self._sin_cached: Optional[Tensor] = None
        self._cos_k_cached: Optional[Tensor] = None
        self._sin_k_cached: Optional[Tensor] = None

    def _compute_inv_freq(self, device: Optional[torch.device] = None) -> Tensor:
        """Compute inverse frequencies."""
        return 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        )

    def _update_cos_sin_cache(
        self,
        seqlen: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Update the cached cos/sin values."""
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen

            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq.to(device)
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq.to(device)

            freqs = torch.outer(t, inv_freq)

            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=device) - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device) ** power.unsqueeze(-1)
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        qkv: Tensor,
        kv: Optional[Tensor] = None,
        seqlen_offset: Union[int, Tensor] = 0,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Apply rotary embedding to query and key tensors.

        Args:
            qkv: Query tensor or packed QKV tensor
                 Shape: [batch, seqlen, num_heads, head_dim] or [total, num_heads, head_dim]
                 or [batch, seqlen, 3, num_heads, head_dim]
            kv: Optional key-value tensor for separate Q and KV
            seqlen_offset: Offset for sequence position (for inference with KV cache)
            cu_seqlens: Cumulative sequence lengths for variable length sequences
            max_seqlen: Maximum sequence length

        Returns:
            Tensor(s) with rotary embedding applied
        """
        if qkv.dim() == 5:
            # Packed QKV: [batch, seqlen, 3, num_heads, head_dim]
            seqlen = qkv.shape[1]
            q, k, v = qkv.unbind(dim=2)
        elif qkv.dim() == 4:
            # Standard format: [batch, seqlen, num_heads, head_dim]
            seqlen = qkv.shape[1]
            if kv is not None:
                q = qkv
                k, v = kv.unbind(dim=2) if kv.dim() == 5 else (kv, None)
            else:
                # Just query
                q = qkv
                k = None
                v = None
        elif qkv.dim() == 3:
            # Variable length: [total, num_heads, head_dim]
            seqlen = max_seqlen if max_seqlen is not None else qkv.shape[0]
            q = qkv
            k = kv if kv is not None else None
            v = None
        else:
            raise ValueError(f"Unexpected qkv dimension: {qkv.dim()}")

        # Update cache
        if isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype)
        else:
            self._update_cos_sin_cache(
                seqlen + seqlen_offset.max().item(), device=qkv.device, dtype=qkv.dtype
            )

        # Apply rotary embedding
        if self.scale is None:
            cos = self._cos_cached
            sin = self._sin_cached
        else:
            cos = self._cos_cached
            sin = self._sin_cached

        q_rotated = apply_rotary_emb(
            q, cos, sin,
            interleaved=self.interleaved,
            seqlen_offsets=seqlen_offset,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if k is not None:
            if self.scale is not None:
                cos_k = self._cos_k_cached
                sin_k = self._sin_k_cached
            else:
                cos_k = cos
                sin_k = sin

            k_rotated = apply_rotary_emb(
                k, cos_k, sin_k,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

            if qkv.dim() == 5:
                # Return packed QKV
                return torch.stack([q_rotated, k_rotated, v], dim=2)
            elif v is not None:
                return q_rotated, k_rotated, v
            else:
                return q_rotated, k_rotated

        return q_rotated
