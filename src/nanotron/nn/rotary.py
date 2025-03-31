import torch
from flash_attn.layers.rotary import apply_rotary_emb as flash_apply_rotary_emb
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: float = 10000.0,
        interleaved: bool = False,
        seq_len_scaling_factor: float = None,
        fused: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len  # we set it as max_position_embeddings in init. but we ignore it of we provide `seq_length` in forward
        self.interleaved = interleaved
        self.seq_len_scaling_factor = seq_len_scaling_factor
        self.fused = fused
        # Generate inverse frequency buffer directly in the constructor
        self.register_buffer(
            "freqs_cis",
            1.0 / (base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float) / dim)),
            persistent=False,
        )
        # These are caches that are recomputed during inference
        self.register_buffer("cos_values", None, persistent=False)
        self.register_buffer("sin_values", None, persistent=False)

        assert self.freqs_cis.device.type == "cuda"

    def forward(self, seq_length=None, position_offset=0, position_ids=None):
        """Generate rotary position embeddings.

        Args:
            seq_length (int, optional): Sequence length to use. Defaults to max_seq_len.
            position_offset (int, optional): Offset for position ids. Defaults to 0.
            position_ids (Tensor, optional): Position ids to use. Defaults to None. [batch_size, seq_length]

        Returns:
            Tensor: Rotary embeddings of shape [seq_length, 1, 1, dim]
        """
        self.freqs_cis = self.freqs_cis.to(torch.float)  # TODO @nouamane: Fix using `DTypeInvariantTensor` ...

        # Generate position indices
        if position_ids is not None:
            assert seq_length is None, "seq_length must be None if position_ids is provided"
            assert position_offset == 0, "position_offset must be 0 if position_ids is provided"
            # TODO @nouamane: Using position_ids means we compute redundant embeddings for same positions
            positions = position_ids.to(device=self.freqs_cis.device, dtype=self.freqs_cis.dtype)  # [b*s]
            self.max_seq_len = positions.max() + 1
        else:
            seq_length = seq_length or self.max_seq_len
            positions = (
                torch.arange(seq_length, device=self.freqs_cis.device, dtype=self.freqs_cis.dtype) + position_offset
            )  # [seq_length]
            self.max_seq_len = seq_length

        # Apply sequence length scaling if specified
        if self.seq_len_scaling_factor is not None:
            positions = positions / self.seq_len_scaling_factor

        # Compute position frequencies
        # TODO @nouamane: Using position_ids means we compute redundant embeddings for same positions. Only use them in SFT
        position_freqs = torch.outer(positions, self.freqs_cis)  # [seq_length, dim/2]

        # Organize embeddings based on interleaving strategy
        if self.fused:
            embeddings = position_freqs  # [b*s, dim/2] or [seq_length, dim/2]
        else:
            if not self.interleaved:
                embeddings = torch.cat((position_freqs, position_freqs), dim=-1)  # [b*s, dim] or [seq_length, dim]
            else:
                embeddings = torch.stack(
                    (position_freqs.view(-1, 1), position_freqs.view(-1, 1)), dim=-1
                )  # [b*s*dim, 2] or [seq_length*dim, 2]
                embeddings = embeddings.view(position_freqs.shape[0], -1)  # [b*s, dim] or [seq_length, dim]

        return embeddings  # [b*s, dim] or [seq_length, dim] or [b*s, dim/2] or [seq_length, dim/2]

    def rotate_half(self, x):
        """Rotates half the hidden dimensions of the input tensor."""
        if self.interleaved:
            even_dims = x[..., ::2]
            odd_dims = x[..., 1::2]
            return torch.cat((-odd_dims, even_dims), dim=-1)
        else:
            first_half = x[..., : x.shape[-1] // 2]
            second_half = x[..., x.shape[-1] // 2 :]
            return torch.cat((-second_half, first_half), dim=-1)

    def apply_rotary_pos_emb(self, tensor, freqs, multi_latent_attention=False, mscale=1.0, seq_length=None):
        """Apply rotary positional embedding to input tensor.

        Args:
            tensor (Tensor): Input tensor of shape [..., dim] if not fused, [batch_size*seq_length, nheads, dim] if fused
            freqs (Tensor, optional): Pre-computed position embeddings [..., dim] same or broadcastable to tensor
            multi_latent_attention (bool): Whether to use multi-latent attention
            mscale (float): Scaling factor for rotary embeddings

        Returns:
            Tensor: The input tensor after applying rotary positional embedding
        """
        rotary_dim = freqs.shape[-1]

        # Split the tensor for rotary embedding application
        if freqs.shape[-1] != rotary_dim:
            rotary_part, pass_through_part = tensor[..., :rotary_dim], tensor[..., rotary_dim:]
        else:
            rotary_part, pass_through_part = tensor, None

        # Handle multi-latent attention
        if multi_latent_attention:
            x1 = rotary_part[..., 0::2]
            x2 = rotary_part[..., 1::2]
            rotary_part = torch.cat((x1, x2), dim=-1)

        # Get cosine and sine components with scaling
        if self.cos_values is None:
            self.cos_values = (torch.cos(freqs) * mscale).to(tensor.dtype)
            self.sin_values = (torch.sin(freqs) * mscale).to(tensor.dtype)

        # Apply rotary embedding
        rotary_part = rotary_part.view(
            -1, seq_length, rotary_part.shape[1], rotary_part.shape[2]
        )  # [b, s, nheads, dim/2]
        if self.fused:
            rotated_tensor = flash_apply_rotary_emb(
                rotary_part, self.cos_values, self.sin_values, interleaved=self.interleaved, inplace=True
            )
            # TODO @nouamane: support cu_seqlens from position_ids
        else:
            rotated_tensor = (rotary_part * self.cos_values.unsqueeze(1)) + (
                self.rotate_half(rotary_part) * self.sin_values.unsqueeze(1)
            )

        # Concatenate with the pass-through part (if any)
        if pass_through_part is not None and pass_through_part.shape[-1] > 0:
            return torch.cat((rotated_tensor, pass_through_part), dim=-1)
        return rotated_tensor
