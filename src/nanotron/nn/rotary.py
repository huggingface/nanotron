import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: float = 10000.0,
        interleaved: bool = True,
        seq_len_scaling_factor: float = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.interleaved = interleaved
        self.seq_len_scaling_factor = seq_len_scaling_factor

        # Generate inverse frequency buffer directly in the constructor
        self.register_buffer(
            "freqs_cis",
            1.0 / (base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float) / dim)),
            persistent=False,
        )

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
        else:
            seq_length = seq_length or self.max_seq_len
            positions = (
                torch.arange(seq_length, device=self.freqs_cis.device, dtype=self.freqs_cis.dtype) + position_offset
            )  # [seq_length]

        # Apply sequence length scaling if specified
        if self.seq_len_scaling_factor is not None:
            positions = positions / self.seq_len_scaling_factor

        # Compute position frequencies
        # TODO @nouamane: Using position_ids means we compute redundant embeddings for same positions. Only use them in SFT
        position_freqs = torch.outer(positions, self.freqs_cis)  # [seq_length, dim/2]

        # Organize embeddings based on interleaving strategy
        if not self.interleaved:
            embeddings = torch.cat((position_freqs, position_freqs), dim=-1)  # [b*s, dim] or [seq_length, dim]
        else:
            embeddings = torch.stack(
                (position_freqs.view(-1, 1), position_freqs.view(-1, 1)), dim=-1
            )  # [b*s*dim, 2] or [seq_length*dim, 2]
            embeddings = embeddings.view(position_freqs.shape[0], -1)  # [b*s, dim] or [seq_length, dim]

        return embeddings  # [b*s, dim] or [seq_length, dim]

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

    def apply_rotary_pos_emb(self, tensor, freqs, multi_latent_attention=False, mscale=1.0):
        """Apply rotary positional embedding to input tensor.

        Args:
            tensor (Tensor): Input tensor of shape [..., dim]
            freqs (Tensor, optional): Pre-computed position embeddings [..., dim] same or broadcastable to tensor
            multi_latent_attention (bool): Whether to use multi-latent attention
            mscale (float): Scaling factor for rotary embeddings

        Returns:
            Tensor: The input tensor after applying rotary positional embedding
        """
        rotary_dim = freqs.shape[-1]

        # Split the tensor for rotary embedding application
        rotary_part, pass_through_part = tensor[..., :rotary_dim], tensor[..., rotary_dim:]

        # Handle multi-latent attention
        if multi_latent_attention:
            x1 = rotary_part[..., 0::2]
            x2 = rotary_part[..., 1::2]
            rotary_part = torch.cat((x1, x2), dim=-1)

        # Get cosine and sine components with scaling
        cos_values = (torch.cos(freqs) * mscale).to(tensor.dtype)
        sin_values = (torch.sin(freqs) * mscale).to(tensor.dtype)

        # Apply rotary embedding
        rotated_tensor = (rotary_part * cos_values) + (self.rotate_half(rotary_part) * sin_values)

        # Concatenate with the pass-through part (if any)
        if pass_through_part.shape[-1] > 0:
            return torch.cat((rotated_tensor, pass_through_part), dim=-1)
        return rotated_tensor
