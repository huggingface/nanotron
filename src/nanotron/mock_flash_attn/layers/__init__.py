"""Mock flash_attn.layers module."""

from nanotron.mock_flash_attn.layers.rotary import RotaryEmbedding, apply_rotary_emb

__all__ = ["RotaryEmbedding", "apply_rotary_emb"]
