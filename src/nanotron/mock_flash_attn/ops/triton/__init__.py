"""Mock flash_attn.ops.triton module."""

from nanotron.mock_flash_attn.ops.triton.layer_norm import layer_norm_fn, rms_norm_fn

__all__ = ["layer_norm_fn", "rms_norm_fn"]
