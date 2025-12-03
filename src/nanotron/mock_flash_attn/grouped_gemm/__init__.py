"""
Mock grouped_gemm package for local debugging without CUDA.
Provides stub implementations that raise errors when called.
"""

from nanotron.mock_flash_attn.grouped_gemm import ops

__all__ = ["ops"]
