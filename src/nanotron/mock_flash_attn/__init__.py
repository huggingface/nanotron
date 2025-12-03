"""
Mock flash_attn package for local debugging without CUDA.
This replaces flash attention with PyTorch's native scaled dot product attention.

Usage:
    # Auto-enable (recommended): Call this before importing any nanotron modules
    from nanotron.mock_flash_attn import install_mock_flash_attn
    install_mock_flash_attn()

    # Or manually:
    import sys
    sys.modules['flash_attn'] = __import__('nanotron.mock_flash_attn', fromlist=[''])
"""

import sys
import torch

__version__ = "2.5.0"  # Mock version to satisfy version checks

from nanotron.mock_flash_attn import bert_padding
from nanotron.mock_flash_attn import flash_attn_interface
from nanotron.mock_flash_attn import layers
from nanotron.mock_flash_attn import ops
from nanotron.mock_flash_attn import modules
from nanotron.mock_flash_attn import ring_attention
from nanotron.mock_flash_attn import llama3_ring_attention
from nanotron.mock_flash_attn import grouped_gemm

__all__ = [
    "bert_padding",
    "flash_attn_interface",
    "layers",
    "ops",
    "modules",
    "ring_attention",
    "llama3_ring_attention",
    "grouped_gemm",
    "__version__",
    "install_mock_flash_attn",
    "is_mock_enabled",
]

_mock_installed = False


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_mock_enabled() -> bool:
    """Check if mock flash_attn is currently installed."""
    return _mock_installed or "flash_attn" in sys.modules and sys.modules["flash_attn"].__name__.startswith("nanotron.mock")


def install_mock_flash_attn(force: bool = False) -> bool:
    """
    Install mock flash_attn package into sys.modules.

    This function automatically installs the mock if CUDA is not available,
    unless force=True is specified.

    Args:
        force: If True, install mock even if CUDA is available

    Returns:
        True if mock was installed, False if real flash_attn should be used
    """
    global _mock_installed

    if _mock_installed:
        return True

    # Check if we should use the mock
    should_mock = force or not is_cuda_available()

    if not should_mock:
        return False

    # Install the mock package
    import nanotron.mock_flash_attn as mock_pkg

    sys.modules["flash_attn"] = mock_pkg
    sys.modules["flash_attn.bert_padding"] = mock_pkg.bert_padding
    sys.modules["flash_attn.flash_attn_interface"] = mock_pkg.flash_attn_interface
    sys.modules["flash_attn.layers"] = mock_pkg.layers
    sys.modules["flash_attn.layers.rotary"] = mock_pkg.layers.rotary
    sys.modules["flash_attn.ops"] = mock_pkg.ops
    sys.modules["flash_attn.ops.triton"] = mock_pkg.ops.triton
    sys.modules["flash_attn.ops.triton.layer_norm"] = mock_pkg.ops.triton.layer_norm
    sys.modules["flash_attn.modules"] = mock_pkg.modules
    sys.modules["flash_attn.modules.mha"] = mock_pkg.modules.mha
    sys.modules["nanotron.nn.ring_attention"] = mock_pkg.ring_attention
    sys.modules["nanotron.nn.llama3_ring_attention"] = mock_pkg.llama3_ring_attention
    sys.modules["grouped_gemm"] = mock_pkg.grouped_gemm
    sys.modules["grouped_gemm.ops"] = mock_pkg.grouped_gemm.ops

    _mock_installed = True
    print(f"[nanotron] Mock flash_attn installed (CUDA available: {is_cuda_available()})")
    return True


# Auto-install if CUDA is not available
# This allows the mock to work automatically when this module is imported
if not is_cuda_available():
    install_mock_flash_attn()
