"""
Mock grouped_gemm.ops module for local debugging without CUDA.
Provides stub implementations that raise errors when called.
"""

import torch
from typing import Optional


def gmm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
) -> torch.Tensor:
    """
    Mock grouped matrix multiplication.

    This operation requires CUDA and the grouped_gemm library.
    """
    raise NotImplementedError(
        "grouped_gemm.ops.gmm is not available in mock mode. "
        "Grouped GEMM requires CUDA. "
        "Please install grouped_gemm or use a different configuration."
    )


class GroupedGemm(torch.autograd.Function):
    """Mock GroupedGemm autograd function."""

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        trans_a: bool = False,
        trans_b: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "GroupedGemm.forward is not available in mock mode. "
            "Grouped GEMM requires CUDA."
        )

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "GroupedGemm.backward is not available in mock mode. "
            "Grouped GEMM requires CUDA."
        )
